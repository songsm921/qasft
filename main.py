import os
import json
import utils as utils
import think as think
import prune as prune
import train as train
import quant as quant
import eval_utils as eval_utils
import logging
import torch
import glob
import numpy as np
from accelerate import Accelerator
from multiprocessing import Queue
import datasets
from transformers import TrainingArguments
from accelerate import InitProcessGroupKwargs
from datetime import timedelta
from qat_module import quantize_model_weights
def main_accelerate(args):
    ipg_handler = InitProcessGroupKwargs(
            timeout=timedelta(seconds=86400)
    )
    accelerator = Accelerator(
        kwargs_handlers=[ipg_handler],
    )
    
    # Set seed
    utils.set_seed(args.seed)
    
    # Only main process creates output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Wait for main process to create directory
    accelerator.wait_for_everyone()
    
    # Baseline evaluation
    if args.baseline:
        if accelerator.is_main_process:
            print('Starting Baseline Evaluation')
        
        model, tokenizer = utils.load_model(args, multi_gpu=False)
        eval_dataset = utils.load_aime_2024()
        
        baseline_results = eval_utils.evaluate_accelerate(model, tokenizer, eval_dataset, args, k_values=[1], acc=accelerator)
        
        if accelerator.is_main_process and baseline_results is not None:
            print(f"Baseline results: {baseline_results}")
            all_results = {
                "baseline": baseline_results,
                "iterations": []
            }
            results_file = os.path.join(args.output_dir, "all_results.json")
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
        
        # Handle quantization if needed
        if args.quant:
            quant.quant_model(model, args)
            
            if accelerator.is_main_process:
                print('Starting Quantized Baseline Evaluation')
            
            quant_baseline_results = eval_utils.evaluate_accelerate(model, tokenizer, eval_dataset, args, k_values=[1], acc=accelerator)
            
            if accelerator.is_main_process and quant_baseline_results is not None:
                print(f"Quantized Baseline results: {quant_baseline_results}")
                all_results = {
                    "Quantized baseline": quant_baseline_results,
                    "iterations": []
                }
                results_file = os.path.join(args.output_dir, "all_results.json")
                with open(results_file, 'a') as f:
                    json.dump(all_results, f, indent=2)
        
        del model, tokenizer
        torch.cuda.empty_cache()
    
    # Wait for baseline to complete
    accelerator.wait_for_everyone()
    
    # TPT Iterations
    # if args.quant and args.quant_type == 'qat':
    #     model, qat_tokenizer = utils.load_model(args, multi_gpu=False)
    #     from qat_module import quantize_model_weights
    #     qat_model = quantize_model_weights(model, bits=args.bit, per_channel=True)
    #     print(qat_model)
    #     del model
    eval_dataset = utils.load_aime_2024()
    for iteration in range(args.tpt_iterations):
        if accelerator.is_main_process:
            print(f"\n--- Starting TPT Iteration {iteration + 1}/{args.tpt_iterations} ---\n")
        
        train_dataset = utils.load_processed_deepscaler(split='train', max_samples=args.train_samples)
        
        
        if iteration > 0:
            if args.quant:
                args.quant_path = os.path.join(args.output_dir, f"quant_model_iteration_{iteration-1}")
            else:
                args.model_path = os.path.join(args.output_dir, f"tpt_model_iteration_{iteration-1}")

        # Think Stage
        if accelerator.is_main_process:
            print("\n--- Think Stage (Accelerate) ---\n")
        
        # Load model for think phase
        model, tokenizer = utils.load_model(args, multi_gpu=False)
        
        # Run think phase
        solutions_file = os.path.join(args.output_dir, f"solutions_iter_{iteration}.json")
        think.think_phase_accelerate(model, tokenizer, train_dataset, args, output_file=solutions_file, acc=accelerator)
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache()
        
        # Wait for all processes to finish think phase
        accelerator.wait_for_everyone()
        
        # Merge solutions (only main process)
        if accelerator.is_main_process:
            rank_files = glob.glob(
                os.path.join(args.output_dir, f"solutions_iter_{iteration}_rank*.json")
            )
            all_solutions = []
            for fp in sorted(rank_files):
                with open(fp, 'r') as f:
                    sols = json.load(f)
                all_solutions.extend(sols)
            
            merged_file = os.path.join(args.output_dir, f"solutions_iter_{iteration}.json")
            with open(merged_file, 'w') as f:
                json.dump(all_solutions, f, indent=2)
            
            # Clean up rank files
            for fp in rank_files:
                os.remove(fp)
        accelerator.wait_for_everyone()
        # Handle quantized think phase if needed
        if args.quant:
            if accelerator.is_main_process:
                print("\n--- Think Stage (Quantized) ---\n")
            if iteration == 0:
                model, tokenizer = utils.load_model(args, multi_gpu=False)
                quant.quant_model(model, args)
            else:
                model, tokenizer = utils.load_model_quant(args, iteration, multi_gpu=False)
                quant.quant_model(model, args)
            
            solutions_quant_file = os.path.join(args.output_dir, f"solutions_quant_iter_{iteration}.json")
            think.think_phase_accelerate(model, tokenizer, train_dataset, args, output_file=solutions_quant_file, acc=accelerator)
            del model, tokenizer
            torch.cuda.empty_cache()
            
            accelerator.wait_for_everyone()
            
            # Merge quantized solutions (only main process)
            if accelerator.is_main_process:
                rank_files = glob.glob(
                    os.path.join(args.output_dir, f"solutions_quant_iter_{iteration}_rank*.json")
                )
                all_solutions_quant = []
                for fp in sorted(rank_files):
                    with open(fp, 'r') as f:
                        sols = json.load(f)
                    all_solutions_quant.extend(sols)
                
                merged_file_quant = os.path.join(args.output_dir, f"solutions_quant_iter_{iteration}.json")
                with open(merged_file_quant, 'w') as f:
                    json.dump(all_solutions_quant, f, indent=2)
                
                # Clean up rank files
                for fp in rank_files:
                    os.remove(fp)
        
        accelerator.wait_for_everyone()
        
        # Prune Stage (only main process)
        if accelerator.is_main_process:
            print("\n--- Prune Stage ---\n")
            
            merged_file = os.path.join(args.output_dir, f"solutions_iter_{iteration}.json")
            solutions = utils.load_solutions(merged_file)
            
            if args.quant:
                merged_file_quant = os.path.join(args.output_dir, f"solutions_quant_iter_{iteration}.json")
                solutions_quant = utils.load_solutions(merged_file_quant)
            
            pruned_file = os.path.join(args.output_dir, f"pruned_solutions_iter_{iteration}.json")
            
            if args.quant:
                pruned_solutions = prune.prune_solutions_consider_quant(
                    solutions,
                    solutions_quant,
                    args,
                    strategy="correct_only", 
                    max_samples_per_problem=1
                )
            else:
                pruned_solutions = prune.prune_solutions(
                    solutions, 
                    args,
                    strategy="correct_only", 
                    max_samples_per_problem=1
                )
            
            prune.save_pruned_data(pruned_solutions, pruned_file)
        
        accelerator.wait_for_everyone()
        
        # Train Stage (only main process for now - could be distributed later)
        # if accelerator.is_main_process:
        if accelerator.is_main_process:
            print("\n--- Train Stage ---\n")
            
        pruned_file = os.path.join(args.output_dir, f"pruned_solutions_iter_{iteration}.json")
        pruned_solutions = utils.load_solutions(pruned_file)
        # if args.quant_type != 'qat':
        if args.quant:
            if iteration == 0:
                model, tokenizer = utils.load_model(args)
            else:
                model, tokenizer = utils.load_model_quant(args, iteration, multi_gpu=False)
            if args.quant_type == 'qat':
                model = quantize_model_weights(model, bits=args.bit, per_channel=True)
                if accelerator.is_main_process:
                    print(model)
            else:
                quant.quant_model(model, args)

            
        train.train_phase(model, tokenizer, pruned_solutions, iteration, args, accelerator)
                # print('here')
        accelerator.wait_for_everyone()
        # utils.verify_gradient_sync(model, accelerator)
        
        # Evaluation Stage
        if accelerator.is_main_process:
            print('Starting Iteration Evaluation')
        
        # Load trained model for evaluation
        if args.quant:
            model, tokenizer = utils.load_model_quant(args, iteration, False)
            quant.quant_model(model, args)
        else:
            model, tokenizer = utils.load_model(args)
        iter_results = eval_utils.evaluate_accelerate(model, tokenizer, eval_dataset, args, k_values=[1],acc=accelerator)
        
        if accelerator.is_main_process and iter_results is not None:
            print(f"Iteration results: {iter_results}")
            all_results = {
                "baseline": iter_results,
                "iterations": iteration + 1,
                "quant": 'True' if args.quant else 'False'
            }
            results_file = os.path.join(args.output_dir, "all_results.json")
            with open(results_file, 'a') as f:
                json.dump(all_results, f, indent=2)
        
        del model, tokenizer
        torch.cuda.empty_cache()
        
        accelerator.wait_for_everyone()

if __name__ == "__main__":
    args = utils.parse_args()
    main_accelerate(args)
    
    
    
    