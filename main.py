import os
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"
import json
import time
import utils as utils
import think as think
import think_mk2 as think_mk2
import prune as prune
import train as train
import quant as quant
import eval_utils as eval_utils
import logging
import torch
import glob
import numpy as np
# from accelerate import Accelerator
# from multiprocessing import Queue
import datasets
from transformers import TrainingArguments
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig,
)
# from accelerate import InitProcessGroupKwargs
from datetime import timedelta
from qat_module import quantize_model_weights
def main_accelerate(args):
    utils.set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Baseline evaluation
    if args.baseline:
        
        print('Starting Baseline Evaluation')
        
        # model, tokenizer = utils.load_model(args, multi_gpu=False)
        eval_dataset = utils.load_aime_2024()
        
        baseline_results, correction = eval_utils.evaluate_accelerate(args.model_path,eval_dataset, args, k_values=[1])
        
        if baseline_results is not None:
            print(f"Baseline results: {baseline_results}")
            all_results = {
                "baseline": baseline_results,
                "correct#": correction['correct'],
                "incorrect#": correction['incorrect'],
                "iterations": []
            }
            results_file = os.path.join(args.output_dir, "all_results.json")
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
        
        # Handle quantization if needed
        if args.quant:
            model, tokenizer = utils.load_model(args, multi_gpu=False)
            quant.quant_model(model, args)
            baseline_quantized_path = os.path.join(args.output_dir, f"baseline_quantized")
            model.save_pretrained(baseline_quantized_path)
            tokenizer.save_pretrained(baseline_quantized_path)
            del model
            print('Starting Quantized Baseline Evaluation')
            
            quant_baseline_results, correction = eval_utils.evaluate_accelerate(baseline_quantized_path ,eval_dataset, args, k_values=[1])
            
            if quant_baseline_results is not None:
                print(f"Quantized Baseline results: {quant_baseline_results}")
                all_results = {
                    "Quantized baseline": quant_baseline_results,
                    "correct#": correction['correct'],
                    "incorrect#": correction['incorrect'],
                    "iterations": []
                }
                results_file = os.path.join(args.output_dir, "all_results.json")
                with open(results_file, 'a') as f:
                    json.dump(all_results, f, indent=2)
        
        torch.cuda.empty_cache()
    
    
    eval_dataset = utils.load_aime_2024()
    _, tokenizer_t = utils.load_model(args, multi_gpu=False)
    # model_t, tokenizer_t = utils.load_model(args, multi_gpu=False)
    # model_t.to('cpu')
    for iteration in range(args.tpt_iterations):
        print(f"--- Starting TPT Iteration {iteration + 1}/{args.tpt_iterations} ---")
        
        train_dataset = utils.load_processed_deepscaler(split='train', max_samples=args.train_samples)
        
        if iteration > 0:
            args.quant_path = os.path.join(args.output_dir, f"quant_model_iteration_{iteration-1}")
        if not os.path.exists(os.path.join(args.output_dir, f'solutions_iter_{iteration}.json')): 
            print("\n--- Think Stage (Accelerate) ---\n")
            
            # Load model for think phase
            # 
            
            # Run think phase
            solutions_file = os.path.join(args.output_dir, f"solutions_iter_{iteration}.json")
            start_time = time.perf_counter()
            think_mk2.think_phase_optimized(args.model_path, train_dataset, args, output_file=solutions_file)
            end_time = time.perf_counter()
            elapsed_seconds = end_time - start_time
            elapsed_minutes = elapsed_seconds / 60
            print(f"Think Phase Time: {elapsed_minutes:.2f}분 ({elapsed_seconds:.2f}초)")
            if args.quant:
                print("\n--- Think Stage (Quantized) ---\n")
                if iteration > 0:
                    model, tokenizer = utils.load_model_quant(args, iteration, multi_gpu=False)
                    quant.quant_model(model, args)
                    # model, tokenizer = utils.load_model(args, multi_gpu=False)
                    # quant.quant_model(model, args)

                    temporary_quantized_path = os.path.join(args.output_dir, f"temporary_quantized_iter_{iteration}") 
                    model.save_pretrained(temporary_quantized_path)
                    tokenizer.save_pretrained(temporary_quantized_path)
                    del model
                    torch.cuda.empty_cache()
                    print('Saved quantized parameter.')
                baseline_quantized_path = os.path.join(args.output_dir, f"baseline_quantized")
                solutions_quant_file = os.path.join(args.output_dir, f"solutions_quant_iter_{iteration}.json")
                start_time = time.perf_counter()
                think_mk2.think_phase_optimized(temporary_quantized_path if iteration > 0 else baseline_quantized_path ,  train_dataset, args, output_file=solutions_quant_file)
                end_time = time.perf_counter()
                elapsed_seconds = end_time - start_time
                elapsed_minutes = elapsed_seconds / 60
                print(f"Think Phase Time (Quant): {elapsed_minutes:.2f}분 ({elapsed_seconds:.2f}초)")

            print("\n--- Prune Stage ---\n")
        if not os.path.exists(os.path.join(args.output_dir, f'pruned_solutions_iter_{iteration}.json')):       
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
                        max_samples_per_problem=args.N
                )
            else:
                pruned_solutions = prune.prune_solutions(
                        solutions, 
                        args,
                        strategy="correct_only", 
                        max_samples_per_problem=args.N
                )
                
            prune.save_pruned_data(pruned_solutions, pruned_file)
            
        
        # Train Stage (only main process for now - could be distributed later)
        # if accelerator.is_main_process:
        if not os.path.exists(os.path.join(args.output_dir, f'quant_model_iteration_{iteration}')):
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
                    print(model)
                else:
                    quant.quant_model(model, args)

                
            train.train_phase(model, tokenizer, pruned_solutions, iteration, args)
        
        # Evaluation Stage
        if not os.path.exists(os.path.join(args.output_dir, f'temporary_eval_iter_{iteration}')):
            print('Starting Iteration Evaluation')
            
            # Load trained model for evaluation
            if args.quant:
                path = os.path.join(args.output_dir, f"quant_model_iteration_{iteration}")
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    trust_remote_code = True,
                    torch_dtype = torch.bfloat16,
                    # attn_implementation="eager", # flash_attention_2
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    path,
                    trust_remote_code = True,
                    padding_side='left'
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                quant.quant_model(model, args)
                temporary_eval_path = os.path.join(args.output_dir, f"temporary_eval_iter_{iteration}") 
                model.save_pretrained(temporary_eval_path)
                tokenizer.save_pretrained(temporary_eval_path)
                del model

            iter_results, correction = eval_utils.evaluate_accelerate(temporary_eval_path, eval_dataset, args, k_values=[1])
            
            if iter_results is not None:
                print(f"Iteration results: {iter_results}")
                all_results = {
                    "baseline": iter_results,
                    "correct#": correction['correct'],
                    "incorrect#": correction['incorrect'],
                    "iterations": iteration + 1,
                    "quant": 'True' if args.quant else 'False'
                }
                results_file = os.path.join(args.output_dir, "all_results.json")
                with open(results_file, 'a') as f:
                    json.dump(all_results, f, indent=2)

        

if __name__ == "__main__":
    args = utils.parse_args()
    main_accelerate(args)
    
    
    
    