import os
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


def main():
    args = utils.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    utils.set_seed(args.seed)
    iteration = args.i
    print(f"--- Starting TPT Iteration {iteration + 1}/{args.tpt_iterations} ---")
    
    if args.train == 'deepscaler':
        train_dataset = utils.load_processed_deepscaler(split='train', max_samples=args.train_samples)
    elif args.train == 'aime':
        train_dataset = utils.load_processed_aime_23()
    print(args.train)
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
        if not os.path.exists(os.path.join(args.output_dir, f'solutions_quant_iter_{iteration}.json')): 
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
            if args.awq:
                baseline_quantized_path = '/mnt/cephfs/sumin/llm-awq/awq_cache/qwen3_fake_quant'
            else:
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
        return
    
if __name__ == "__main__":
    main()