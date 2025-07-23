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

def main():
    args = utils.parse_args()
    utils.set_seed(args.seed)
    iteration = args.i
    
    if not os.path.exists(os.path.join(args.output_dir, f'temporary_eval_iter_{iteration}')):
        print('Starting Iteration Evaluation')
        eval_dataset = utils.load_aime_2024()
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
        else:
            temporary_eval_path = os.path.join(args.output_dir, f"tpt_model_iteration_{iteration}")

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
    main()