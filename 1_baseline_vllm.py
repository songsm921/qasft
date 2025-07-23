import os
import json
import utils as utils
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
from datetime import timedelta
from qat_module import quantize_model_weights
def main():
    args = utils.parse_args()
    utils.set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print('Starting Baseline Evaluation')
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
    if args.quant:
        if args.awq == False:
            model, tokenizer = utils.load_model(args, multi_gpu=False)
            quant.quant_model(model, args)
            baseline_quantized_path = os.path.join(args.output_dir, f"baseline_quantized")
            model.save_pretrained(baseline_quantized_path)
            tokenizer.save_pretrained(baseline_quantized_path)
            del model
        else:
            baseline_quantized_path = '/mnt/cephfs/sumin/llm-awq/awq_cache/qwen3_fake_quant'
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
    return

if __name__ == "__main__":
    main()