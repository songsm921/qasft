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
    if not os.path.exists(os.path.join(args.output_dir, f'quant_model_iteration_{iteration}')):
        print("\n--- Train Stage ---\n")
            
        pruned_file = os.path.join(args.output_dir, f"pruned_solutions_iter_{iteration}.json")
        pruned_solutions = utils.load_solutions(pruned_file)
        # if args.quant_type != 'qat':
        if iteration > 0:
            if args.quant == False:
                args.model_path = os.path.join(args.output_dir, f"tpt_model_iteration_{iteration-1}")
            else:
                args.quant_path = os.path.join(args.output_dir, f"quant_model_iteration_{iteration-1}")

        if args.quant:
            if iteration == 0:
                if args.awq:
                    args.model_path = '/mnt/cephfs/sumin/llm-awq/awq_cache/qwen3_fake_quant'
                    model, tokenizer = utils.load_model(args)
                else:
                    model, tokenizer = utils.load_model(args)
            else:
                model, tokenizer = utils.load_model_quant(args, iteration, multi_gpu=False)
            if args.quant_type == 'qat':
                model = quantize_model_weights(model, bits=args.bit, per_channel=True)
                print(model)
            else:
                quant.quant_model(model, args)
        else:
            model, tokenizer = utils.load_model(args)
            
        train.train_phase(model, tokenizer, pruned_solutions, iteration, args)
    else:
        print(f"{iteration} iteration: Train Phase passed")
    return

if __name__ == "__main__":
    main()