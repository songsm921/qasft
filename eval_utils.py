import torch
from tqdm import tqdm
import re
import utils as utils
import math
from collections import defaultdict
import gc
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from vllm import LLM, SamplingParams
from vllm.model_executor.models.llama import LlamaForCausalLM
import tempfile
import shutil
from transformers import AutoTokenizer
def prepare_vllm_model(model_path, tensor_parallel_size=1):
    llm = LLM(
        model = model_path,
        disable_log_stats=True,
        tensor_parallel_size = tensor_parallel_size,
        enable_reasoning=True,  # --enable-reasoning 옵션
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=32768, # adjust?
        gpu_memory_utilization=0.9
    )
    
    return llm
class EvaluationDataset(Dataset):
    """Dataset wrapper for evaluation data"""
    def __init__(self, eval_dataset):
        self.eval_dataset = eval_dataset
        
    def __len__(self):
        return len(self.eval_dataset)
    
    def __getitem__(self, idx):
        return self.eval_dataset[idx]

def evaluate_accelerate(model, eval_dataset, args, k_values=[1, 20]):
    """
    Accelerate-based multi-GPU evaluation function
    """
    print(model)
    llm = prepare_vllm_model(model, tensor_parallel_size=1)
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    # Create dataset and sampler
    dataset = EvaluationDataset(eval_dataset)
    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=accelerator.num_processes,
    #     rank=accelerator.process_index,
    #     shuffle=False
    # )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one example at a time for generation
        shuffle=False
    )
    # dataloader = accelerator.prepare(dataloader)
    # print(len(dataloader))
    # Initialize results
    results = {f"pass@{k}": 0 for k in k_values}
    correction = {'correct': [], 'incorrect': []}
    max_k = max(k_values)
    processed_examples = 0
    
    for batch in tqdm(dataloader, desc=f"Evaluation for AIME2024..."):
        # print(batch)
        example = batch  # batch_size=1이므로 첫 번째 요소만 사용
        
        question = example['Problem'][0]
        golden_answer = example['Answer'][0].item()
        length = -1
        
        prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        prompt = prompt + question
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Set to False to strictly disable thinking
        )
        
        correct_count = 0
        for _ in range(max_k):
            sampling_params = SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k = 20,
                    max_tokens=args.max_new_tokens,
                    n=1,  # 한번에 하나씩
                    stop=None
                )
            outputs = llm.generate([text], sampling_params) 
                
            generated_text =outputs[0].prompt + outputs[0].outputs[0].text
            
            # print(generated_text)
            
            if args.system_prompt:
                # <think> 태그 찾기
                think_start = generated_text.find('<think>')
                
                if think_start != -1:
                    # <think>부터 시작하는 텍스트에서 \boxed{} 찾기
                    text_from_think = generated_text[think_start:]
                    
                    # \boxed{} 패턴 찾기
                    pattern = r'\\boxed\{([^}]+)\}'
                    match = re.search(pattern, text_from_think)
                    
                    if match:
                        # <think>부터 \boxed{}까지
                        end_idx = match.end()
                        solution_text_full = text_from_think[:end_idx]
                    else:
                        # \boxed{}가 없으면 <think>부터 끝까지
                        solution_text_full = text_from_think
                else:
                    # <think>가 없으면 전체 텍스트 사용
                    solution_text_full = generated_text
            else:
                solution_text_full = generated_text[length:]
                
            # Extract answer using pattern
            answer_pattern = r'\\boxed\{([^}]+)\}'
            match = re.search(answer_pattern, solution_text_full)
            if match:
                # 괄호 안의 내용만 추출
                answer = match.group(1)
                # 전체 매치된 부분까지의 텍스트
                end_idx = match.end()
                solution_part = solution_text_full[:end_idx]
            else:
                solution_part = solution_text_full
                answer = None
                
            extracted_answer = utils.extract_number_advanced(answer) #utils.extract_answer(solution_part)
            
            if args.debug:
                print("Generated text:", generated_text)
                print("Solution part:", solution_part)
                print("Extracted:", extracted_answer)
                print("Golden:", golden_answer)
                print('-----------------')
                
            if utils.check_answer(model_answer=extracted_answer, ref_answer=golden_answer):
                correct_count += 1
                correction['correct'].append(processed_examples + 1)
            else:
                correction['incorrect'].append(processed_examples + 1)
                
        for k in k_values:
            if correct_count >= 1 and k <= max_k:
                results[f'pass@{k}'] += 1
                
        processed_examples += 1
        if processed_examples % 10 == 0:
            pass_at_1 = (results["pass@1"] / processed_examples) * 100
            print(f"Processed {processed_examples} examples, Pass@1: {pass_at_1:.2f}%")
    
    total_examples = len(eval_dataset)
    final_results = {}
    for k in k_values:
        final_results[f"pass@{k}"] = (results[f"pass@{k}"] / total_examples ) * 100
    
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    for k in k_values:
        print(f"Pass@{k}: {final_results[f'pass@{k}']:.2f}%")
    print("="*50)
    
    return final_results, correction
