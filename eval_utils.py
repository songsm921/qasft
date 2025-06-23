import torch
from tqdm import tqdm
import re
import utils as utils
import math
from collections import defaultdict
import gc
import numpy as np
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler, Dataset

class EvaluationDataset(Dataset):
    """Dataset wrapper for evaluation data"""
    def __init__(self, eval_dataset):
        self.eval_dataset = eval_dataset
        
    def __len__(self):
        return len(self.eval_dataset)
    
    def __getitem__(self, idx):
        return self.eval_dataset[idx]

def evaluate_accelerate(model, tokenizer, eval_dataset, args, k_values=[1, 20], acc=None):
    """
    Accelerate-based multi-GPU evaluation function
    """
    accelerator = acc
    
    # Prepare model with accelerator
    # model = accelerator.prepare(model)
    model = model.to(accelerator.device) 
    model.eval()
    
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
    dataloader = accelerator.prepare(dataloader)
    # print(len(dataloader))
    # Initialize results
    results = {f"pass@{k}": 0 for k in k_values}
    max_k = max(k_values)
    processed_examples = 0
    
    for batch in tqdm(dataloader, desc=f"GPU {accelerator.process_index}"):
        # print(batch)
        example = batch  # batch_size=1이므로 첫 번째 요소만 사용
        
        question = example['Problem'][0]
        golden_answer = example['Answer'][0].item()
        length = -1
        
        if args.system_prompt:
            # system, user = utils.prompt_question_with_system(question)
            # length = len(system + user)
            qwen_pt = "Please reason step by step, and put your final answer within \\boxed{}."
            prompt = [
                # {"role": "system", "content": qwen_pt},
                {"role": "user", "content": qwen_pt + question + '<think>\n'}  
            ]
            length = len(qwen_pt + question)
        else:
            prompt = utils.prompt_question(question)
            length = len(prompt)
        
        correct_count = 0
        for _ in range(max_k):
            if args.system_prompt:
                inputs = tokenizer.apply_chat_template(prompt, return_tensors='pt', return_dict=True)
                inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
            else:
                inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
                
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
                
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
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
            
            if args.debug and accelerator.is_main_process:
                print(f"GPU {accelerator.process_index}: Generated text:", generated_text)
                print(f"GPU {accelerator.process_index}: Solution part:", solution_part)
                print(f"GPU {accelerator.process_index}: Extracted:", extracted_answer)
                print(f"GPU {accelerator.process_index}: Golden:", golden_answer)
                print('-----------------')
                
            if utils.check_answer(model_answer=extracted_answer, ref_answer=golden_answer):
                correct_count += 1
                
        for k in k_values:
            if correct_count >= 1 and k <= max_k:
                results[f'pass@{k}'] += 1
                
        processed_examples += 1
        if processed_examples % 10 == 0 and accelerator.is_main_process:
            pass_at_1 = (results["pass@1"] / processed_examples) * 100
            print(f"GPU {accelerator.process_index}: Processed {processed_examples} examples, Pass@1: {pass_at_1:.2f}%")
    
    # Gather results from all processes
    gathered_results = {}
    for k in k_values:
        local_result = torch.tensor(results[f"pass@{k}"], device=accelerator.device)
        print(local_result)
        gathered_result = accelerator.gather(local_result)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            gathered_results[f"pass@{k}"] = gathered_result.sum().item()
    
    # Calculate final percentages on main process
    if accelerator.is_main_process:
        total_examples = len(eval_dataset)
        final_results = {}
        for k in k_values:
            final_results[f"pass@{k}"] = (gathered_results[f"pass@{k}"] / total_examples ) * 100
        
        print("\n" + "="*50)
        print("FINAL RESULTS:")
        for k in k_values:
            print(f"Pass@{k}: {final_results[f'pass@{k}']:.2f}%")
        print("="*50)
        
        return final_results
    
    return None