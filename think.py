import utils as utils
import os
import re
import json
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from accelerate import Accelerator

class ThinkDataset(Dataset):
    """Dataset wrapper for think phase data"""
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx], idx  # Return data and index

def think_phase_accelerate(model, tokenizer, dataset, args, output_file=None , acc=None):
    """
    Accelerate-based think phase function
    """
    accelerator = acc
    
    # Prepare model
    # model = accelerator.prepare(model)
    model = model.to(accelerator.device) 
    model.eval()
    
    # Create dataset and sampler
    think_dataset = ThinkDataset(dataset)
    # sampler = DistributedSampler(
    #     think_dataset,
    #     num_replicas=accelerator.num_processes,
    #     rank=accelerator.process_index,
    #     shuffle=False
    # )
    
    # Create dataloader
    dataloader = DataLoader(
        think_dataset,
        batch_size=1,  # Process one example at a time
        shuffle=False
    )
    # dataloader = dataloader.to(accelerator.device)
    dataloader = accelerator.prepare(dataloader)
    
    local_solutions = []
    num_solutions = args.num_solutions
    
    for batch in tqdm(dataloader, desc=f"Think GPU {accelerator.process_index}"):

        example, problem_idx = batch
        # example = example[0]  # batch_size=1
        problem_idx = problem_idx.item()
        
        question = example["problem"][0]
        # print(example["solution"])
        golden_answer = example["solution"][0]
        golden_digit = example["answer"][0]
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
            
        if args.system_prompt:
            inputs = tokenizer.apply_chat_template(prompt, return_tensors='pt', return_dict=True)
            inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
        
        # Generate multiple solutions
        for solution_idx in range(num_solutions):
            solution_seed = args.seed + problem_idx * 100 + solution_idx
            torch.manual_seed(solution_seed)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode and process
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if args.system_prompt:
                # <think> 태그 찾기
                think_start = generated_text.find('<think>')
                
                if think_start != -1:
                    # <think>부터 시작하는 텍스트에서 \boxed{} 찾기
                    text_from_think = generated_text[think_start+8:]
                    prefix = generated_text[:think_start+8]
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
                
            answer_pattern = r'\\boxed\{([^}]+)\}'
            match = re.search(answer_pattern, solution_text_full)
            if match:
                # 괄호 안의 내용만 추출
                answer = match.group(1)
                # 전체 매치된 부분까지의 텍스트
                end_idx = match.end()
                solution_text = solution_text_full[:end_idx]
            else:
                solution_text = solution_text_full
                answer = None
                
            extracted_answer = utils.extract_number_advanced(answer)#answer #utils.extract_answer(solution_text)
            
            local_solutions.append({
                "problem_id": problem_idx,
                "solution_id": solution_idx,
                "question": question,
                "reference_answer": golden_answer,
                "gold_answer": golden_digit,
                "solution": solution_text,
                'text': prefix  + solution_text if prefix is not None else generated_text,
                "extracted_answer": extracted_answer,
            })
        
        if accelerator.is_main_process and (problem_idx + 1) % 10 == 0:
            print(f"Processed {problem_idx + 1} problems")
    
    # Save solutions for this process
    if output_file:
        rank_output_file = output_file.replace('.json', f'_rank{accelerator.process_index}.json')
        utils.save_solutions(local_solutions, rank_output_file)
    
    # Wait for all processes to finish
    # accelerator.wait_for_everyone()
    
    return local_solutions