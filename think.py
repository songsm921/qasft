import utils as utils
import os
import re
import json
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.model_executor.models.llama import LlamaForCausalLM
import tempfile
import shutil

class ThinkDataset(Dataset):
    """Dataset wrapper for think phase data"""
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx], idx  # Return data and index
    
def prepare_vllm_model(model_path, tensor_parallel_size=1):
    llm = LLM(
        model = model_path,
        tensor_parallel_size = tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=32678, # adjust?
        gpu_memory_utilization=0.9
    )
    
    return llm

def think_phase(model_path, dataset, args, output_file=None):
    print(model_path)
    llm = prepare_vllm_model(model_path, tensor_parallel_size=1)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        n=args.num_solutions,  # 한 번에 N개 생성 ?
        stop=None
    )
    local_solutions = []
        
    # 각 문제 처리
    i = 0
    for problem_idx, example in enumerate(tqdm(dataset, desc=f"Think vLLM ")):
        print(f"Question {i} processed.")
        i = i+1
        question = example["problem"]
        golden_answer = example["solution"]
        golden_digit = example["answer"]
        
        # 프롬프트 준비
        if args.system_prompt:
            qwen_pt = "<\uff5cUser\uff5c>" + "Please reason step by step, and put your final answer within \\boxed{}."
            prompt_text = qwen_pt + question + '<think>\n'
        else:
            prompt_text = utils.prompt_question(question)
        
        # vLLM으로 생성 (배치로 N개 한번에 생성)
        try:
            # 같은 프롬프트를 여러 번 반복하여 다양한 답안 생성
            prompts = [prompt_text]  # 단일 프롬프트
            
            # 각 solution마다 다른 seed 사용을 위해 여러 번 생성
            generated_solutions = []
            for solution_idx in range(args.num_solutions):
                solution_seed = args.seed + problem_idx * 100 + solution_idx
                temp_sampling_params = SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_new_tokens,
                    seed=solution_seed,
                    n=1,  # 한번에 하나씩
                    stop=None
                )
                
                outputs = llm.generate(prompts, temp_sampling_params)
                generated_solutions.extend(outputs)
            
            # 생성된 답안들 처리
            for solution_idx, output in enumerate(generated_solutions):
                generated_text = output.outputs[0].text
                
                # 기존 코드와 동일한 답안 추출 로직
                if args.system_prompt:
                    # <think> 태그 처리
                    think_start = generated_text.find('<think>')
                    if think_start != -1:
                        text_from_think = generated_text[think_start+8:]
                        prefix = prompt_text + generated_text[:think_start+8-len(prompt_text)]
                        
                        pattern = r'\\boxed\{([^}]+)\}'
                        match = re.search(pattern, text_from_think)
                        
                        if match:
                            end_idx = match.end()
                            solution_text_full = text_from_think[:end_idx]
                        else:
                            solution_text_full = text_from_think
                    else:
                        solution_text_full = generated_text
                        prefix = prompt_text
                else:
                    solution_text_full = generated_text
                    prefix = prompt_text
                    
                # 답안 추출
                answer_pattern = r'\\boxed\{([^}]+)\}'
                match = re.search(answer_pattern, solution_text_full)
                if match:
                    answer = match.group(1)
                    end_idx = match.end()
                    solution_text = solution_text_full[:end_idx]
                else:
                    solution_text = solution_text_full
                    answer = None
                    
                extracted_answer = utils.extract_number_advanced(answer)
                
                # 결과 저장 (기존 형식 유지)
                local_solutions.append({
                    "problem_id": problem_idx,
                    "solution_id": solution_idx,
                    "question": question,
                    "reference_answer": golden_answer,
                    "gold_answer": golden_digit,
                    "solution": solution_text,
                    'text': prefix + solution_text if args.system_prompt else prompt_text + solution_text,
                    "extracted_answer": extracted_answer,
                })
                
        except Exception as e:
            print(f"Error generating for problem {problem_idx}: {e}")
            continue
        
        if (problem_idx + 1) % 10 == 0:
            print(f"Processed {problem_idx + 1} problems")
    
    # 결과 저장

    utils.save_solutions(local_solutions, output_file)
    
    return local_solutions