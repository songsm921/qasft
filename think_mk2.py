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
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue
from transformers import AutoTokenizer
class ThinkDataset(Dataset):
    """Dataset wrapper for think phase data"""
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx], idx  # Return data and index

def prepare_vllm_model(model_path, tensor_parallel_size=4):
    """4개 GPU를 사용하도록 tensor_parallel_size를 4로 설정"""
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,  # 4개 GPU 사용
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        # 추가 최적화 옵션들
        swap_space=24,  # GB, CPU 메모리 활용
        max_num_seqs=512,  # 배치 크기 증가
    )
    return llm

def prepare_batch_prompts(tokenizer,examples, args, batch_size=64):
    """배치 처리를 위한 프롬프트 준비"""
    batched_prompts = []
    batch_metadata = []
    
    current_batch = []
    current_metadata = []
    
    for problem_idx, example in enumerate(examples):
        if args.train == 'deepscaler':
            question = example["problem"]
            golden_answer = example["solution"]
            golden_digit = example["answer"]
        elif args.train == 'aime':
            question = example["Question"]
            golden_digit = example["Answer"]
            golden_answer = None
        
        # flag = '<|ASSISTANT|>' #'<think>'#'<|ASSISTANT|>' #'<think>\n'
        # # 프롬프트 준비
        # if args.system_prompt:
        #     qwen_pt = "<\uff5cUser\uff5c>" + "Please reason step by step, and put your final answer within \\boxed{}.\n"
        #     prompt_text = qwen_pt + question + flag + '\n'
        # else:
        #     prompt_text = utils.prompt_question(question)
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
        # 각 문제에 대해 여러 솔루션 생성
        for solution_idx in range(args.num_solutions):
            current_batch.append(text)
            current_metadata.append({
                "problem_idx": problem_idx,
                "solution_idx": solution_idx,
                "question": question,
                "golden_answer": golden_answer,
                "golden_digit": golden_digit,
                "prompt_text": prompt
            })
            
            if len(current_batch) >= batch_size:
                batched_prompts.append(current_batch)
                batch_metadata.append(current_metadata)
                current_batch = []
                current_metadata = []
    
    # 남은 배치 추가
    if current_batch:
        batched_prompts.append(current_batch)
        batch_metadata.append(current_metadata)
    
    return batched_prompts, batch_metadata

def process_batch_outputs(outputs, metadata_batch, args):
    """배치 출력 처리"""
    local_solutions = []
    
    for output, metadata in zip(outputs, metadata_batch):
        generated_text = output.outputs[0].text
        # print(generated_text[:50])
        
        # 기존 답안 추출 로직
        if args.system_prompt:
            flag = '<think>'
            think_start = generated_text.find(flag)
            if think_start != -1:
                text_from_think = generated_text[think_start:]
                prefix = output.prompt
                
                pattern = r'\\boxed\{([^}]+)\}'
                match = re.search(pattern, text_from_think)
                
                if match:
                    end_idx = match.end()
                    solution_text_full = text_from_think[:end_idx]
                else:
                    solution_text_full = text_from_think
            else:
                solution_text_full = generated_text
                prefix = metadata["prompt_text"]
        else:
            solution_text_full = generated_text
            prefix = output.prompt
            
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
        
        # 결과 저장
        local_solutions.append({
            "problem_id": metadata["problem_idx"],
            "solution_id": metadata["solution_idx"],
            "question": metadata["question"],
            "reference_answer": metadata["golden_answer"],
            "gold_answer": metadata["golden_digit"],
            "solution": solution_text,
            'text': prefix + solution_text ,
            "extracted_answer": extracted_answer,
        })
    
    return local_solutions

def think_phase_optimized(model_path, dataset, args, output_file=None, batch_size=128):
    """최적화된 think phase 함수 - 4개 GPU + 대용량 배치 처리"""
    print(f"Using model: {model_path}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    # 4개 GPU로 모델 로드
    llm = prepare_vllm_model(model_path, tensor_parallel_size=4) #4
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 배치별로 다른 시드 사용을 위한 샘플링 파라미터
    base_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=1,
        stop=None
    )
    
    # 배치 프롬프트 준비
    print("Preparing batched prompts...")
    batched_prompts, batch_metadata= prepare_batch_prompts(tokenizer,dataset, args, batch_size)
    
    print(f"Total batches: {len(batched_prompts)}")
    print(f"Average batch size: {sum(len(batch) for batch in batched_prompts) / len(batched_prompts):.1f}")
    
    all_solutions = []
    
    # 배치별로 처리
    for batch_idx, (prompt_batch, metadata_batch) in enumerate(tqdm(
        zip(batched_prompts, batch_metadata), 
        total=len(batched_prompts), 
        desc="Processing batches"
    )):
        # try:
            # 각 프롬프트마다 다른 시드 설정
        batch_sampling_params = []
        for i, metadata in enumerate(metadata_batch):
            seed = args.seed + metadata["problem_idx"] * 100 + metadata["solution_idx"]
            sampling_params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k = 20,
                max_tokens=args.max_new_tokens,
                seed=seed,
                n=1,
                stop=None
            )
            batch_sampling_params.append(sampling_params)
        
        # 배치 생성 (vLLM이 자동으로 최적 배치 크기로 조정)
        # print(prompt_batch)
        outputs = llm.generate(prompt_batch, base_sampling_params)
        
        # 출력 처리
        batch_solutions = process_batch_outputs(outputs, metadata_batch, args)
        all_solutions.extend(batch_solutions)
        
        # 진행상황 출력
        if (batch_idx + 1) % 10 == 0:
            total_problems = sum(1 for _ in set(sol["problem_id"] for sol in all_solutions))
            print(f"Processed {batch_idx + 1} batches, {total_problems} unique problems")
        
        # except Exception as e:
        #     print(f"Error processing batch {batch_idx}: {e}")
        #     continue
    
    print(f"Total solutions generated: {len(all_solutions)}")
    
    # 결과 저장
    if output_file:
        utils.save_solutions(all_solutions, output_file)
    
    return all_solutions

