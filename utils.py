import argparse
import re
import os
import json
import random
import datasets
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig,
)
import torch.distributed as dist
def setup_distributed():
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    
    if world_size > 1:
        if not dist.is_initialized():
            # 명시적으로 device_id 설정
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank
            )
    
    return rank, world_size, local_rank
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def load_gsm8k_dataset(split="train", max_samples=None, seed=None):
    dataset = datasets.load_dataset("gsm8k", "main")[split]
    if max_samples and max_samples < len(dataset):
        if seed is not None:
            random.seed(seed)
        else:
            random.seed(None)
        indices = random.sample(range(len(dataset)), max_samples)
        dataset = dataset.select(indices)
    
    return dataset
def load_aime_2024():
    dataset = datasets.load_dataset("Maxwell-Jia/AIME_2024")['train']
    return dataset

def load_processed_deepscaler(split="train", max_samples=None, seed=None):
    from datasets import load_from_disk
    dataset = load_from_disk("/mnt/cephfs/sumin/TPT_reproduce/dataset/deepscaler_only_numeric")[split]
    if max_samples and max_samples < len(dataset):
        if seed is not None:
            random.seed(seed)
        else:
            random.seed(None)
        indices = random.sample(range(len(dataset)), max_samples)
        dataset = dataset.select(indices)
    return dataset
        
def load_processed_aime_23(split="train", max_samples=None, seed=None):
    from datasets import load_from_disk
    dataset = load_from_disk("/mnt/cephfs/sumin/TPT_reproduce/dataset/aime_without_2024")[split]
    # if max_samples and max_samples < len(dataset):
    #     if seed is not None:
    #         random.seed(seed)
    #     else:
    #         random.seed(None)
    #     indices = random.sample(range(len(dataset)), max_samples)
    #     dataset = dataset.select(indices)
    
    return dataset
def load_model(args, multi_gpu = False):
    if multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                # attn_implementation="eager", # flash_attention_2
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code = True,
            torch_dtype = torch.bfloat16,
            # attn_implementation="eager", # flash_attention_2
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code = True,
        # padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_model_quant(args, iteration, multi_gpu = False):
    if multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(
                args.quant_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="eager", # flash_attention_2
            )
    else:
        if iteration > 0:
            print(args.quant_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.quant_path if iteration > 0 else os.path.join(args.output_dir, "quant_model_iteration_0"),
            trust_remote_code = True,
            torch_dtype = torch.bfloat16,
            # attn_implementation="eager", # flash_attention_2
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.quant_path if iteration > 0 else os.path.join(args.output_dir, "quant_model_iteration_0"),
        trust_remote_code = True,
        # padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def prompt_question(question):
    cot_prompt_template: str = """
    You are an expert mathematician.
    You are provided with a math problem. Your task is to solve the problem step-by-step, clearly showing all relevant calculations and reasoning.
    
    Problem: "{problem}"
    Requirements:
    1. Provide a complete and correct solution in a markdown block.
    2. Explain each step of the solution in detail.
    3. Conclude with the final numerical answer on a new line in the format #### . You must generate ''only'' answer after ####.
 
    Example: 
    She works 8 hours a day for $18 per hour so she makes 8*18 = $<<8*18=144.00>>144.00 per 8-hour shift
    She works 10 hours a day and anything over 8 hours is eligible for overtime, so she gets 10-8 = <<10-8=2>>2 hours of overtime
    Overtime is calculated as time and a half so and she makes $18/hour so her overtime pay is 18*.5 = $<<18*.5=9.00>>9.00
    Her overtime pay is 18+9 = $<<18+9=27.00>>27.00
    Her base pay is $144.00 per 8-hour shift and she works 5 days and makes 5 * $144 = $<<144*5=720.00>>720.00
    Her overtime pay is $27.00 per hour and she works 2 hours of overtime per day and makes 27*2 = $<<27*2=54.00>>54.00 in overtime pay
    2 hours of overtime pay for 5 days means she makes 54*5 = $270.00
    In 5 days her base pay is $720.00 and she makes $270.00 in overtime pay so she makes $720 + $270 = $<<720+270=990.00>>990.00
    #### 990
    
    Solution:
    """
    return cot_prompt_template.format(problem=question)

def prompt_question_with_system(question):
    system_prompt = """
    You are an expert mathematician.
    You are provided with a math problem. Your task is to solve the problem step-by-step, clearly showing all relevant calculations and reasoning.
    Requirements:
    1. Do not repeat problem statement in your response.
    2. Provide a complete and correct solution in a markdown block.
    3. Explain each step of the solution in detail.
    4. Conclude with the final numerical answer on a new line in the format #### . You **must** generate **only** answer after ####.
    
    Example: 
        She works 8 hours a day for $18 per hour so she makes 8*18 = $<<8*18=144.00>>144.00 per 8-hour shift
        She works 10 hours a day and anything over 8 hours is eligible for overtime, so she gets 10-8 = <<10-8=2>>2 hours of overtime
        Overtime is calculated as time and a half so and she makes $18/hour so her overtime pay is 18*.5 = $<<18*.5=9.00>>9.00
        Her overtime pay is 18+9 = $<<18+9=27.00>>27.00
        Her base pay is $144.00 per 8-hour shift and she works 5 days and makes 5 * $144 = $<<144*5=720.00>>720.00
        Her overtime pay is $27.00 per hour and she works 2 hours of overtime per day and makes 27*2 = $<<27*2=54.00>>54.00 in overtime pay
        2 hours of overtime pay for 5 days means she makes 54*5 = $270.00
        In 5 days her base pay is $720.00 and she makes $270.00 in overtime pay so she makes $720 + $270 = $<<720+270=990.00>>990.00
        #### 990
    """
    user_prompt = """
    Problem: "{problem}"
    
    Solution:
    """
    return system_prompt, user_prompt.format(problem=question)
def extract_answer(text):
    # 패턴: #### 숫자
    pattern = r"####\s*(-?\d+\.?\d*)"
    # print('utils: ', text)
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None

def check_answer(model_answer, ref_answer):
    if model_answer is None:
        return False
    try:
        A = float(model_answer)
        B = float(ref_answer)
        return abs(A - B) < 1e-6
    except:
        return False

def load_solutions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_solutions(solutions, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(solutions, f, ensure_ascii=False, indent=2)

def format_trl_dataset(pruned_solutions, config):
    """TRL 형식의 데이터셋 생성"""
    formatted_data = []
    
    for solution in pruned_solutions:
        # 프롬프트와 응답 구분
        prompt = config.cot_prompt_template.format(problem=solution["question"])
        response = solution["solution"]
        
        # TRL 형식으로 저장
        formatted_data.append({
            "text": prompt + response,  # 전체 텍스트
            "prompt": prompt,           # 프롬프트 부분 (학습에서 제외됨)
            "response": response,       # 응답 부분 (학습 대상)
        })
    
    return formatted_data
def verify_gradient_sync(model, accelerator):
    """
    Gradient synchronization이 제대로 이루어졌는지 확인하는 함수
    """
    if accelerator.num_processes > 1:
        # 각 프로세스에서 모델의 첫 번째 파라미터 값을 가져옴
        first_param = next(model.parameters()).clone()
        
        # 모든 프로세스에서 값을 수집
        gathered_params = accelerator.gather(first_param)
        
        if accelerator.is_main_process:
            # 모든 프로세스의 파라미터가 동일한지 확인
            param_values = gathered_params.view(accelerator.num_processes, -1)
            are_synced = torch.allclose(param_values[0], param_values[1:], rtol=1e-5)
            
            if are_synced:
                print("✅ Gradients are properly synchronized across all processes")
            else:
                print("❌ Warning: Gradients may not be fully synchronized")
                
    return True
def extract_number_advanced(answer):
    """
    더 복잡한 경우를 처리하는 고급 버전
    - 분수 (1/2, 3/4 등)
    - 쉼표가 포함된 숫자 (1,234)
    - 여러 숫자가 있을 때 첫 번째 의미있는 숫자 추출
    """
    answer = str(answer).strip()
    
    # 쉼표 제거
    answer = answer.replace(',', '')
    
    # LaTeX 명령어 제거
    answer = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', answer)  # \text{...} 등 제거
    answer = re.sub(r'\\[a-zA-Z]+', '', answer)  # \cdot, \times 등 제거
    
    # 분수 처리 (예: 1/2, 3/4)
    fraction_pattern = r'(\d+)\s*/\s*(\d+)'
    fraction_match = re.search(fraction_pattern, answer)
    if fraction_match:
        numerator = int(fraction_match.group(1))
        denominator = int(fraction_match.group(2))
        return numerator / denominator
    
    # 일반 숫자 추출
    number_pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
    matches = re.findall(number_pattern, answer)
    
    if matches:
        # 첫 번째 숫자 반환
        try:
            if '.' in matches[0] or 'e' in matches[0].lower():
                return float(matches[0])
            else:
                return int(matches[0])
        except ValueError:
            return None
    
    return None
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", 
        action=argparse.BooleanOptionalAction, default=False, 
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output", 
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/mnt/cephfs/sumin/model/Llama-3.2-1B-Instruct", 
    )
    parser.add_argument(
        "--quant_path", 
        type=str, 
        default="/mnt/cephfs/sumin/TPT_reproduce", 
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
    )
    parser.add_argument(
        "--train", 
        type=str, 
        default="deepscaler", 
    )
    parser.add_argument(
        "--train_samples", 
        type=int, 
        default=2000, 
    )
    parser.add_argument(
        "--eval_samples", 
        type=int, 
        default=800, 
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=1024, 
    )
    parser.add_argument(
        "--num_solutions", 
        type=int, 
        default=10, 
    )
    parser.add_argument(
        "--N", 
        type=int, 
        default=1, 
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7, 
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.95, 
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
    )
    parser.add_argument(
        "--tpt_iterations", 
        type=int, 
        default=4, 
    )
    parser.add_argument(
        "--i", 
        type=int, 
        default=0, 
    )
    parser.add_argument(
        "--train_epoch", 
        type=int, 
        default=1, 
    )
    parser.add_argument(
        "--system_prompt", 
        action=argparse.BooleanOptionalAction, default=False, 
    )
    parser.add_argument(
        "--awq", 
        action=argparse.BooleanOptionalAction, default=False, 
    )
    parser.add_argument(
        "--resume", 
        action=argparse.BooleanOptionalAction, default=False, 
    )
    parser.add_argument(
        "--add_gold_answer", 
        action=argparse.BooleanOptionalAction, default=False, 
    )   
    parser.add_argument(
        "--quant", 
        action=argparse.BooleanOptionalAction, default=False, 
    ) 
    parser.add_argument(
        "--bit", 
        type=int, 
        default=8, 
    )
    parser.add_argument(
        "--quant_type", 
        type=str, 
        default="rtn", 
    )
    parser.add_argument(
        "--baseline", 
        action=argparse.BooleanOptionalAction, default=False, 
    ) 
    parser.add_argument(
        "--eval_multi", 
        action=argparse.BooleanOptionalAction, default=False, 
    ) 
    return parser.parse_args()
    
    
    