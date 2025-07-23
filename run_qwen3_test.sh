#!/bin/bash

# CUDA_VISIBLE_DEVICES=4,5,6,7 python 1_baseline_vllm.py --model_path /mnt/cephfs/sumin/model/Qwen3-1.7B --bit 4 --quant --quant_type qat --train_samples 2000 --max_new_tokens 32000 --system_prompt --num_solutions 5 --temperature 0.6 --train_epoch 2 --output_dir ./2k_32k_t0.6_ep2_qwen3 --N 2 --tpt_iterations 4 --debug

for ((i=0; i<4; i++)); do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python 2_think_prune_vllm.py --model_path /mnt/cephfs/sumin/model/Qwen3-1.7B --bit 4 --quant --quant_type qat --train_samples 1 --max_new_tokens 32000 --system_prompt --num_solutions 2 --temperature 0.6 --train_epoch 2 --output_dir ./test --N 2 --tpt_iterations 4 --i $i
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 --master_port 12341 3_sft_torchrun.py --model_path /mnt/cephfs/sumin/model/Qwen3-1.7B --bit 4 --quant --quant_type qat --train_samples 2000 --max_new_tokens 32000 --system_prompt --num_solutions 5 --temperature 0.6 --train_epoch 2 --output_dir ./test --N 2 --tpt_iterations 4 --i $i
    CUDA_VISIBLE_DEVICES=0,1,2,3 python 4_iteration_eval_vllm.py --model_path /mnt/cephfs/sumin/model/Qwen3-1.7B --bit 4 --quant --quant_type qat --train_samples 2000 --max_new_tokens 32000 --system_prompt --num_solutions 5 --temperature 0.6 --train_epoch 2 --output_dir ./test --N 2 --tpt_iterations 4 --i $i
done