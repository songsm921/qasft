#!/bin/bash

# CUDA_VISIBLE_DEVICES=4,5,6,7 python 1_baseline_vllm.py --model_path /mnt/cephfs/echoi/models/L1-Qwen-1.5B-Max --bit 4 --quant --quant_type qat --train_samples 3000 --max_new_tokens 4096 --system_prompt --num_solutions 5 --temperature 1.0 --train_epoch 2 --output_dir ./3k_4k_t1.0_ep2 --N 2 --tpt_iterations 4

for ((i=0; i<4; i++)); do
    CUDA_VISIBLE_DEVICES=4,5,6,7 python 2_think_prune_vllm.py --model_path /mnt/cephfs/echoi/models/L1-Qwen-1.5B-Max --train_samples 3000 --max_new_tokens 4096 --system_prompt --num_solutions 5 --temperature 1.0 --train_epoch 2 --output_dir ./3k_4k_t1.0_ep2 --N 2 --tpt_iterations 4 --i $i
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node 4 --master_port 12345 3_sft_torchrun.py --model_path /mnt/cephfs/echoi/models/L1-Qwen-1.5B-Max --train_samples 3000 --max_new_tokens 4096 --system_prompt --num_solutions 5 --temperature 1.0 --train_epoch 2 --output_dir ./3k_4k_t1.0_ep2 --N 2 --tpt_iterations 4 --i $i
    CUDA_VISIBLE_DEVICES=4,5,6,7 python 4_iteration_eval_vllm.py --model_path /mnt/cephfs/echoi/models/L1-Qwen-1.5B-Max --train_samples 3000 --max_new_tokens 4096 --system_prompt --num_solutions 5 --temperature 1.0 --train_epoch 2 --output_dir ./3k_4k_t1.0_ep2 --N 2 --tpt_iterations 4 --i $i
done