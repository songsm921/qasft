import eval_utils as eval_utils
import utils as utils
import os
args = utils.parse_args()
eval_dataset = utils.load_aime_2024()
iteration = 16
baseline_results = 0.0
quantized_results = 0.0
# for i in range(iteration):
#     iter_results, _ = eval_utils.evaluate_accelerate(args.model_path, eval_dataset, args, k_values=[1])
#     baseline_results += iter_results["pass@1"]
    
temporary_eval_path = os.path.join(args.output_dir, f"baseline_quantized") 
for i in range(iteration):
    iter_results, _ = eval_utils.evaluate_accelerate(temporary_eval_path, eval_dataset, args, k_values=[1])
    quantized_results += iter_results["pass@1"]
# avg_baseline = baseline_results / iteration    
avg_quantized = quantized_results / iteration
# print(f"Baseline average({iteration}): ", avg_baseline)
print(f"Quantized average({iteration}): ", avg_quantized)


