import eval_utils as eval_utils
import utils as utils
import os
args = utils.parse_args()
eval_dataset = utils.load_aime_2024()
iteration = 1
temporary_eval_path = os.path.join(args.output_dir, f"temporary_eval_iter_{iteration}") 
iter_results = eval_utils.evaluate_accelerate(temporary_eval_path, eval_dataset, args, k_values=[1])
print(iter_results)