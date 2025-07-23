import os
import random
import utils as utils
import json

def prune_solutions(solutions,args,  strategy="correct_only", max_samples_per_problem=None):

    print(f"Pruning solutions using strategy: {strategy}...")
    
    if strategy == "correct_only":
        problems_dict = {}
        for solution in solutions:
            problem_id = solution["problem_id"]
            if problem_id not in problems_dict:
                problems_dict[problem_id] = {
                    "all_solutions": [],
                    "correct_solutions": [],
                    "reference": solution["reference_answer"],
                    "refernce_answer":solution["gold_answer"],
                    "question": solution["question"],
                    "text": solution["text"]
                }
            
            problems_dict[problem_id]["all_solutions"].append(solution)
            
            if utils.check_answer(solution["extracted_answer"], solution["reference_answer"]):
                problems_dict[problem_id]["correct_solutions"].append(solution)
        pruned_solutions = []
        

        no_correct_count = 0
        total_solutions_count = 0
        total_problems_count = len(problems_dict)
        
        for problem_id, problem_data in problems_dict.items():
            if problem_data["correct_solutions"]:

                correct_sols = problem_data["correct_solutions"]
                

                if max_samples_per_problem and len(correct_sols) > max_samples_per_problem:
                    selected_sols = random.sample(correct_sols, max_samples_per_problem)
                else:
                    selected_sols = correct_sols
                
                pruned_solutions.extend(selected_sols)
                total_solutions_count += len(selected_sols)
            else:
                no_correct_count += 1
                if args.add_gold_answer:
                    marker = "Solution:assistant\n\n"
                    prefix = problem_data['text'].split(marker, 1)[0]
                        # solution_text_full = generated_text.split(marker, 1)[1]
                    
                    reference_solution = {
                        "problem_id": problem_id,
                        "solution_id": -1,  
                        "question": problem_data["question"],
                        "reference_answer": problem_data["reference"],
                        "solution": problem_data["reference"],
                        "text": prefix + marker+problem_data["reference"],
                        "extracted_answer": utils.extract_answer(problem_data["reference"]),
                    }
                    pruned_solutions.append(reference_solution)
                    total_solutions_count += 1
        
        print(f"Selected {total_solutions_count} solutions from {total_problems_count} problems")
        if no_correct_count > 0:
            if args.add_gold_answer:
                print(f"Added golden answers for {no_correct_count} problems with no correct solutions")
            else:
                print(f"{no_correct_count} problems has no correction solution.")
    else:
        pruned_solutions = solutions
    
    print(f"Pruned from {len(solutions)} to {len(pruned_solutions)} solutions")
    
    return pruned_solutions



def prune_solutions_consider_quant(solutions, quant_solutions, args, strategy="correct_only", max_samples_per_problem=None):
    
    print(f"Pruning solutions considering quantized model using strategy: {strategy}...")
    
    if strategy == "correct_only":
        # 문제별로 solutions 그룹화
        problems_dict = {}
        
        # Quantized solutions 처리
        for solution in quant_solutions:
            problem_id = solution["problem_id"]
            if problem_id not in problems_dict:
                problems_dict[problem_id] = {
                    "quant_solutions": [],
                    "quant_correct_solutions": [],
                    "regular_solutions": [],
                    "regular_correct_solutions": [],
                    "reference": solution["reference_answer"],
                    "reference_answer": solution["gold_answer"],
                    "question": solution["question"],
                    "text": solution["text"]
                }
            
            problems_dict[problem_id]["quant_solutions"].append(solution)
            
            # Quantized solution이 정답인지 확인
            if utils.check_answer(solution["extracted_answer"], solution["gold_answer"]):
                problems_dict[problem_id]["quant_correct_solutions"].append(solution)
        
        # Regular solutions 처리
        for solution in solutions:
            problem_id = solution["problem_id"]
            if problem_id not in problems_dict:
                problems_dict[problem_id] = {
                    "quant_solutions": [],
                    "quant_correct_solutions": [],
                    "regular_solutions": [],
                    "regular_correct_solutions": [],
                    "reference": solution["reference_answer"],
                    "reference_answer": solution["gold_answer"],
                    "question": solution["question"],
                    "text": solution["text"]
                }
            
            problems_dict[problem_id]["regular_solutions"].append(solution)
            
            # Regular solution이 정답인지 확인
            if utils.check_answer(solution["extracted_answer"], solution["gold_answer"]):
                problems_dict[problem_id]["regular_correct_solutions"].append(solution)
        
        pruned_solutions = []
        
        # 통계 변수
        quant_correct_count = 0
        regular_correct_count = 0
        gold_answer_count = 0
        no_correct_count = 0
        total_solutions_count = 0
        total_problems_count = len(problems_dict)
        
        for problem_id, problem_data in problems_dict.items():
            selected_sols = []
            
            # 1. Quantized model에서 정답이 있는 경우 우선 선택
            if problem_data["quant_correct_solutions"]:
                quant_correct_sols = problem_data["quant_correct_solutions"]
                
                # max_samples_per_problem 제한이 있는 경우
                if max_samples_per_problem:
                    if len(quant_correct_sols) >= max_samples_per_problem:
                        # quantized model 정답이 충분한 경우
                        selected_sols = random.sample(quant_correct_sols, max_samples_per_problem)
                    else:
                        # quantized model 정답이 부족한 경우, 모든 정답을 선택
                        selected_sols = quant_correct_sols.copy()
                        remaining_slots = max_samples_per_problem - len(selected_sols)
                        
                        # regular model에서 추가로 선택
                        if remaining_slots > 0 and problem_data["regular_correct_solutions"]:
                            regular_correct_sols = problem_data["regular_correct_solutions"]
                            
                            # 남은 슬롯만큼 추가 선택
                            additional_count = min(remaining_slots, len(regular_correct_sols))
                            additional_sols = random.sample(regular_correct_sols, additional_count)
                            selected_sols.extend(additional_sols)
                else:
                    # max_samples_per_problem 제한이 없는 경우
                    selected_sols = quant_correct_sols
                
                quant_correct_count += 1
                
            # 2. Quantized model에 정답이 없지만 regular model에 정답이 있는 경우
            elif problem_data["regular_correct_solutions"]:
                correct_sols = problem_data["regular_correct_solutions"]
                
                if max_samples_per_problem and len(correct_sols) > max_samples_per_problem:
                    selected_sols = random.sample(correct_sols, max_samples_per_problem)
                else:
                    selected_sols = correct_sols
                
                regular_correct_count += 1
                
            # 3. 두 모델 모두 정답이 없는 경우
            else:
                no_correct_count += 1
                if args.add_gold_answer:
                    # Gold answer 추가
                    marker = "Solution:assistant\n\n"
                    prefix = problem_data['text'].split(marker, 1)[0]
                    
                    reference_solution = {
                        "problem_id": problem_id,
                        "solution_id": -1,  
                        "question": problem_data["question"],
                        "reference_answer": problem_data["reference"],
                        "solution": problem_data["reference"],
                        "text": prefix + marker + problem_data["reference"],
                        "extracted_answer": utils.extract_answer(problem_data["reference"]),
                    }
                    selected_sols = [reference_solution]
                    gold_answer_count += 1
            
            # 선택된 solutions 추가
            if selected_sols:
                pruned_solutions.extend(selected_sols)
                total_solutions_count += len(selected_sols)

        
        # 결과 출력
        print(f"Selected {total_solutions_count} solutions from {total_problems_count} problems")
        print(f"- Used quantized model solutions: {quant_correct_count} problems")
        print(f"- Used regular model solutions: {regular_correct_count} problems")
        stats = {
            'total_solution_counts': total_solutions_count,
            'quant_solutions': quant_correct_count,
            'regular_model_solutions':regular_correct_count
        }
        results_file = os.path.join(args.output_dir, "all_results.json")
        with open(results_file, 'a') as f:
                json.dump(stats, f, indent=2)
        
        if no_correct_count > 0:
            if args.add_gold_answer:
                print(f"- Added golden answers for {gold_answer_count} problems with no correct solutions")
            else:
                print(f"- {no_correct_count} problems had no correct solution and were skipped")
    
    else:
        # 다른 strategy의 경우 기존 로직 사용
        pruned_solutions = solutions + quant_solutions
    
    # total_input_solutions = len(solutions) + len(quant_solutions)
    # print(f"Pruned from {total_input_solutions} to {len(pruned_solutions)} solutions")
    
    return pruned_solutions
def save_pruned_data(pruned_solutions, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    utils.save_solutions(pruned_solutions, output_file)
    print(f"Saved pruned data to {output_file}")