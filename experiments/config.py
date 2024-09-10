import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_llm", action="store_true", required=False, help="Flag to enable the use of a large language model")
    parser.add_argument("--use_prompt_evol", action="store_true", required=False, help="Flag to enable the use of prompt evolution")
    parser.add_argument("--disable_prompt_concepts", action="store_true", required=False, help="Flag to enable the use of prompt concepts")
    parser.add_argument("--use_hints", action="store_true", required=False, help="Flag to enable the use of llm context hints")
    parser.add_argument("--model", type=str, required=False, help="Model to be used")
    parser.add_argument("--resume_from", type=str, required=False, default=None, help="all_equations.json file generated from all_equations.py")
    parser.add_argument("--model_url", type=str, required=False, default="http://localhost:11440/v1/", help="URL of the model endpoint")
    parser.add_argument("--max_tokens", type=int, required=False, default=1024, help="Maximum number of tokens for the model response")
    parser.add_argument("--num_pareto_context", type=int, required=False, default=3, help="Number of equations to sample from the pareto front")
    parser.add_argument("--num_iterations", type=int, required=False, default=40, help="Number of iterations to run the experiment")
    parser.add_argument("--num_workers", type=int, required=False, default=0, help="Number of equations to evaluate in parallel (Defualts to 0; sequential evaluation)")
    parser.add_argument("--idea_threshold", type=int, required=False, default=30, help="Threshold for the number of ideas to generate")
    parser.add_argument("--api_key", type=str, required=False, default="data/api.key", help="Location of the API key")
    parser.add_argument("--llm_mutate_weight", type=float, required=False, default=0.05, help="Weight for mutation operations in prompt evolution")
    parser.add_argument("--llm_crossover_weight", type=float, required=False, default=0.05, help="Weight for crossover operations in prompt evolution")
    parser.add_argument("--llm_gen_random_weight", type=float, required=False, default=0.05, help="Weight for generating random prompts in prompt evolution")
    parser.add_argument("--llm_recorder_dir", type=str, required=False, default="lasr_runs", help="Directory to save the records of the large language model")
    parser.add_argument("--exp_idx", type=int, required=True, help="Experiment index")
    parser.add_argument("--hints_path", type=str, required=False, help="Path to the hints file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--prompts_path", type=str, required=False,  help="Path to the directory containing the prompts.")
    parser.add_argument("--start_idx", type=int, required=False, default=0, help="Index to start the evaluation")
    parser.add_argument("--end_idx", type=int, required=False, default=None, help="Index to end the evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use", choices=["Feynman", "Synthetic", "SRSD", "Bigbench"])
    parser.add_argument("--dataset_order", type=str, required=False, choices=["Easy SRSD", "Medium SRSD", "Hard SRSD"], help="Order of the SRSD dataset")
    parser.add_argument("--noise", type=float, required=False, default=0.0001, help="Noise to add to the dataset")
    parser.add_argument("--num_samples", type=int, required=False, default=2000, help="Number of samples to generate")
    parser.add_argument("--ablation_mode", type=str, required=False, default="None", help="Ablation mode")
    parser.add_argument("--early_stopping_condition", type=str, required=False, default='None', help="Early stopping condition for the model")
    args = parser.parse_args()
    args.early_stopping_condition = eval(args.early_stopping_condition)
    return args

def setup_logging(args):
    log_file_path = os.path.join(args.llm_recorder_dir, f"exp_{args.exp_idx}")
    log_files = ["ideas.txt", "mutate.txt", "crossover.txt", "gen_random.txt", "tree-expr.txt", "llm-calls.txt", "n_iterations.txt"]

    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    
    if log_file_path[-1] != "/":
        log_file_path += "/"

    return log_file_path, log_files

def setup_api_key(args):
    with open(args.api_key, "r") as file:
        return file.read().strip()