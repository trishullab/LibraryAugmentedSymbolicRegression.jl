from time import sleep
import json
from .config import parse_args, setup_logging, setup_api_key
from .dataset import feynman_dataset, synthetic_dataset, srsd_dataset
from .model import eval_dataset


def main():
    args = parse_args()
    log_file_path, log_files = setup_logging(args)
    api_key = setup_api_key(args)
    print(f"Starting experiment {args.exp_idx}")
    print(f"Running on model {args.model} @ {args.model_url}")

    if args.resume_from is not None:
        with open(args.resume_from, "r") as file:
            all_exp_resume_data = json.load(file)
            if str(int(args.exp_idx) % 10) in all_exp_resume_data:
                resume_data = all_exp_resume_data[str(int(args.exp_idx) % 10)]
                processed_equations = list(map(int, resume_data.keys()))
            else:
                processed_equations = []

    match args.dataset:
        case "Feynman":
            equations_to_keep = set(range(1, 101))  # keep all 100
            if int(args.exp_idx) % 10 == 0:
                equations_to_keep = {19, 50, 68, 86, 87, 91, 2, 9, 17, 18, 24, 30, 56, 64, 65, 67, 71, 72, 3, 5, 6, 29, 33, 44, 80, 89, 90, 99}

            equations_to_keep -= {26, 31, 81}  # remove 26, 31, 81
            if args.resume_from is not None:
                equations_to_keep -= set(processed_equations)

            end_idx = args.end_idx if args.end_idx else 100
            equations_to_keep = set(filter(lambda x: args.start_idx <= x < end_idx, equations_to_keep))
            print("Running {n} equations".format(n=len(equations_to_keep)))
            sleep(3)
            dataset, all_hints = feynman_dataset(
                dataset_path=args.dataset_path,
                equations_to_keep=equations_to_keep,
                num_samples=args.num_samples,
                noise=args.noise,
                use_hints=args.use_hints,
                hints_path=args.hints_path,
            )
        case "Synthetic":
            equations_to_keep = set(range(0, 42))  # keep all 42
            if args.resume_from is not None:
                equations_to_keep -= set(processed_equations)

            end_idx = args.end_idx if args.end_idx else 42

            print("Running {n} equations".format(n=len(equations_to_keep)))
            sleep(3)

            dataset, all_hints = synthetic_dataset(
                num_samples=args.num_samples,
                noise=args.noise,
                use_hints=args.use_hints,
                hints_path=args.hints_path,
                equations_to_keep=equations_to_keep,
            )
        case "SRSD":
            assert (
                args.equations_order
            ), """Please provide equations order for SRSD. One of
            'Easy SRSD', 'Medium SRSD', 'Hard SRSD'""".replace("\n", " ")
            dataset, all_hints = srsd_dataset(
                equations_order=args.equations_order,
                num_samples=args.num_samples,
                noise=args.noise,
                use_hints=args.use_hints,
                hints_path=args.hints_path,
            )
            end_idx = args.end_idx if args.end_idx else 100
        case "Bigbench":
            raise NotImplementedError("Bigbench dataset is not implemented yet")
        case _:
            raise ValueError(f"Dataset {args.dataset} is not supported")

    assert args.prompts_path.endswith("/")
    assert not args.ablation_mode == "no-concepts" or (
        not args.use_prompt_evol
    ), "Ablation mode 'no-concepts' requires the deactivation of prompt evolution"
    assert (
        not args.ablation_mode == "no-crossover"
        or (args.disable_prompt_concepts and (not args.use_prompt_evol))
    ), "Ablation mode 'no-crossover' requires the deactivation of prompt concepts and prompt evolution"
    # redirect stdout to log file
    eval_dataset(
        dataset,
        args,
        llm_options=dict(
            active=args.use_llm,
            weights=dict(
                llm_mutate=args.llm_mutate_weight,
                llm_crossover=args.llm_crossover_weight,
                llm_gen_random=args.llm_gen_random_weight,
            ),
            prompt_evol=args.use_prompt_evol,
            prompt_concepts=(not args.disable_prompt_concepts),
            num_pareto_context=args.num_pareto_context,
            api_key=api_key,
            model=args.model,
            api_kwargs=dict(
                max_tokens=args.max_tokens,
                url=args.model_url,
            ),
            http_kwargs=dict(
                retries=5,
                readtimeout=360,
            ),
            llm_recorder_dir=log_file_path,
            idea_threshold=args.idea_threshold,
            prompts_dir=args.prompts_path,
            is_parametric=False,
        ),
        start_idx=args.start_idx,
        end_idx=end_idx,
        hints=all_hints,
        log_file_path=log_file_path,
        log_files=log_files,
    )


if __name__ == "__main__":
    main()
