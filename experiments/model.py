from contextlib import redirect_stdout
import copy
import os
import numpy as np
from pysr import PySRRegressor
from .utils import idx_model_selection

pi = np.pi
cos = np.cos
sin = np.sin
sqrt = np.sqrt
exp = np.exp
arcsin = np.arcsin
arccos = np.arccos
log = np.log
ln = np.log
tanh = np.tanh

custom_loss = """
function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    return sum( (1000 .* (prediction .- dataset.y) ) .^ 2) / dataset.n
end
"""

def eval_equation(idx, eq, X, Y, var_order, args, llm_options, hints, log_files, log_file_path, name):
    if not os.path.exists(f"{log_file_path}/{idx}/"):
        os.makedirs(f"{log_file_path}/{idx}/", exist_ok=True)


    for fp in log_files:
        f = open(f"{log_file_path}/{idx}/{fp}", "w+")
        f.write(f"Dataset {name}\nEquation: {eq}\n----------------\n")
        f.close()

    set_llm_options = llm_options
    set_llm_options['llm_recorder_dir'] = f"{log_file_path}/{idx}/"
    set_llm_options["llm_context"] = (
        hints[idx - 1] if hints is not None else ""
    )
    if args.ablation_mode == "no-variables":
        default_var_names = [
            "x",
            "y",
            "z",
            "k",
            "j",
            "l",
            "m",
            "n",
            "p",
            "a",
            "b",
        ]
        set_llm_options["var_order"] = {
            k: default_var_names[i] for i, k in enumerate(var_order.keys())
        }
    else:
        set_llm_options["var_order"] = var_order

    if args.use_llm:
        model = PySRRegressor(
            niterations=args.num_iterations,
            ncyclesperiteration=550,
            populations=15,
            population_size=33,
            maxsize=30,
            binary_operators=["+", "*", "-", "/", "^"],
            unary_operators=[
                "exp",
                "log",
                "sqrt",
                "sin",
                "cos",
            ],
            full_objective=custom_loss,
            early_stop_condition=f"f(loss, complexity) = (loss < {format(float(args.early_stopping_condition), 'f')})"
            if args.early_stopping_condition
            else None,
            verbosity=0,
            temp_equation_file=True,
            tempdir="pysr_runs",
            delete_tempfiles=True,
            llm_options=set_llm_options,
            weight_randomize=0.1,
            should_simplify=True,
            constraints={
                "sin": 10,
                "cos": 10,
                "exp": 20,
                "log": 20,
                "sqrt": 20,
                "pow": (-1, 20),
            },
            nested_constraints={
                "sin": {"sin": 0, "cos": 0},
                "cos": {"sin": 0, "cos": 0},
                "exp": {"exp": 0, "log": 0},
                "log": {"exp": 0, "log": 0},
                "sqrt": {"sqrt": 0},
            },
        )
    else:
        model = PySRRegressor(
            niterations=args.num_iterations,
            timeout_in_seconds=360,
            ncyclesperiteration=550,
            populations=15,
            population_size=33,
            maxsize=30,
            binary_operators=["+", "*", "-", "/", "^"],
            unary_operators=[
                "exp",
                "log",
                "sqrt",
                "sin",
                "cos",
            ],
            full_objective=custom_loss,
            early_stop_condition=f"f(loss, complexity) = (loss < {format(float(args.early_stopping_condition), 'f')})"
            if args.early_stopping_condition
            else None,
            verbosity=0,
            temp_equation_file=True,
            tempdir="pysr_runs",
            delete_tempfiles=True,
            weight_randomize=0.1,
            nested_constraints={
                "sin": {"sin": 0, "cos": 0},
                "cos": {"sin": 0, "cos": 0},
                "exp": {"exp": 0, "log": 0},
                "log": {"exp": 0, "log": 0},
                "sqrt": {"sqrt": 0},
            },
            constraints={
                "sin": 10,
                "cos": 10,
                "exp": 20,
                "log": 20,
                "sqrt": 20,
                "pow": (-1, 20),
            },
        )

    model.fit(X, Y)
    chosen_idx = idx_model_selection(
        model.equations_, model.model_selection
    )
    pred_eq = str(model.equations["equation"][chosen_idx])
    score = model.equations["score"][chosen_idx]
    loss = model.equations["loss"][chosen_idx]
    complexity = model.equations["complexity"][chosen_idx]

    for x in reversed(list(var_order.keys())):
        pred_eq = pred_eq.replace(x, var_order[x])
    pred_eq = eq.split(" = ")[0] + " = " + pred_eq

    print("-" * 20)
    print(f"GT Equation #{idx}: ", eq)
    print(f"Pred Equation #{idx}: ", pred_eq)
    print(f"Score: {score}, Loss: {loss}, Complexity: {complexity}")
    print("")
    print(f"Input Variable Order: {var_order}")
    print("PySR Model: ", model)
    print("-" * 20)
    print("\n")
    return


def eval_dataset(
    dataset,
    args,
    llm_options,
    start_idx,
    end_idx,
    hints,
    log_file_path,
    log_files,
):
    summary_path = os.path.join(log_file_path, "summary.txt")
    with open(summary_path, "w") as summary_path_fp:
        with redirect_stdout(summary_path_fp):
            print("-" * 20)
            print("Starting Evaluation\n\n")
            name = f"Feynman Equations - {args.num_iterations} iterations - Prompt Evol = {llm_options['prompt_evol']} - Prompt concepts = {llm_options['prompt_concepts']}, LLM Mutate = {llm_options['weights']['llm_mutate']}, LLM Crossover = {llm_options['weights']['llm_crossover']}, LLM Gen Random = {llm_options['weights']['llm_gen_random']}, Num Pareto Context = {llm_options['num_pareto_context']}"

            for idx, (eq, (X, Y, var_order)) in dataset:
                if idx < start_idx or idx >= end_idx:
                    continue
                eval_equation(
                    idx=idx,
                    eq=eq,
                    X=X,
                    Y=Y,
                    var_order=var_order,
                    args=args,
                    llm_options=copy.deepcopy(llm_options),
                    hints=hints,
                    log_files=log_files,
                    log_file_path=log_file_path,
                    name=name
                )
                summary_path_fp.flush()

