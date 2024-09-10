import json
import numpy as np
import pandas as pd

# useful constants
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


def load_json(pth):
    with open(pth, "r") as file:
        return json.load(file)


def idx_model_selection(equations: pd.DataFrame, model_selection: str) -> int:
    """Select an expression and return its index."""
    if model_selection == "accuracy":
        chosen_idx = equations["loss"].idxmin()
    elif model_selection == "best":
        threshold = 1.5 * equations["loss"].min()
        filtered_equations = equations.query(f"loss <= {threshold}")
        chosen_idx = filtered_equations["score"].idxmax()
    elif model_selection == "score":
        chosen_idx = equations["score"].idxmax()
    else:
        raise NotImplementedError(
            f"{model_selection} is not a valid model selection strategy."
        )
    return chosen_idx


def sample(method, b, num_samples):
    if method == "constant":  # const
        return np.full(num_samples, b)
    elif method == "uniform":  # (low, high)
        return np.random.uniform(low=b[0], high=b[1], size=num_samples)
    elif method == "normal":  # (mean, std)
        return np.random.normal(loc=b[0], scale=b[1], size=num_samples)
    elif method == "loguniform":  # logU(a, b) ~ exp(U(log(a), log(b))
        return np.exp(
            np.random.uniform(low=np.log(b[0]), high=np.log(b[1]), size=num_samples)
        )
    elif method == "lognormal":  # ln of var is normally distributed
        return np.random.lognormal(mean=b[0], sigma=b[1], size=num_samples)
    else:
        print("Not valid method")

def sample_equation(equation, bounds, num_samples, noise, add_extra_vars):
    out = []
    for var in bounds:
        if bounds[var] is None:  # goal
            continue
        out.append((var, sample(*bounds[var], num_samples=num_samples)))

    exp = equation.split(" = ")[1].replace("^", "**")
    exp_as_func = eval(f"lambda {','.join([x[0] for x in out])}: {exp}")

    Y = list()
    X_temp = np.transpose(np.stack([x[1] for x in out]))
    for i in range(num_samples):
        Y.append(exp_as_func(*list(X_temp[i])))
    Y = np.array(Y)
    Y = Y + np.random.normal(0, np.sqrt(np.square(Y).mean()) * noise, Y.shape)

    if add_extra_vars:
        total_vars = len(["x", "y", "z", "k", "j", "l", "m", "n", "p", "a", "b"])
        extra_vars = {
            chr(ord("A") + c): ("uniform", (1, 20))
            for c in range(total_vars - len(bounds) + 1)
        }
        for var in extra_vars:
            out.append((var, sample(*extra_vars[var], num_samples=num_samples)))


    np.random.shuffle(out)
    var_order = {"x" + str(i): out[i][0] for i in range(len(out))}
    X = np.transpose(np.stack([x[1] for x in out]))

    return X, Y, var_order

def sample_dataset(equations, num_samples, noise, add_extra_vars):
    dataset = []
    for (idx, (eq, bounds)) in equations:
        X, Y, var_order = sample_equation(eq, bounds, num_samples, noise, add_extra_vars=add_extra_vars)
        dataset.append((idx, (eq, (X, Y, var_order))))
    return dataset
