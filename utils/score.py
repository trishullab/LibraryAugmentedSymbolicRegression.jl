

import numpy as np
from utils.optim import optimize_N_constants
from utils.context import context_loss

sin = np.sin
cos = np.cos
pi = np.pi

def data_loss(program, X, Y, num_constant_samples=100, num_eval_samples=1000):
    try:
        sol_const, program_const = optimize_N_constants(program, X, Y, num_constant_samples)
    except:
        return None, None
    if sol_const is None:
        return None, None

    data_var = np.var(sol_const, axis=0).mean()
    if data_var == 0:
        data_var = 0.001
    args = np.mean(sol_const, axis=0).flatten()

    exp_as_func = eval('lambda x, y, z, args: ' + program_const)

    idx = np.random.choice(np.arange(len(X)), num_eval_samples)
    pred_Y = [exp_as_func(*X[i], args) for i in idx]
    if np.var(pred_Y) < 1e-10: # can't just be constant predictions everytime
        return None, None

    data_mse = (np.square(np.array(pred_Y) - Y[idx])).mean()
    return data_mse, data_var

def score(program, context, X, Y):
    data_mse, data_var = data_loss(program, X, Y)
    con_loss = context_loss(program, context)

    return data_mse, data_var, con_loss # combine somehow for fitness score

def test():
    program = "A*(x**2 + y**2) + B*z + C*sin(D*x) + J*cos(F*y) + G*z**2 + H"
    context = "Note that x and y are radii."

    eq = "(4 * pi * z) / ((1/x) - (1/y))"
    num_samples = 2000
    num_vars = 3

    X = 10 * np.random.rand(num_samples, num_vars) - 5
    Y = list()
    points = ""
    for sample in range(num_samples):
        x, y, z = X[sample]
        k = eval(eq)
        Y.append(k)
    Y = np.array(Y)

    import time
    start_time = time.time()
    print(score(program, context, X, Y))
    print(time.time() - start_time)