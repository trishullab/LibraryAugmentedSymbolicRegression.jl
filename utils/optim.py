# All constant optimization

import re
import ast
import numpy as np
from scipy.optimize import minimize


sin = np.sin
cos = np.cos
pi = np.pi

def optimize_constants(program, X, Y, x0):
    (x, y, z) = X
    exp_as_func = eval('lambda x,y,z,args: ' + program + ' - '+str(Y))

    def opt_func(args):
        return exp_as_func(x,y,z,args)
    
    try:
        return minimize(opt_func, x0, method='BFGS')
    except (Exception, Warning) as e:
        return None


def optimize_N_constants(program, X, Y, N):
    root = ast.parse(program, mode='eval')
    constants_list = list({node.id for node in ast.walk(root) if isinstance(node, ast.Name)} - {'x','y','z','sin','cos'})
    for i,v in enumerate(constants_list):
        program = re.sub(r'\b'+v+r'\b', f"args[{i}]", program)

    program = re.sub(r'\^', '**', program)
    program = re.sub(r'\]\(', ']*(', program)
    
    N = min(N, len(X))
    idx = np.random.choice(np.arange(len(X)), N)
    x0 = np.ones(len(constants_list))
    sol = list()
    rej = list()
    for i in idx:
        res = optimize_constants(program, X[i], Y[i], x0)
        if res is None:
            return None, None
        if res.success:
            sol.append(res.x)
        else:
            rej.append(res.x)

    if len(sol) < 2:
        sol += rej[: 2 - len(sol)]

    return sol, program
    