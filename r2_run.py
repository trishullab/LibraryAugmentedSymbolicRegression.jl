import numpy as np
from sklearn.metrics import r2_score

def sample(method, b, num_samples):
    if method == "constant": # const
        return np.full(num_samples, b)
    elif method == "uniform": # (low, high)
        return np.random.uniform(low=b[0], high=b[1], size=num_samples)
    elif method == "normal": # (mean, std)
        return np.random.normal(loc=b[0], scale=b[1], size=num_samples)
    elif method == "loguniform": # logU(a, b) ~ exp(U(log(a), log(b))
        return np.exp(np.random.uniform(low=np.log(b[0]), high=np.log(b[1]), size=num_samples))
    elif method == "lognormal": # ln of var is normally distributed
        return np.random.lognormal(mean=b[0], sigma=b[1], size=num_samples)
    else:
        print("Not valid method")

eqs = []
exp_pred = []
import json
with open('all_equations.json', 'r') as f:
    all_equations = json.load(f)
    all_equations = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in all_equations.items()}

exp_id = 1
data = all_equations[exp_id]
for d in data:
    eqs.append(data[d]['gt_eq'])
    exp_pred.append(data[d]['pred_eq'])
    


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


dataset = list()

for id,eq in enumerate(eqs):
    bounds = {"x" + str(i) : ("uniform", (1,10)) for i in range(1,6)}
    dataset.append((id, (eq, bounds)))

dataset.sort() # sort ids

num_samples = 2000
r2 = list()

take = [False, False, True, False, False, False, False, False, True, False, False, True, False, False, True, False, False, False, False, True, True, False, False, False, False, True, False, False, False, False, False, True, True, False, False, True, False, True, False, True, False]

for (idx,(eq, bounds)) in dataset:
    if not take[idx]:
        continue
    out = []
    for var in bounds:
        if bounds[var] is None: # goal
            continue 
        out.append((var, sample(*bounds[var], num_samples=num_samples)))

    total_vars = len(["x","y","z","k","j","l","m","n","p","a","b"])
    extra_vars = {chr(ord('A') + c) : ("uniform", (1,20)) for c in range(total_vars - len(bounds) + 1)}

    for var in extra_vars:
        out.append((var, sample(*extra_vars[var], num_samples=num_samples)))

    import IPython; IPython.embed()
    expr = eq.split(' = ')[1].replace('^', '**')
    expr = expr.replace('y', 'x')
    exp_as_func_gt = eval(f"lambda {','.join([x[0] for x in out])}: {expr}")

    expr_pred = exp_pred[idx].split(' = ')[1].replace('^', '**')
    expr_pred = expr_pred.replace('y', 'x')
    exp_as_func_pred = eval(f"lambda {','.join([x[0] for x in out])}: {expr_pred}")

    Y = list()
    YP = list()
    X_temp = np.transpose(np.stack([x[1] for x in out]))
    for i in range(num_samples):
        Y.append(exp_as_func_gt(*list(X_temp[i])))
        YP.append(exp_as_func_pred(*list(X_temp[i])))
    Y = np.array(Y)
    YP = np.array(YP)

    x = r2_score(Y, YP)
    # if x < 0: x = 0
    r2.append(x)

print(r2)
print(sum(r2) / len(r2))