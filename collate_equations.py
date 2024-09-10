import os
import json
from glob import glob
import re

eq_number_regex = re.compile(r"-{20}\s*GT Equation #(\d+):(.+?)\s*-{20}", re.DOTALL)
eq_regex = re.compile(
    r"GT Equation #(?P<eqn_number>\d+):\s*(?P<gt_eq>.*?)\n"
    r"Pred Equation #\d+:\s*(?P<pred_eq>.*?)\n"
    r"Score:\s*(?P<score>[\d\.]+),\s*Loss:\s*(?P<loss>[\d\.]+),\s*Complexity:\s*(?P<complexity>\d+)\s*(?P<eqn_text>.*?)\n"
    r"-{20}", re.DOTALL)


# Loss: 0.14614905
loss_regex = re.compile(r"Loss: ([0-9.]+)")

def parse_equations(concat_summary):
    # return {int(eqn_number): dict(eqn_text=eqn_text) for (eqn_number, eqn_text) in sorted(eq_number_regex.findall(concat_summary))}
    return {int(eqn_number) : dict(eqn_number=int(eqn_number), gt_eq=gt_eq, pred_eq=pred_eq, score=float(score), loss=float(loss), complexity=int(complexity), eqn_text=eqn_text)
                    for (eqn_number, gt_eq, pred_eq, score, loss, complexity, eqn_text) in eq_regex.findall(concat_summary)}

def get_iterations(exp_dir):
    # Two cases:
    # case 1: exp_dir/n_iterations.txt is present and has more than 1 line that begins with -
    # case 2: exp_dir/exp_*/n_iterations.txt is present and has more than 1 line that begins with -
    if os.path.exists(f"{exp_dir}/n_iterations.txt"):
        all_iterations = []
        with open(f"{exp_dir}/n_iterations.txt", "r") as file:
            lines = file.readlines()
            lines = [line for line in lines if line.strip('\n- ').isdigit()]
        
        if len(lines) > 2:
            # batch mode
            for line in lines:
                num = line.strip('\n- ')
                if num.isdigit():
                    all_iterations.append(int(num))

            all_expt_idxs = sorted([int(os.path.basename(os.path.dirname(f))) for f in glob(f"{exp_dir}/*/n_iterations.txt")])
            all_expt_idxs.append(all_expt_idxs[-1] + 1)
            iterations = dict()
            cnt = 0
            for idx in range(len(all_iterations) // 2):
                iterations[all_expt_idxs[cnt]] = all_iterations[idx * 2 + 1]
                cnt += 1
            return iterations

    iterations = dict()
    for expt_dir in glob(f"{exp_dir}/*/n_iterations.txt"):
        eqn_idx = int(os.path.basename(os.path.dirname(expt_dir)))
        with open(expt_dir, "r") as file:
            lines = file.readlines()
            lines = [line for line in lines if line.strip('\n- ').isdigit()]
        
        if len(lines) > 1:
            num = lines[-1].strip('\n- ')
            if num.isdigit():
                iterations[eqn_idx] = int(num)
    return iterations

def get_equations(exp_dir):
    all_eqns = dict()
    for i in range(0, 10):
        files = glob(f"{exp_dir}/exp_*{i}/summary.txt")
        if len(files) == 0:
            continue
        concat_eqns = dict()
        for f in files:
            with open(f, "r") as file:
                summary = file.read()
            iterations = get_iterations(os.path.dirname(f))
            eqns = parse_equations(summary)
            for eqn, eqn_v in eqns.items():
                n_iterations = iterations.get(eqn, 0)
                eqns[eqn]['n_iterations'] = n_iterations

            concat_eqns = merge_expt(concat_eqns, eqns)
        all_eqns[i] = concat_eqns

    print_summary(all_eqns)
    return all_eqns


def print_summary(all_eqns):
    for i in range(0, 10):
        if i not in all_eqns:
            continue
        print(f"Experiment {i}: {len(all_eqns[i])} equations")


def merge_expt(e1, e2):
    dd = dict()
    all_keys1 = set(e1.keys()).union(set(e2.keys()))
    for kk in all_keys1:
        if kk in e2 and not isinstance(e2[kk], dict):
            e2[kk] = dict(eqn_text=e2[kk])
        if kk in e1 and not isinstance(e1[kk], dict):
            e1[kk] = dict(eqn_text=e1[kk])
        if kk in e1 and kk not in e2:
            dd[kk] = e1[kk]
        elif kk in e2 and kk not in e1:
            dd[kk] = e2[kk]
        else:
            # key is in both e1 and e2
            try:
                # loss1 = e1[kk]['n_iterations']
                loss1 = e1[kk]['loss']
                # loss1 = float(loss_regex.search(e1[kk]['eqn_text']).group(1))
            except:
                loss1 = 1e10
            try:
                # loss2 = e2[kk]['n_iterations']
                loss2 = e2[kk]['loss']
                # loss2 = float(loss_regex.search(e2[kk]['eqn_text']).group(1))
            except:
                loss2 = 1e10

            if loss1 == loss2 == 1e10:
                continue

            if loss1 == loss2:
                dd[kk] = e1[kk]
                continue
            # choose the one with lower loss
            dd[kk] = e1[kk] if loss1 < loss2 else e2[kk]
    return dd


def merge(d1, d2):
    d = dict()
    all_keys = set(d1.keys()).union(set(d2.keys()))
    for k in all_keys:
        if k in d1 and k not in d2:
            d[k] = d1[k]
        elif k in d2 and k not in d1:
            d[k] = d2[k]
        else:
            d[k] = merge_expt(d1[k], d2[k])
    return d


def merge_archived_results(dir) -> dict:
    """These results are in a different format and need special care."""

    def get_iterations(path):
        iterations = dict()
        for f in glob(f"{path}/*/n_iterations.txt"):
            eqn_idx = int(os.path.basename(os.path.dirname(f)))
            with open(f, "r") as file:
                lines = file.readlines()
            
            if len(lines):
                num = lines[-1].strip('\n- ')
                if num.isdigit():
                    iterations[eqn_idx] = int(num)
        return iterations

    path1 = os.path.join(dir, "logs/ablation_results_foundry_old/logs/exp17") 
    path2 = os.path.join(dir, "logs/ablation_results_foundry_old/logs/exp15")
    path3 = os.path.join(dir, "logs/ablation_results_foundry_old/logs/exp16")
    path4 = os.path.join(dir, "logs/ablation_results_foundry_old/logs/exp1")
    path5 = os.path.join(dir, "logs/ablation_results_foundry_old/logs/exp0")
    path6 = os.path.join(dir, "logs/ablation_results_foundry_old/logs/exp18")

    # summary1 = /ablation_results_foundry/exp17.log

    tasks = {
        path1: "Llama3 - 0.1% - Hints",
        path2: "Llama3 - 0.1%",
        path3: "Llama3 - 0.1% - No Concept Evol",
        path4: "Llama3 - 0.1% - No Vars",
        path5: "PySR",
        path6: "Llama3 - 0.1% - No Concept Crossover",
    }

    idx2iteration = {
        3: get_iterations(path4), # no vars
        4: get_iterations(path3), # no concept evolution
        5: get_iterations(path6), # no concept crossover
        6: get_iterations(path1), # hints
        7: get_iterations(path2), # 400 iterations
        8: get_iterations(path5), # pysr
    }
    # idx 0 = LaSR Feynman runs with p=0.10
    # idx 1 = LaSR Synthetic runs with p=0.01
    # idx 2 = LaSR Feynman runs with p=0.01 gpt-3.5-turbo-0125
    # idx 3 = Ablation experiments with no variables.
    # idx 4 = Ablation experiments with no concept evolution (No concepts at all)
    # idx 5 = Ablation experiments with no concept crossover (Concepts aren't refined)
    # idx 6 = Ablation experiments with hints
    # idx 7 = LaSR Feynman run with 400 iterations.
    # idx 8 = PySR

    idx2summary = {
        3: os.path.join(dir, f"ablation_results_foundry/{os.path.basename(path4)}.log"),
        4: os.path.join(dir, f"ablation_results_foundry/{os.path.basename(path3)}.log"),
        5: os.path.join(dir, f"ablation_results_foundry/{os.path.basename(path6)}.log"),
        6: os.path.join(dir, f"ablation_results_foundry/{os.path.basename(path1)}.log"),
        7: os.path.join(dir, f"ablation_results_foundry/{os.path.basename(path2)}.log"),
        8: os.path.join(dir, f"ablation_results_foundry/{os.path.basename(path5)}.log"),
    }
    all_eqns = dict()
    for i, path in idx2summary.items():
        if os.path.exists(path):
            with open(path, "r") as file:
                summary = file.read()
            eqns = parse_equations(summary)
        else:
            eqns = dict()

        merged_keys = list(set(eqns.keys()) | set(idx2iteration[i].keys()))
        for eqn in merged_keys:
            if eqn not in eqns:
                eqns[eqn] = dict(n_iterations=0, eqn_number=eqn, gt_eq="", pred_eq="", score=-1, loss=-1, complexity=-1, eqn_text="")
            n_iterations = idx2iteration[i].get(eqn, 0)
            eqns[eqn]['n_iterations'] = n_iterations

        all_eqns[i] = eqns
    return all_eqns

if __name__ == "__main__":
    if os.path.exists("all_equations.json"):
        with open("all_equations.json", "r") as file:
            old_eqns = json.load(file)
            old_eqns = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in old_eqns.items()}
    all_eqns1 = get_equations("./dataset_runs")
    print()
    dataset_runs = merge(old_eqns, all_eqns1)
    print_summary(dataset_runs)

    with open("all_equations.json", "w") as file:
        # sort all_eqns
        json.dump(dataset_runs, file, indent=4)
    
