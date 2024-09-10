import matplotlib.pyplot as plt
import seaborn as sns

path1 = "/home/ubuntu/llm-sr/ablation_results_foundry/logs/exp17/" # 14
path2 = "/home/ubuntu/llm-sr/ablation_results_foundry/logs/exp15/" # 15
path3 = "/home/ubuntu/llm-sr/ablation_results_foundry/logs/exp16/" # 16
path4 = "/home/ubuntu/llm-sr/ablation_results_foundry/logs/exp1/"
path5 = "/home/ubuntu/llm-sr/ablation_results_foundry/logs/exp0/"
path6 = "/home/ubuntu/llm-sr/ablation_results_foundry/logs/exp18/"

# NOTE: do one graph w/ path2 - 5 and then one graph with path 1 + 2 and 5
# x-axis is number of iterations, y-axis is solution rate
tasks = {
    path1: "Llama3 - 0.1% - Hints",
    path2: "Llama3 - 0.1%",
    # path3: "Llama3 - 0.1% - No Concept Evolution",
    # path6: "Llama3 - 0.1% - No Concept Crossover",
    # path4: "Llama3 - 0.1% - No Vars",
    path5: "PySR",
}

num_steps = 40
num_eq = 100

def process(path, N=100):
    out = dict()
    for i in range(N):
        file = open(path + str(i) + "/n_iterations.txt", "r")
        lines = file.readlines()

        if len(lines) < 4:
            res = ' '
        else:
            res = lines[3]

        if res[0] != '-':
            res = 1
        else:
            res = int(res[1:].strip())

        if res not in out:
            out[res] = 0
        out[res] += 1

        file.close()

    y = list()
    x = 0
    for v in range(num_steps):
        if v in out:
            x += out[v]
        y.append(x)
    return y

fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

markers = ['o',  '>', 'p', 'h', '*']

stepsize = 3
colors = sns.color_palette("colorblind", len(tasks))
for i, x in enumerate(tasks):
    y_values = process(x, N=num_eq)
    ax.plot(
        list(range(num_steps)), 
        y_values, 
        label=tasks[x], 
        color=colors[i], 
        linewidth=3
    )
    ax.plot(
        list(range(0, num_steps, stepsize)),
        y_values[0:num_steps:stepsize],
        color=colors[i], 
        marker=markers[i % len(markers)], 
        linestyle='None', 
        # label=tasks[x], 
        markersize=10
    )

ax.set_ylim(0, num_eq)
ax.set_xlim(0, num_steps )
ax.legend(fontsize=22, loc='upper left', title='Tasks', title_fontsize='22')
ax.set_xlabel("Iterations", fontsize=26, fontweight='bold')
ax.set_ylabel("Equation Solved under MSE", fontsize=26, fontweight='bold')

ax.grid(True, which='both', linestyle='--', linewidth=0.25)
ax.tick_params(axis='both', which='major', labelsize=20)

plt.tight_layout()
plt.savefig("iter_graph.png", format='png')
plt.close()
