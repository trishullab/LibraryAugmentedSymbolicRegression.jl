# LLM-SR
import numpy as np
import ast
from openai import OpenAI
import tiktoken
from copy import deepcopy

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations, 
    implicit_multiplication,
    convert_xor
)
from sympy import symbols, nsolve

# Step 0: pick an equation, generate data from it, manually write a context for it 
eq = None
with open("pysr_dataset.log", "r") as f:
    equations = f.readlines()
    eq = equations[0]

# eq = "(4 * pi * z) / ((1/x) - (1/y))" # similar to spherical capacitance formula, we want something twice the complexity for something interesting
eq = "(z) / ((1/x) - (1/y))"
context = "Note that x and y are radii." # stuff with asymptotic behavior (radius goes to infinity), use radial symmetry

print("Equation: ", eq)
sin = np.sin
cos = np.cos
pi = np.pi

def sqrt(x): # wrapper for sqrt so no NaNs occur
    if x < 0:
        return 0
    else:
        return np.sqrt(x)

num_samples = 50
num_batches = 5
num_prompt_samples = 50
num_vars = 3
X = 10 * np.random.rand(num_samples, num_vars) - 5
Y = list()
points = ""
for sample in range(num_samples):
    if sample < num_prompt_samples:
        x, y, z = X[sample].round(2)
        k = round(eval(eq), 6)
        Y.append(k)
        points += f"f({x}, {y}, {z}) = {k}\n"
    else:
        x, y, z = X[sample]
        k = eval(eq)
        Y.append(k)
Y = np.array(Y)

# Step 1: Prompt GPT-4 to give initial guess (prompting) and convert into a program tree
with open('api.key', 'r') as fp:
    secret_api_key = fp.read().strip()
client = OpenAI(
    api_key=secret_api_key,
)
enc = tiktoken.encoding_for_model("gpt-4")

def get_chat_response(current_context, temp=0.5, long=False):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=current_context,
        temperature=temp,
        max_tokens=512 if long else 256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response
    # response = openai.ChatCompletion.create(
    # model="gpt-4",
    # messages=current_context,
    # temperature=0.5,
    # max_tokens=256,
    # top_p=1,
    # frequency_penalty=0,
    # presence_penalty=0)
    # return response

start_ctx = [
    {
      "role": "system",
      "content": "Given a function f(x, y, z) = k, your task is to give an approximation for the expression of f given the following data points. It doesn't need to be accurate just try your best. Please abstract away any constants and represent them with capital letters. You cannot use I, E, O, S or Q as constants."
    }, # need to restrict possible constants so it doesn't overlap w/ var names?
]

curr_ctx = deepcopy(start_ctx)

user_prompt = "Here are the data points: \n" + points + "\n Operate under the following assumptions: "+context+". Please complete your task to the best of your ability and provide reasoning behind your choices."

curr_ctx += [dict(role='user', content=user_prompt)]
response = get_chat_response(current_context=curr_ctx, long=True)

assert len(response.choices), 'No Response!'
chat_response = response.choices[0].message
chat_response_content = chat_response.content
print("GPT-4 Full Response: ", chat_response_content)

curr_ctx = deepcopy(start_ctx)
curr_ctx += [dict(role='user', content=chat_response_content), dict(role='user', content="Respond with your answer in the form f(x, y, z) = k. For example, f(x, y, z) = a*x*y + b*y*z. Do not respond with any other text. DO NOT GIVE YOUR REASONING.")]
response = get_chat_response(current_context=curr_ctx, temp=0.1)

assert len(response.choices), 'No Response!'
chat_response = response.choices[0].message
chat_response_content = chat_response.content.strip()
print("Extracted Program: ", chat_response_content)

# chat_response_content = "f(x, y, z) = A*(x^2 + y^2) + B*z + C*sin(D*x) + J*cos(F*y) + G*z^2 + H"

def extract_program(res):
    # convert text to a program
    init_eq = res.split("f(x, y, z) = ")[-1].strip()
    return ast.parse(init_eq, mode='eval')


def prob(root, X, Y, context):
    p = root.body
    p_str = ast.unparse(p)

    # compare Y and pred_Y
    # do it probablistically by using possible values for constants (variance in the fitted values from a regression?)
    # also take into account regression solution wouldn't be perfect, use the error from that to weight it too
    x, y, z = symbols("x y z")
    constants_list = list({node.id for node in ast.walk(root) if isinstance(node, ast.Name)} - {'x','y','z','sin','cos'})
    constants = symbols("".join(constants_list))

    expr = parse_expr(p_str, transformations=standard_transformations + (implicit_multiplication,) + (convert_xor,))
    size = len(constants_list)
    assert size >= len(constants_list) # otherwise infinite solutions

    sol = list()
    sol = [np.ones(len(constants_list)), np.ones(len(constants_list))]
    idx = np.arange(num_samples)
    while len(sol) < 2:
        for batch in range(num_samples//size):
            if len(sol) >= num_batches:
                break
            sample_expr = list()
            for sample in range(size):
                xt, yt, zt = X[idx[batch * size + sample]]
                sample_expr.append(expr.subs([(x,xt),(y,yt),(z,zt)]) - Y[idx[batch * size + sample]])

            try:
                sol.append(np.array(nsolve(sample_expr, constants_list, [float(i) for i in range(len(constants_list))], verbose=False, verify=False)))
            except:
                pass
        np.random.shuffle(idx)

    data_var = np.var(sol, axis=0).mean()
    if data_var == 0:
        data_var = 0.001
    avg = np.mean(sol, axis=0).flatten()

    expr = expr.subs(list(zip(constants_list, avg.tolist())))
    # print(expr)
    pred_Y = list()
    for sample in range(num_samples):
        xt, yt, zt = X[sample]
        pred_Y.append(expr.subs([(x,xt),(y,yt),(z,zt)]))
    
    # print("Generated")

    data_mse = (np.square(np.array(pred_Y) - Y)).mean()
    # print(data_mse)

    # use LLM log probs to see adherence of p to context
    # for program up to each point, log prob of using the given value/var (next token)
    # need to generalize to different uses of constants, good to keep them consistent to this LLM
    import time
    start_time = time.time()
    curr_ctx = [
        {
        "role": "system",
        "content": "Given a function f(x, y, z) = k, your task is to see how well it fits with the following assumptions: "+context+". Please abstract away any constants and represent them with capital letters. You cannot use I, E, O, S or Q as constants."
        }, # need to restrict possible constants so it doesn't overlap w/ var names?
    ]
    curr_ctx += [dict(role='user', content="Respond with your answer in the form f(x, y, z) = k. For example, f(x, y, z) = a*x*y + b*y*z. Do not respond with any other text. DO NOT GIVE YOUR REASONING."), dict(role='user', content="")]
    
    context_loss = 0.0
    token_enc = enc.encode(p_str)
    for token in token_enc:
        curr_ctx[-1]['content'] += enc.decode([token])
    
        response = client.chat.completions.create(
            model="gpt-4",
            messages=curr_ctx,
            temperature=0.5,
            max_tokens=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            logprobs=True, 
            top_logprobs=5,
        )
        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        min_logprob = 0.0
        for obj in top_logprobs:
            if enc.encode(obj.token) == token:
                context_loss += obj.logprob
                min_logprob = 0.0
                break
            min_logprob = min(obj.logprob, min_logprob)
        
        context_loss += min_logprob

    # print(data_mse, data_var, context_loss)
    return data_mse, data_var, -context_loss # flip sign


def tot(program, N): # Tree of Thought (LLM propositions)
    curr_ctx = [
        {
        "role": "system",
        "content": "Given a function f(x, y, z) = "+ast.unparse(program)+". Your job is to propose "+str(N)+" functions similar to it that fit with the following assumptions: "+context+". Please abstract away any constants and represent them with capital letters. You cannot use I, E, O, S or Q as constants."
        }, # need to restrict possible constants so it doesn't overlap w/ var names?
    ]
    curr_ctx += [dict(role='user', content="Respond with your answer in a numbered list with each entry in the form f(x, y, z) = k. For example, f(x, y, z) = a*x*y + b*y*z. Do not respond with any other text. DO NOT GIVE YOUR REASONING.")]

    response = get_chat_response(curr_ctx, long=True, temp=0.8).choices[0].message.content.strip()
    
    out = list()
    for prop_eq in response.split("\n\n"):
        prop_eq = prop_eq.split(" = ")[-1].strip().strip('.')
        out.append(ast.parse(prop_eq, mode='eval'))
    
    return out

def ga(program, N):
    pass

program = extract_program(chat_response_content)

epochs = 100
num_props = 5
for e in range(epochs):
    # Step 2: Apply a lateral program synthesis technique to change program tree
    propositions = tot(program, num_props)

    # Step 3: Evaluate each program on data + adherence to context w/ LLM
    init_data_mse_prob, init_data_var_prob, init_context_prob = prob(program, X, Y, context)
    scores = []
    for i,p in enumerate(propositions):
        data_mse_prob, data_var_prob, context_prob = prob(p, X, Y, context)
        score = ((data_mse_prob - init_data_mse_prob) / init_data_mse_prob) + ((data_var_prob - init_data_var_prob) / init_data_var_prob) + ((context_prob - init_context_prob) / init_context_prob)  # need to figure out how to combine better
        scores.append((score, i, p))

    # Step 4: Choose top one (deterministic for now), repeat to step 2 until it converges (how to measure that)
    _, _, program = max(scores)
    print(f"Program after {e+1} steps: ", ast.unparse(program.body))


# Step 5: Fit constants using regression

# for each data point, substitute into equation, use regression to get values for constant
# take mean of these values (instead of variance for scoring)
p_str = ast.unparse(program.body)

x, y, z = symbols("x y z")
constants_list = list({node.id for node in ast.walk(program) if isinstance(node, ast.Name)} - {'x','y','z','sin','cos'})
constants = symbols("".join(constants_list))

expr = parse_expr(p_str, transformations=standard_transformations + (implicit_multiplication,) + (convert_xor,))
size = len(constants_list)
assert size >= len(constants_list) # otherwise infinite solutions

sol = list()
idx = np.arange(num_samples)
while len(sol) == 0:
    for batch in range(num_samples//size):
        if len(sol) >= num_batches:
            break
        sample_expr = list()
        for sample in range(size):
            xt, yt, zt = X[idx[batch * size + sample]]
            sample_expr.append(expr.subs([(x,xt),(y,yt),(z,zt)]) - Y[idx[batch * size + sample]])

        try:
            sol.append(np.array(nsolve(sample_expr, constants_list, [float(i) for i in range(len(constants_list))], verbose=False, verify=False)))
        except:
            pass
    np.random.shuffle(idx)

    

data_var = np.var(sol, axis=0).flatten()
avg = np.mean(sol, axis=0).flatten()

expr = expr.subs(list(zip(constants_list, avg.tolist())))

# Step 6: Evaluate performance on train + val dataset, compare with ground truth
print("Ground Truth: ", eq)
print("Final Equation: ", expr)

pred_Y = list()
for sample in range(num_samples):
    xt, yt, zt = X[sample]
    pred_Y.append(expr.subs([(x,xt),(y,yt),(z,zt)]))

train_mse = (np.square(np.array(pred_Y) - Y)).mean()

# compare pred_Y w/ Y, do same for a val dataset

X_val = 10 * np.random.rand(num_samples, num_vars) - 5
Y_val = list()
pred_Y = list()
for sample in range(num_samples):
    x, y, z = X_val[sample]
    k = eval(eq)
    xt, yt, zt = x, y, z
    x, y, z = symbols("x y z")
    pred_Y.append(expr.subs([(x,xt),(y,yt),(z,zt)]))
    Y_val.append(k)

val_mse = (np.square(np.array(pred_Y) - np.array(Y_val))).mean()

print("Train MSE: ", train_mse)
print("Val MSE: ", val_mse)

