# All helper functions for working w/ GPT-4

from openai import OpenAI
import tiktoken
from copy import deepcopy

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
        max_tokens=1024 if long else 256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response


def context_loss(program, context):
    curr_ctx = [
        {
        "role": "system",
        "content": "Given a function f(x, y, z) = k, your task is to see how well it fits with the following assumptions: "+context+". Please abstract away any constants and represent them with capital letters. You cannot use I, E, O, S or Q as constants."
        }, # need to restrict possible constants so it doesn't overlap w/ var names?
    ]
    curr_ctx += [dict(role='user', content="Respond with your answer in the form f(x, y, z) = k. For example, f(x, y, z) = a*x*y + b*y*z. Do not respond with any other text. DO NOT GIVE YOUR REASONING."), dict(role='user', content="")]
    
    context_loss = 0.0
    token_enc = enc.encode(program)
    for token in token_enc:
        curr_ctx[-1]['content'] += enc.decode([token])
    
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106", # gpt-4 is faster for 1 token at a time than gpt-3.5, gpt-3.5 cheaper
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
            min_logprob = min(obj.logprob, min_logprob) # use smallest of 5 logprobs if not present (upper bound)
        
        context_loss += min_logprob

    return -context_loss # flip sign to make it a positive number trying to be 0


def llm_propose(programs, context, num_props=3, init=False):
    if init:
        programs = ["k"]

    programs_format = ""
    for i in range(len(programs)):
        programs_format += "f_priority_v"+str(i)+"(x, y, z) = "+programs[i]+", "

    start_ctx = [
        {
        "role": "system",
        "content": "Given the functions: "+programs_format+"your task is to propose "+str(num_props)+" unique approximations for the expression of f_priority_v"+str(len(programs))+" given the following assumptions: "+context+". Please abstract away any constants and represent them with capital letters. You cannot use I, E, O, S or Q as constants. Please respond with a numbered list where each entry begins with f_priority_v"+str(len(programs))+"(x, y, z). Be explicit with the functions. Do not give any reasoning or any other text."
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4" if init else "gpt-3.5-turbo-1106", # gpt-4 6x slower so only use for initial props
        messages=start_ctx,
        temperature=0.5,
        max_tokens=100*num_props,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    assert len(response.choices), 'No Response!'
    chat_response = response.choices[0].message
    chat_response_content = chat_response.content.strip().replace('≈','=')
    # print("GPT-4 Full Response: ", chat_response_content)

    out = list()
    for prop_eq in chat_response_content.split("\n"):
        prop_eq = prop_eq.split(" = ")[-1].strip().strip('.')
        out.append(prop_eq)

    return out
