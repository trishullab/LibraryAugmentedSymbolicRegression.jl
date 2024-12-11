# LaSR: Library augmented Symbolic Regression

<center style="font-weight: bold;">
    <a href="https://trishullab.github.io/lasr-web/">Website</a> | 
    <a href="https://arxiv.org/abs/2409.09359/">Paper</a> | 
    <a href="https://trishullab.github.io/lasr-web/static/lasr-slides.pdf"> Short Slide Deck</a>
</center>

**Purpose**: This branch contains code to run LaSR on the synthetic dataset, the Feynman Equations dataset, and the bigbench dataset. We hope this is useful for future work which wishes to compare against LaSR.

## Installation

```bash
# Install Julia
$ curl -fsSL https://install.julialang.org | sh

# Make a new python environment (I'm using Anaconda; feel free to use your favorite package manager).
$ conda create --name lasr python==3.10
$ conda activate lasr
$ vim pysr/juliapkg.jl
# {
#     "julia": "~1.6.7, ~1.7, ~1.8, ~1.9, =1.10.0, ^1.10.3",
#     "packages": {
#         "SymbolicRegression": {
#             "uuid": "8254be44-1295-4e6a-a16d-46603ac705cb",
#             "dev": true,
#             "path": "<PREPEND WITH ABSOLUTE PATH TO PROJECT DIRECTORY>/SymbolicRegression.jl"
#         },
#         "Serialization": {
#             "uuid": "9e88b42a-f829-5b0c-bbe9-9e923198166b",
#             "version": "1"
#         }
#     }
# }
(lasr) $ pip install .
(lasr) $ cat vllm_api.key
token-abc123
# ^ There is no newline at the end of the file.
(lasr) $ python -m experiments.main --use_llm --use_prompt_evol --model "meta-llama/Meta-Llama-3-8B-Instruct" --api_key "vllm_api.key" --model_url "http://localhost:11440/v1" --exp_idx 0 --dataset_path data/FeynmanEquations.csv  --dataset "Feynman" --start_idx 0 
# For more commands, read `run.sh`

# To run bigbench experiments
(lasr) $ unzip bigbench/csvs.zip # Use the pre-generated datasets.
(lasr) $ open bigbench/generate_datasets.ipynb # Follow instructions to generate bigbench datasets. Might require $ pip install bigbench
(lasr) $ julia search.jl # Edit the file to change the dataset path and other parameters
(lasr) $ open bigbench/evaluate_equations.ipynb # Follow instructions to evaluate the equations found by LaSR


# VLLM Setup: https://docs.vllm.ai/en/latest/
$ conda activate vllm
(vllm) $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --api-key token-abc123 --port 11440 --tensor-parallel-size 8 --disable-log-requests --block-size 32 --swap-space 4

#  The Julia code doesn't validate the backend API. I used this to validate the backend API:
curl {your_model_url}/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer {your_api_key}" \
-d '{
"model": "{your_model_name}",
"messages": [
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "role": "user",
    "content": "Hello!"
  }
]
}' 
```
