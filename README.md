# LaSR: Language aided Symbolic Regression

<center style="font-weight: bold;">
    <a href="https://trishullab.github.io/lasr-web/">Website</a> | 
    <a href="https://atharvas.net/static/lasr.pdf/">Paper</a> | 
    <a href="https://trishullab.github.io/lasr-web/static/lasr-slides.pdf"> Short Slide Deck</a>
</center>

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
#         "PromptingTools": {
#             "uuid": "670122d1-24a8-4d70-bfce-740807c42192",
#             "dev": true,
#             "path": "<PREPEND WITH ABSOLUTE PATH TO PROJECT DIRECTORY>/PromptingTools.jl"
#         },
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
# Run run.sh to run all experiments.
(lasr) $ cat vllm_api.key
token-abc123
# ^ There is no newline at the end of the file.
(lasr) $ python -m experiments.main --use_llm --use_prompt_evol --model "meta-llama/Meta-Llama-3-8B-Instruct" --api_key "vllm_api.key" --model_url "http://localhost:11440/v1" --exp_idx 0 --dataset_path data/FeynmanEquations.csv  --dataset "Feynman" --start_idx 0 
# For more commands, checkout `run.sh`

# To run bigbench experiments
(lasr) $ unzip bigbench.zip
(lasr) $ julia search.jl


# VLLM Setup: https://docs.vllm.ai/en/latest/
$ conda activate vllm
(vllm) $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --api-key token-abc123 --port 11440 --tensor-parallel-size 8 --disable-log-requests --block-size 32 --swap-space 4
```