# LaSR: Library augmented Symbolic Regression
LibraryAugmentedSymbolicRegression.jl (LaSR.jl) accelerates the search for symbolic expressions using library learning. 


<!-- prettier-ignore-start -->
<div align="center">

| Latest release | Website | Forums | Paper |
| :---: | :---: | :---: | :---: |
| [![version](https://juliahub.com/docs/General/LibraryAugmentedSymbolicRegression/stable/version.svg)](https://juliahub.com/ui/Packages/General/LibraryAugmentedSymbolicRegression) | [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://trishullab.github.io/lasr-web/) | [![Discussions](https://img.shields.io/badge/discussions-github-informational)](https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl/discussions) | [![Paper](https://img.shields.io/badge/arXiv-2409.09359-b31b1b)](https://arxiv.org/abs/2409.09359) |

| Build status | Coverage |
| :---: | :---: |
| [![CI](https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl/workflows/CI/badge.svg)](.github/workflows/CI.yml) | [![Coverage Status](https://coveralls.io/repos/github/trishullab/LibraryAugmentedSymbolicRegression.jl/badge.svg?branch=master)](https://coveralls.io/github/trishullab/LibraryAugmentedSymbolicRegression.jl?branch=master) |

LaSR is integrated with [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl). Check out [PySR](https://github.com/MilesCranmer/PySR) for
a Python frontend.

[Cite this software](https://arxiv.org/abs/2409.09359)

</div>
<!-- prettier-ignore-end -->

**Contents**:

- [Benchmarking](#benchmarking)
- [Quickstart](#quickstart)
- [Search options](#search-options)
- [Organization](#organization)

## Benchmarking

If you'd like to compare with LaSR, we've archived the code used in the paper in the [`lasr-experiments`](https://github.com/trishullab/LaSR.jl/tree/lasr-experiments) branch. Clone this repository and run:
```bash
$ git switch lasr-experiments
```
to switch to the branch and follow the instructions in the README to reproduce our results. This directory contains the data and code for running and evaluating LaSR on the following datasets: 

- [x] Feynman Equations dataset
- [x] Synthetic equations dataset
    - [x] and generation code
- [x] Bigbench experiments
    - [x] and evaluation code
     
> [!NOTE]
> The code in the `lasr-experiments` branch directly modifies a 'frozen' version of SymbolicRegression.jl and PySR. While we gradually work on integrating LaSR into the main PySR repository, you can still use LaSR within Python by installing the pip package in this branch.


## Quickstart

Install in Julia with:

```julia
using Pkg
Pkg.add("LibraryAugmentedSymbolicRegression")
```

LaSR uses the same interface as [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl), and is integrated into SymbolicRegression.jl through the [`SymbolicRegressionLaSRExt`](integration). However, LaSR can be directly used with [MLJ](https://github.com/alan-turing-institute/MLJ.jl) as well. The only difference is that you need to pass an `LLMOptions` object to the `LaSRRegressor` constructor.


For example, we can modify the `example.jl` from the SymbolicRegression.jl documentation to use LaSR as follows:

> [!NOTE]
> LaSR searches for the LLM query prompts in a  a directory called `prompts/` at the location you start Julia. You can download and extract the `prompts.zip` folder from [here](https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl/raw/refs/heads/master/prompts.zip) to the desired location. If you wish to use a different location, you can pass a different `prompts_dir` argument to the `LLMOptions` object.

```julia
import LibraryAugmentedSymbolicRegression: LaSRRegressor, LLMOptions, LLMWeights
import MLJ: machine, fit!, predict, report

# Dataset with two named features:
X = (a = rand(500), b = rand(500))

# and one target:
y = @. 2 * cos(X.a * 23.5) - X.b ^ 2

# with some noise:
y = y .+ randn(500) .* 1e-3

model = LaSRRegressor(
    niterations=50,
    binary_operators=[+, -, *],
    unary_operators=[cos],
    llm_options=LLMOptions(
        active=true,
        weights=LLMWeights(llm_mutate=0.1, llm_crossover=0.1, llm_gen_random=0.1),
        prompt_evol=true,
        prompt_concepts=true,
        api_key="token-abc123",
        prompts_dir="prompts",
        llm_recorder_dir="lasr_runs/debug_0",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        api_kwargs=Dict("url" => "http://localhost:11440/v1"),
        var_order=Dict("a" => "angle", "b" => "bias")
        llm_context="We believe the function to be a trigonometric function of the angle and a quadratic function of the bias.",
    )
)
mach = machine(model, X, y)

# ensure ./prompts/ exists. If not, download and extract the prompts.zip file from the repository.
fit!(mach)
# open ./lasr_runs/debug_0/llm_calls.txt to see the LLM interactions.
report(mach)
predict(mach, X)
```

## Search options

Other than `LLMOptions`, We have the same search options as SymbolicRegression.jl. See https://astroautomata.com/SymbolicRegression.jl/stable/api/#Options

### LLM Options

LaSR uses PromptingTools.jl for zero shot prompting. If you wish to make changes to the prompting options, you can pass an `LLMOptions` object to the `LaSRRegressor` constructor. The options available are:
```julia
llm_options = LLMOptions(
    active=true,                                                                # Whether to use LLM inference or not
    weights=LLMWeights(llm_mutate=0.1, llm_crossover=0.1, llm_gen_random=0.1),  # Probability of using LLM for mutation, crossover, and random generation
    num_pareto_context=5,                                                       # Number of equations to sample from the Pareto frontier for summarization.
    prompt_evol=true,                                                           # Whether to evolve natural language concepts through LLM calls.
    prompt_concepts=true,                                                       # Whether to use natural language concepts in the search.
    api_key="token-abc123",                                                     # API key to OpenAI API compatible server.
    model="meta-llama/Meta-Llama-3-8B-Instruct",                                # LLM model to use.
    api_kwargs=Dict("url" => "http://localhost:11440/v1"),                      # Keyword arguments passed to server.
    http_kwargs=Dict("retries" => 3, "readtimeout" => 3600),                    # Keyword arguments passed to HTTP requests.
    prompts_dir="prompts",                                                      # Directory to look for zero shot prompts to the LLM.
    llm_recorder_dir="lasr_runs/debug_0",                                       # Directory to log LLM interactions.
    llm_context="",                                                             # Natural language concept to start with. You should also be able to initialize with a list of concepts.
    var_order=nothing,                                                          # Dict(variable_name => new_name).
    idea_threshold=30                                                           # Number of concepts to keep track of.
    is_parametric=false,                                                        # This is a special flag to allow sampling parametric equations from LaSR. This won't be needed for most users.
)
```

### Best Practices

1. Always make sure you cannot find a satisfactory solution with `active=false` before using LLM guidance.
1. Start with a LLM OpenAI compatible server running on your local machine before moving onto paid services. There are many online resources to set up a local LLM server [1](https://ollama.com/blog/openai-compatibility) [2](https://docs.vllm.ai/en/latest/getting_started/installation.html) [3](https://github.com/sgl-project/sglang?tab=readme-ov-file#backend-sglang-runtime-srt) [4](https://old.reddit.com/r/LocalLLaMA/comments/16y95hk/a_starter_guide_for_playing_with_your_own_local_ai/)
1. If you are using LLM, do a back-of-the-envelope calculation to estimate the cost of running LLM for your problem.  Each iteration will make around 60k calls to the LLM model. With the default prompts (in `prompts/`), each call usually requires generating 250 to 1000 tokens. This gives us an upper bound of 60M tokens per iteration if `p=1.00`. Hence, running the model at `p=0.01` for 40 iterations will result in 24M tokens for each equation.


## Organization

LibraryAugmentedSymbolicRegression.jl development is kept independent from the main codebase. However, to ensure LaSR can be used easily, it is integrated into SymbolicRegression.jl via the [`ext/SymbolicRegressionLaSRExt`](https://www.example.com) extension module. This, in turn, is loaded into PySR. This cartoon summarizes the interaction between the different packages:

![LibraryAugmentedSymbolicRegression.jl organization](https://raw.githubusercontent.com/trishullab/lasr-web/main/static/lasr-code-interactions.svg)

> [!NOTE]  
> The `ext/SymbolicRegressionLaSRExt` module is not yet available in the released version of SymbolicRegression.jl. It will be available in the release `vX.X.X` of SymbolicRegression.jl.
