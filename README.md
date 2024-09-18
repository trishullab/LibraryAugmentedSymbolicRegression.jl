<!-- prettier-ignore-start -->
<div align="center">

LibraryAugmentedSymbolicRegression.jl (LaSR.jl) accelerates the search for symbolic expressions using library learning.

| Latest release | Website | Forums | Paper |
| :---: | :---: | :---: | :---: |
| [![version](https://juliahub.com/docs/LibraryAugmentedSymbolicRegression/version.svg)](https://juliahub.com/ui/Packages/LibraryAugmentedSymbolicRegression/X2eIS) | [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://trishullab.github.io/lasr-web/) | [![Discussions](https://img.shields.io/badge/discussions-github-informational)](https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl/discussions) | [![Paper](https://img.shields.io/badge/arXiv-2409.09359-b31b1b)](https://arxiv.org/abs/2409.09359) |

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

If you'd like to compare with LaSR, we've archived the code used in the paper in the `lasr-experiments` branch. Clone this repository and run:
```bash
$ git switch lasr-experiments
```
to switch to the branch and follow the instructions in the README to reproduce our results. This directory contains the code for evaluating LaSR on the 

- [x] Feynman Equations dataset
- [x] Synthetic equations dataset
    - [x] and generation code
- [x] Bigbench experiments
    - [x] and evaluation code


## Quickstart

Install in Julia with:

```julia
using Pkg
Pkg.add("LibraryAugmentedSymbolicRegression")
```

LaSR uses the same interface as [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl). The easiest way to use LibraryAugmentedSymbolicRegression.jl
is with [MLJ](https://github.com/alan-turing-institute/MLJ.jl).
Let's see an example:

```julia
import LibraryAugmentedSymbolicRegression: LaSRRegressor, LLMOptions
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
      ...
    )
)
```

Now, let's create and train this model
on our data:

```julia
mach = machine(model, X, y)

fit!(mach)
```

You will notice that expressions are printed
using the column names of our table. If,
instead of a table-like object,
a simple array is passed
(e.g., `X=randn(100, 2)`),
`x1, ..., xn` will be used for variable names.

Let's look at the expressions discovered:

```julia
report(mach)
```

Finally, we can make predictions with the expressions
on new data:

```julia
predict(mach, X)
```

This will make predictions using the expression
selected by `model.selection_method`,
which by default is a mix of accuracy and complexity.

You can override this selection and select an equation from
the Pareto front manually with:

```julia
predict(mach, (data=X, idx=2))
```

where here we choose to evaluate the second equation.

For fitting multiple outputs, one can use `MultitargetLaSRRegressor`
(and pass an array of indices to `idx` in `predict` for selecting specific equations).
For a full list of options available to each regressor, see the [API page](https://astroautomata.com/LibraryAugmentedSymbolicRegression.jl/dev/api/).


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
    llm_recorder_dir="lasr_runs/debug_0",                                       # Directory to log LLM interactions.
    llm_context="",                                                             # Natural language concept to start with. You should also be able to initialize with a list of concepts.
    var_order=nothing,                                                          # Dict(variable_name => new_name).
    idea_threshold=30                                                           # Number of concepts to keep track of.
)
```

### Best Practices

1. Always make sure you cannot find a satisfactory solution with `active=false`. This will save you time and money.
1. Start with a LLM OpenAI compatible server running on your local machine before moving onto paid services. There are many online resources to set up a local LLM server [1](https://ollama.com/blog/openai-compatibility) [2](https://docs.vllm.ai/en/latest/getting_started/installation.html) [3](https://github.com/sgl-project/sglang?tab=readme-ov-file#backend-sglang-runtime-srt) [4](https://old.reddit.com/r/LocalLLaMA/comments/16y95hk/a_starter_guide_for_playing_with_your_own_local_ai/)
1. If you are using LLM, do a back-of-the-envelope calculation to estimate the cost of running LLM for your problem.  Each iteration will make around 60k calls to the LLM model. Each call to the LLM (with the default prompt) is around 1k tokens. This gives us an upper bound of 60M tokens per iteration if `p=1.00`. Running the model at `p=0.01` for 40 iterations will result in just under 25M tokens for each equation.


## Organization

LibraryAugmentedSymbolicRegression.jl development is kept independent from the main codebase. However, to ensure LaSR can be used easily, it is integrated into SymbolicRegression.jl via the [`ext/SymbolicRegressionLaSRExt`](https://www.example.com) extension module. This, in turn, is loaded into PySR. This cartoon summarizes the interaction between the different packages:

![LibraryAugmentedSymbolicRegression.jl organization](https://raw.githubusercontent.com/trishullab/lasr-web/main/static/lasr-code-interactions.svg)

> [!NOTE]  
> The `ext/SymbolicRegressionLaSRExt` module is not yet available in the released version of SymbolicRegression.jl. It will be available in the release `vX.X.X` of SymbolicRegression.jl.
