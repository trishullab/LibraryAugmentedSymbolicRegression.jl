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
import LibraryAugmentedSymbolicRegression:
    LaSROptions, LaSRRegressor, LaSRMutationWeights, LLMOperationWeights
import MLJ: machine, fit!, predict, report

# Dataset with two named features:
X = (a=rand(500), b=rand(500))

# and one target:
y = @. 2 * cos(X.a * 23.5) - X.b^2

# with some noise:
y = y .+ randn(500) .* 1e-3

model = LaSRRegressor(;
    niterations=50,
    binary_operators=[+, -, *],
    unary_operators=[cos],
    use_llm=true,
    use_concepts=true,
    use_concept_evolution=true,
    lasr_mutation_weights=LaSRMutationWeights(; llm_mutate=0.1, llm_randomize=0.1),
    llm_operation_weights=LLMOperationWeights(; llm_crossover=0.1),
    llm_context="We believe the function to be a trigonometric function of the angle and a quadratic function of the bias.",
    llm_recorder_dir="lasr_runs/",
    variable_names=Dict("a" => "angle", "b" => "bias"),
    prompts_dir="prompts/",
    api_key="token-abc123",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    api_kwargs=Dict("url" => "http://localhost:11440/v1"),
    verbose=true,
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
llm_options = LaSRRegressor(
    ...
    # SR.jl options
    use_llm=true,
    # Whether to use LLM inference or not. (default: false)

    use_concepts=true,
    # Whether to use natural language concepts in the search. (default: false)
    # This makes the algorithm equivanlent to a specialization of FunSearch.

    use_concept_evolution=true,
    # Whether to evolve the concepts after every iteration. (default: false)

    lasr_mutation_weights=LaSRMutationWeights(; llm_mutate=0.1, llm_randomize=0.1),
    # Unnormalized mutation weights for the mutation operators.

    llm_operation_weights=LLMOperationWeights(; llm_crossover=0.1),
    # Normalized probability of using the LLM for crossover v/s using symbolic crossover.

    num_pareto_context=5
    # Number of equations to sample from the Pareto frontier for summarization.
    num_generated_equations=5,
    # Number of equations to generate from the LLM.
    num_generated_concepts=5,
    # Number of concepts to generate from the LLM.
    max_concepts=30,
    # Size of the concept library. Only active if use_concepts=true.
    is_parametric::Bool=false,
    # @TODO: Need to change the interface to use node_type.
    # This boolean variable is a special flag to allow sampling parametric equations from LaSR.
    llm_context="We believe the function to be a trigonometric function of the angle and a quadratic function of the bias.",
    # A natural language concept to start with.


    llm_recorder_dir="lasr_runs/",
    # Directory to log LLM interactions. Creates a file called llm_calls.txt in this directory.
    variable_names=Dict("a" => "angle", "b" => "bias"),
    # The variable name that is passed to the LLM. This is useful to introduce domain knowledge to the LLM.
    prompts_dir="prompts/",
    # The location of the zero-shot prompts for the LLM. Specialize these prompts to your problem for better performance.
    idea_database=[],
    # A list of concepts that we will use to seed the LLM. Starts with an empty
    # list. Concepts are chosen uniformly at random from this list.

    api_key="token-abc123",
    # API key to OpenAI API compatible server.
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    # Model name for the OpenAI API compatible server.
    api_kwargs=Dict("url" => "http://localhost:11440/v1"),
    # Keyword arguments passed to server. URL is the only required argument.
    http_kwargs=Dict("retries" => 3, "readtimeout" => 3600),
    # Keyword arguments passed to HTTP requests.
    verbose=true,
    # Whether to print the tokens generated for each LLM call.
    # (Useful for debugging to get a general sense of the performance of the LLM server.)
)
```

### Best Practices

1. Always make sure you cannot find a satisfactory solution with `use_llm=false` before using LLM guidance.
1. Start with a LLM OpenAI compatible server running on your local machine before moving onto paid services. There are many online resources to set up a local LLM server [1](https://ollama.com/blog/openai-compatibility) [2](https://docs.vllm.ai/en/latest/getting_started/installation.html) [3](https://github.com/sgl-project/sglang?tab=readme-ov-file#backend-sglang-runtime-srt) [4](https://old.reddit.com/r/LocalLLaMA/comments/16y95hk/a_starter_guide_for_playing_with_your_own_local_ai/)
1. If you are using LLM, do a back-of-the-envelope calculation to estimate the cost of running LLM for your problem.  Each iteration will make around 60k calls to the LLM model. With the default prompts (in `prompts/`), each call usually requires generating 250 to 1000 tokens. This gives us an upper bound of 60M tokens per iteration if `p=1.00`. Hence, running the model at `p=0.01` for 40 iterations will result in 24M tokens for each equation.


## Organization

LibraryAugmentedSymbolicRegression.jl development is kept independent from the main codebase. However, to ensure LaSR can be used easily, it is integrated into SymbolicRegression.jl via the [`ext/SymbolicRegressionLaSRExt`](https://www.example.com) extension module. This, in turn, is loaded into PySR. This cartoon summarizes the interaction between the different packages:

![LibraryAugmentedSymbolicRegression.jl organization](https://raw.githubusercontent.com/trishullab/lasr-web/main/static/lasr-code-interactions.svg)

> [!NOTE]  
> The `ext/SymbolicRegressionLaSRExt` module is not yet available in the released version of SymbolicRegression.jl. It will be available in the release `vX.X.X` of SymbolicRegression.jl.



## Running with Ollama

LaSR can be paired with any LLM server that is compatible with OpenAI's API. Ollama is a free and open-source LLM server geared towards running LLMs on commodity laptops. You can download and setup Ollama from [here](https://ollama.com/download). After this, run:

```bash
$ ollama help
Large language model runner

Usage:
  ollama [flags]
  ollama [command]

Available Commands:
  serve       Start ollama
  create      Create a model from a Modelfile
  show        Show information for a model
  run         Run a model
  stop        Stop a running model
  pull        Pull a model from a registry
  push        Push a model to a registry
  list        List models
  ps          List running models
  cp          Copy a model
  rm          Remove a model
  help        Help about any command

Flags:
  -h, --help      help for ollama
  -v, --version   Show version information

Use "ollama [command] --help" for more information about a command.
$ ollama pull llama3.1
# This downloads a 4GB-ish file that contains the Llama3.1 8B model.
# Ollama, by default, runs on port 11434 of your local machine. Let's try a debug query to make sure we can connect to Ollama.
$ curl http://localhost:11434/v1/models
{"object":"list","data":[{"id":"llama3.1:latest","object":"model","created":1730973855,"owned_by":"library"},{"id":"mistral:latest","object":"model","created":1697556753,"owned_by":"library"},{"id":"wizard-math:latest","object":"model","created":1697556753,"owned_by":"library"},{"id":"codellama:latest","object":"model","created":1693414395,"owned_by":"library"},{"id":"nous-hermes-llama2:latest","object":"model","created":1691000950,"owned_by":"library"}]}

$ curl http://localhost:11434/v1/completions -H "Content-Type: application/json"   -H "Authorization: Bearer token-abc123"   -d '{
    "model": "llama3.1:latest",
    "prompt": "Once upon a time,",
    "max_tokens": 50,
    "temperature": 0.7
  }'

{"id":"cmpl-626","object":"text_completion","created":1730977391,"model":"llama3.1:latest","system_fingerprint":"fp_ollama","choices":[{"text":"...in a far-off kingdom, hidden behind a veil of sparkling mist and whispering leaves, there existed a magical realm unlike any other.","index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":29,"total_tokens":44}}
```

Now, we can run the simple example in Julia with model_name as `llama3.1:latest` and the HTTP URL as `http://localhost:11434/v1`:

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
        use_llm=true,
        lasr_weights=LLMWeights(llm_mutate=0.1, llm_crossover=0.1, llm_gen_random=0.1),
        use_concept_evolution=true,
        use_concepts=true,
        api_key="token-abc123",
        prompts_dir="prompts/",
        llm_recorder_dir="lasr_runs/debug_0/",
        model="llama3.1:latest",
        api_kwargs=Dict("url" => "http://127.0.0.1:11434/v1"),
        variable_names=Dict("a" => "angle", "b" => "bias"),
        llm_context="We believe the function to be a trigonometric function of the angle and a quadratic function of the bias."
    )
)

mach = machine(model, X, y)
fit!(mach)
# julia> fit!(mach)
# [ Info: Training machine(LaSRRegressor(binary_operators = Function[+, -, *], …), …).
# ┌ Warning: You are using multithreading mode, but only one thread is available. Try starting julia with `--threads=auto`.
# └ @ LibraryAugmentedSymbolicRegression ~/Desktop/projects/004_scientific_discovery/LibraryAugmentedSymbolicRegression.jl/src/Configure.jl:55
# [ Info: Tokens: 476 in 22.4 seconds
# [ Info: Started!
# [ Info: Tokens: 542 in 49.2 seconds
# [ Info: Tokens: 556 in 51.1 seconds
# [ Info: Tokens: 573 in 53.2 seconds
report(mach)
predict(mach, X)
```
