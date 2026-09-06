# LaSR: Library Augmented Symbolic Regression
LibraryAugmentedSymbolicRegression.jl (LaSR.jl) accelerates the search for symbolic expressions using library learning.


<!-- prettier-ignore-start -->
<div align="center">

| Latest release | Website | Forums | Paper |
| :---: | :---: | :---: | :---: |
| [![version](https://juliahub.com/docs/General/LibraryAugmentedSymbolicRegression/stable/version.svg)](https://juliahub.com/ui/Packages/General/LibraryAugmentedSymbolicRegression) | [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://trishullab.github.io/lasr-web/) | [![Discussions](https://img.shields.io/badge/discussions-github-informational)](https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl/discussions) | [![Paper](https://img.shields.io/badge/arXiv-2409.09359-b31b1b)](https://arxiv.org/abs/2409.09359) |

| Build status | Coverage |
| :---: | :---: |
| [![CI](https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl/workflows/CI/badge.svg)](.github/workflows/CI.yml) | [![codecov](https://codecov.io/gh/trishullab/LibraryAugmentedSymbolicRegression.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/trishullab/LibraryAugmentedSymbolicRegression.jl) |

LaSR is a **plugin** for [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl) v2. Check out [PySR](https://github.com/MilesCranmer/PySR) for
a Python frontend.

[Cite this software](https://arxiv.org/abs/2409.09359)

</div>
<!-- prettier-ignore-end -->

**Contents**:

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Prompt templates](#prompt-templates)
- [MLJ](#mlj)
- [Extending the parser](#extending-the-parser)
- [Debugging LLM output](#debugging-llm-output)
- [Running with Ollama](#running-with-ollama)
- [Best practices](#best-practices)
- [Examples](#examples)
- [Benchmarking](#benchmarking)

## Installation

As SR.jl v2.0 is still in beta, we cannot yet release LaSR.jl to the registry. Install both from source instead:

```julia
using Pkg
Pkg.add(; url="https://github.com/MilesCranmer/SymbolicRegression.jl.git", rev="v2.0.0-beta.2")
Pkg.develop(; path="/path/to/LibraryAugmentedSymbolicRegression.jl")
```

A `[sources]` entry in a *dependency* is ignored by Pkg, so the SymbolicRegression rev has
to be pinned in your own environment. [`examples/Project.toml`](examples/Project.toml) is a
working environment that already does this:

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

## Quickstart

Simply pass the `LaSRPlugin` to SymbolicRegression's `Options` constructor. 

```julia
using SymbolicRegression
using LibraryAugmentedSymbolicRegression

const MODEL_NAME = "my-model"
const API_KEY = "token-abc123"
const API_URL = "http://localhost:11440/v1"

p = 0.001
model = Options(;
    plugins=(
        LaSRPlugin(;
            model=MODEL_NAME,
            api_key=API_KEY,
            api_kwargs=Dict(
                "url" => API_URL,
                "max_tokens" => 4096,
                "chat_template_kwargs" => Dict("enable_thinking" => false),
            ),
            verbose=true,
            use_concepts=true,
            use_concept_evolution=true,
            # The context provided to the LLM should describe the relationship between the input variables and the output variable. The more specific the context, the better the LLM can guide the search.
            context="We believe the relationship between the theta and offset parameter is a function of the cosine of the theta variable and the square of the offset.",
            variable_names=Dict("x1" => "theta", "x2" => "offset"),
            mutate_weight=p,
            randomize_weight=p,
            crossover_probability=p,
        ),
    ),
    binary_operators=[+, -, *, /, ^],
    unary_operators=[cos],
    populations=20,
)

X = randn(Float64, 2, 100)
y = @. 2 * cos(X[1, :]) + X[2, :]^2 - 2
hall_of_fame = equation_search(X, y; options, niterations=40)
```

Point `url`/`model` at any OpenAI-compatible server (vLLM, SGLang, Ollama, a paid API).

## LaSR configuration


### Backend model configuration

We use `PromptingTools.jl` to communicate with OpenAI-compatible servers. The following keywords configure the backend model and its connection.

| Keyword | Default | Description |
| --- | --- | --- |
| `model` | `nothing` | Model name on the OpenAI-compatible server. |
| `api_key` | `nothing` | API key for that server. Local servers usually accept any string. |
| `api_kwargs` | `Dict("max_tokens" => 4096)` | Forwarded to PromptingTools' OpenAI schema. `"url"` is required. |
| `http_kwargs` | `Dict("retries" => 3, "readtimeout" => 3600)` | Forwarded to the HTTP layer. |
| `verbose` | `true` | Prints the per-call token count and elapsed time. |

### LaSR main configuration

| Keyword | Default | Description |
| --- | --- | --- |
| `use_llm` | `true` | Use the LLM to generate equation proposals. |
| `use_concepts` | `false` | Incorporate concepts learned during the search into LLM prompts. |
| `use_concept_evolution` | `false` | Continuously evolve concepts throughout the search process. |
| `context` | `""` | A natural-language description of the problem, prepended to every prompt. Domain knowledge here is the single highest-leverage knob. |
| `variable_names` | `nothing` | Map from dataset names to meaningful names, e.g. `Dict("x1" => "theta")`. Falls back to the dataset's own names. |
| `parse_rules` | `NormalizationRule[]` | Custom string normalization rules, appended in order after the built-in defaults. See [Extending the parser](#extending-the-parser). |
| `lasr_logger` | `nothing` | `LaSRLogger(SRLogger(...))` records each LLM call — prompt, raw output, chosen expression, etc. |


### LaSR search hyperparameters

| Keyword | Default | Description |
| --- | --- | --- |
| `mutate_weight` | `0.0` | Probability of using LLM mutation. Mutation simply asks the LLM to suggest changes to the current equation. |
| `randomize_weight` | `0.0` | REWRITE Unnormalized weight of `LLMRandomizeMutation` (LLM-proposed replacement for a random restart). |
| `generate_weight` | `0.0` | REWRITE Unnormalized weight of `LLMGenerateMutation`: best-of-K structural generation — asks for `num_generated_equations` complete expressions, constant-fits each, keeps the best. |
| `crossover_probability` | `0.0` | REWRITE Probability in `[0, 1]` of using `LLMCrossover` **given** that SymbolicRegression already chose to cross over (set by SR's own `Options(; crossover_probability=...)`). LaSR pins subtree crossover to `1 - p` so this is a true conditional probability. |
| `num_pareto_context` | `5` | How many Pareto-frontier members — and how many concepts from the idea store — are shown to the LLM per call. |
| `num_generated_equations` | `5` | Expressions requested per call. Best-of-K for generation; the unused ones feed `suggestion_cache`. |
| `num_generated_concepts` | `5` | Concepts requested per concept-generation call. |
| `num_concept_crossover` | `2` | Concept pairs merged per concept-evolution round. |
| `prompts_dir` | `default_prompts_dir()` | Directory of `.prompt` templates. See [Prompt templates](#prompt-templates). A path that does not exist is rejected at construction, not minutes into a search. |
| `amnesty_complexity` | `0` | Any population member at or above this complexity has its constants re-optimized at the end of a generation. This operation 'rescues' any equation that has good structure but would have been deleted due to a bad constant fit. |


### LaSR library configuration

| Keyword | Default | Description |
| --- | --- | --- |
| `idea_database` | `String[]` | Concepts to seed the default store with. |
| `max_concepts` | `30` | Sampling window of the default store: retrieval draws from the `max_concepts` most recently refined concepts. |
| `idea_store` | `nothing` | Pass an `AbstractIdeaStore` to change how concepts are retrieved; overrides `idea_database`/`max_concepts`. |

LaSR implements two kinds of concept stores. New information retrieval techniques (RAG, etc.) can be added by implementing the `AbstractIdeaStore` interface.

- **`WindowedIdeaStore(; window=30, seed=String[])`** — A simple FIFO store. Newly refined concepts are added to the front. Concepts are retrieved from the first `window` entries. Concept evolution distills concepts from the rest of the entries. Ignores the concept's relevance to the query.

- **`ScoredIdeaStore(; k1=1.5, b=0.75, decay=0.99, refined_prior=2.0, seed=String[])`** — Returns the `k` most relevant concepts to the current query, using standard information retrieval techniques (here, BM25). 

### LaSR budgeting configuration

| Keyword | Default | Description |
| --- | --- | --- |
| `suggestion_cache` | `nothing` | By default, LaSR generates `num_generated_equations` proposals per call and only selects one of them. A `SuggestionCache(; capacity=8192)` saves these proposals and serves them for later requests. Entries are *consumed*, so population doesn't collapse. Inspect with `cache_stats`. |
| `max_llm_calls` | `nothing` | Hard ceiling on LLM calls for the whole run. After the ceiling is reached, the LLM operators fall back to their symbolic counterparts for the rest of the search. Inspect with `budget_used(plugin.call_budget)`. |


## Prompt templates

LaSR ships a set of default prompt templates inside the package ( `default_prompts_dir()`). To edit them, copy the directory to a writable location and pass that path to `LaSRPlugin`:

```julia
dir = copy_prompts("~/my_lasr_prompts")   # writable copies of every .prompt
plugin = LaSRPlugin(; prompts_dir=dir)
```
On a new installation, you can locate the `default_prompts_dir()` with the following python script (using `juliacall`):

```python
from juliacall import Main as jl
jl.seval("using SymbolicRegression, LibraryAugmentedSymbolicRegression")
LaSR = jl.LibraryAugmentedSymbolicRegression

print(LaSR.default_prompts_dir())                          # read the shipped defaults
prompts_dir = str(LaSR.copy_prompts("~/my_lasr_prompts"))  # edit these, then pass along
```


## Extending the parser

LaSR normalizes LLM outputs using a series of `NormalizationRule`s before passing them to SymbolicRegression's parser to handle domain-specific notation (e.g. `pow(a, b)` or `a^b` or `a**b` all map to the same julia operator). You can register your own domain-specific `NormalizationRule`s. These rules are parsed after the built-in defaults:

```julia
using LibraryAugmentedSymbolicRegression: LaSRPlugin, NormalizationRule, parse_expr

# "any word followed by `!` means safe_factorial(word)"
factorial_rule = NormalizationRule(r"(\w+)!" => s"safe_factorial(\1)")

options = Options(;
    binary_operators=[+, -, *, /],
    unary_operators=[safe_factorial, sin, cos],
    plugins=(LaSRPlugin(; use_llm=false, parse_rules=[factorial_rule]),),
)

parse_expr(Float64, "x0!", options)   # -> safe_factorial(x0)
```

Without the rule, `x0!` is not valid Julia and the whole proposal falls back to a
constant. [`examples/example_2_operator_extension.jl`](examples/example_2_operator_extension.jl)
walks through the full two-step process, including making the operator differentiable.

### Debugging LLM output

LaSR provides two strategies to find parse errors. (1) Use the `lasr_logger` to record every LLM call, including the raw output and the chosen expression. However, it's not always convenient to inspect the logger after a long run. (2) Using the `ParseFailureStore`. LaSR records every unparseable output in this store. You can inspect it after the search to see what went wrong.

```julia
using LibraryAugmentedSymbolicRegression: LaSRPlugin, parse_failures, parse_failure_summary
using LibraryAugmentedSymbolicRegression.ParseFailuresModule: ParseFailureStore

sink = ParseFailureStore()
plugin = LaSRPlugin(; parse_failure_sink=sink, mutate_weight=0.01, model=..., api_key=...)
# ... run the search ...

parse_failure_summary(sink)   # the most frequent offending strings
parse_failures(sink)          # raw, normalized, stage, reason
```

The store aggregates across `:serial` and `:multithreading` workers; `:multiprocessing`
workers live in separate address spaces, so use `lasr_logger` for those.

## Running with Ollama

LaSR works with any OpenAI-compatible server. Ollama is a free one geared towards commodity laptops; download it [here](https://ollama.com/download), then:

```bash
$ ollama pull llama3.1
# This requires about ~4GB of disk space.
# Ollama can be _very_ slow.
# Ollama runs on port 11434 by default. Test the server with...
$ curl http://localhost:11434/v1/models
{"object":"list","data":[{"id":"llama3.1:latest","object":"model","created":1730973855,"owned_by":"library"}]}

$ curl http://localhost:11434/v1/completions -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-abc123" -d '{
    "model": "llama3.1:latest",
    "prompt": "Once upon a time,",
    "max_tokens": 50,
    "temperature": 0.7
  }'
{"id":"cmpl-626","object":"text_completion","created":1730977391,"model":"llama3.1:latest","choices":[{"text":"...in a far-off kingdom, hidden behind a veil of sparkling mist and whispering leaves, there existed a magical realm unlike any other.","index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":29,"total_tokens":44}}
```

Then, replace the `model`, `api_key`, and `url` in the `LaSRPlugin` constructor with your Ollama settings. The default prompt templates are compatible with Ollama, but you can also customize them as described in [Prompt templates](#prompt-templates).

Specifically, in the [Quickstart](#quickstart) example, you would set:

```julia
const MODEL_NAME = "llama3.1:latest"
const API_KEY = "any-string"  # Ollama accepts any string as an API key
const API_URL = "http://localhost:11434/v1"
```


## Best practices

1. Always make sure you cannot find a satisfactory solution with `use_llm=false` before
   reaching for LLM guidance.
1. Start with an OpenAI-compatible server on your own machine before moving to paid
   services. There are many guides for setting one up:
   [1](https://ollama.com/blog/openai-compatibility)
   [2](https://docs.vllm.ai/en/latest/getting_started/installation.html)
   [3](https://github.com/sgl-project/sglang?tab=readme-ov-file#backend-sglang-runtime-srt)
   [4](https://old.reddit.com/r/LocalLLaMA/comments/16y95hk/a_starter_guide_for_playing_with_your_own_local_ai/).
1. Budget before you run. The operator weights are per-*mutation*, so the resulting call
   count also depends on `populations`, `population_size` and `ncycles_per_iteration` —
   raising any of those silently buys more LLM traffic. Measured at the defaults with
   `p=0.01`, a search issues roughly **25 calls per iteration**, about **1000 over a
   40-iteration run**. Set `max_llm_calls` to make that a number you choose rather than one
   you discover, and turn on `suggestion_cache` to reuse proposals you already paid for.
1. Spend your effort on `context` and `variable_names` before tuning weights. Domain
   knowledge in the prompt beats more calls.

## Examples

Runnable, tested end to end:

- [`examples/example_1_regressor.jl`](examples/example_1_regressor.jl) — `SRRegressor` +
  `LaSRPlugin`, reading LLM connection settings from a `.env` (copy
  [`.env.example`](.env.example)).
- [`examples/example_2_operator_extension.jl`](examples/example_2_operator_extension.jl) —
  adding a domain operator and the parse rule that goes with it.

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/example_1_regressor.jl
```

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
