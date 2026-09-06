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

> [!IMPORTANT]
> This README documents the **unreleased** plugin rewrite. The version in the General
> registry (and the JuliaHub badge above) is the older fork-based API — `LaSRRegressor`,
> `LaSROptions`, `LLMOptions` — none of which exists here. See
> [Installation](#installation) and [CHANGELOG.md](CHANGELOG.md).

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

LaSR pins SymbolicRegression.jl to `v2.0.0-beta.2`; a registry release is blocked until
SymbolicRegression 2.0 is final. `Pkg.add("LibraryAugmentedSymbolicRegression")` gets you
the *old* fork-based API, not this one. Install both from source instead:

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

LaSR does **not** re-export SymbolicRegression. Import both:

```julia
using SymbolicRegression, LibraryAugmentedSymbolicRegression
```

`LaSRPlugin` comes from LaSR; `Options`, `SRRegressor`, `equation_search`,
`calculate_pareto_frontier`, `string_tree` and the rest come from SymbolicRegression.

## Quickstart

Configure the LLM boundary in a `LaSRPlugin` and pass the plugin to the ordinary
SymbolicRegression `Options` constructor:

```julia
using SymbolicRegression
using LibraryAugmentedSymbolicRegression

p = 0.001
plugin = LaSRPlugin(;
    model="my-model",
    api_key="token-abc123",
    api_kwargs=Dict(
        "url" => "http://localhost:11440/v1",
        # `api_kwargs` REPLACES the default, so restate `max_tokens` whenever you set it.
        "max_tokens" => 4096,
    ),
    use_concepts=true,
    use_concept_evolution=true,
    context="We believe the relationship between the theta and offset parameter is a function of the cosine of the theta variable and the square of the offset.",
    variable_names=Dict("x1" => "theta", "x2" => "offset"),
    mutate_weight=p,
    randomize_weight=p,
    crossover_probability=p,
)

options = Options(;
    binary_operators=[+, -, *, /, ^],
    unary_operators=[cos],
    populations=20,
    plugins=(plugin,),
)

X = randn(Float64, 2, 100)
y = @. 2 * cos(X[1, :]) + X[2, :]^2 - 2
hall_of_fame = equation_search(X, y; options, niterations=40)
```

Point `url`/`model` at any OpenAI-compatible server (vLLM, SGLang, Ollama, a paid API).

> [!NOTE]
> Reasoning models (gemma-4 / Qwen, served with a `--reasoning-parser`) return
> an **empty** `content` field unless thinking is disabled, and every LLM suggestion is
> silently dropped. Pass
> `"chat_template_kwargs" => Dict("enable_thinking" => false)` inside `api_kwargs`.

## Configuration

`LaSRPlugin` is the sole configuration surface. Structural mutation weights
(`mutate_constant`, `add_node`, …) still belong to SymbolicRegression's own
`Options(; mutation_weights=...)`; everything LLM-specific lives on the plugin.

### LLM connection

| Keyword | Default | Description |
| --- | --- | --- |
| `model` | `nothing` | Model name on the OpenAI-compatible server. |
| `api_key` | `nothing` | API key for that server. Local servers usually accept any string. |
| `api_kwargs` | `Dict("max_tokens" => 4096)` | Forwarded to PromptingTools' OpenAI schema. `"url"` is required. **Replaces** the default wholesale — if you set it, restate `max_tokens`, or responses truncate before their closing fence and are discarded. |
| `http_kwargs` | `Dict("retries" => 3, "readtimeout" => 3600)` | Forwarded to the HTTP layer. |
| `llm_generate` | `PromptingTools.aigenerate` | The generation function. Inject a deterministic local function to test the whole LaSR path without a server. |
| `verbose` | `true` | Forwarded to `llm_generate`, so PromptingTools prints the per-call token count and elapsed time; also enables `@debug` messages when model output cannot be read. |

### Feature switches

| Keyword | Default | Description |
| --- | --- | --- |
| `use_llm` | `true` | Master switch for every LLM operator. |
| `use_concepts` | `false` | Feed the evolving natural-language concept library into prompts. With this off, the algorithm is a specialized FunSearch. |
| `use_concept_evolution` | `false` | Distill and merge concepts at the end of each generation. |

### How often the LLM runs

All weights default to `0.0`. With `use_llm=true` and every weight zero, the plugin warns
at construction rather than silently attaching a plugin that never fires.

| Keyword | Default | Description |
| --- | --- | --- |
| `mutate_weight` | `0.0` | Unnormalized weight of `LLMMutateMutation` in SymbolicRegression's mutation table, exactly like the other entries in `Options.mutations`. |
| `randomize_weight` | `0.0` | Unnormalized weight of `LLMRandomizeMutation` (LLM-proposed replacement for a random restart). |
| `generate_weight` | `0.0` | Unnormalized weight of `LLMGenerateMutation`: best-of-K structural generation — asks for `num_generated_equations` complete expressions, constant-fits each, keeps the best. |
| `crossover_probability` | `0.0` | Probability in `[0, 1]` of using `LLMCrossover` **given** that SymbolicRegression already chose to cross over (set by SR's own `Options(; crossover_probability=...)`). LaSR pins subtree crossover to `1 - p` so this is a true conditional probability. |

### What goes into a prompt

| Keyword | Default | Description |
| --- | --- | --- |
| `context` | `""` | A natural-language description of the problem, prepended to every prompt. Domain knowledge here is the single highest-leverage knob. |
| `variable_names` | `nothing` | Map from dataset names to meaningful names, e.g. `Dict("x1" => "theta")`. Falls back to the dataset's own names. |
| `num_pareto_context` | `5` | How many Pareto-frontier members — and how many concepts from the idea store — are shown to the LLM per call. |
| `num_generated_equations` | `5` | Expressions requested per call. Best-of-K for generation; the unused ones feed `suggestion_cache`. |
| `num_generated_concepts` | `5` | Concepts requested per concept-generation call. |
| `num_concept_crossover` | `2` | Concept pairs merged per concept-evolution round. |
| `prompts_dir` | `default_prompts_dir()` | Directory of `.prompt` templates. See [Prompt templates](#prompt-templates). A path that does not exist is rejected at construction, not minutes into a search. |

### Concept library

| Keyword | Default | Description |
| --- | --- | --- |
| `idea_database` | `String[]` | Concepts to seed the default store with. |
| `max_concepts` | `30` | Sampling window of the default store: retrieval draws from the `max_concepts` most recently refined concepts. |
| `idea_store` | `nothing` | Pass an `AbstractIdeaStore` to change how concepts are retrieved; overrides `idea_database`/`max_concepts`. |

Two stores ship with LaSR:

- **`WindowedIdeaStore(; window=30, seed=String[])`** — the default and the historical
  behavior. Refined concepts go to the front, raw ones to the back; retrieval is a uniform
  sample from the front `window`. Overflow past the window is what concept evolution
  distills. Ignores the query.
- **`ScoredIdeaStore(; k1=1.5, b=0.75, decay=0.99, refined_prior=2.0, seed=String[])`** —
  query-aware and quality-weighted: retrieves by `value × (1 + BM25 relevance to the
  expression being mutated)`, stochastically. Call `update_idea_value!(store, idea, delta)`
  to reinforce concepts that preceded an improvement, turning the library into a bandit
  over concepts. Values decay on each `add_idea!`, so stale concepts fade.

Implement `add_idea!`, `retrieve_ideas`, `evolution_candidates` and `Base.length` to plug
in your own (BM25, RAG, …).

### Cost control

| Keyword | Default | Description |
| --- | --- | --- |
| `suggestion_cache` | `nothing` | A `SuggestionCache(; capacity=8192)` pools the `num_generated_equations - 1` proposals each call already paid for and serves them to later identical prompts. Entries are *consumed*, so the population still sees varied material. Inspect with `cache_stats`. |
| `max_llm_calls` | `nothing` | Hard ceiling on real LLM calls for the whole run (not per iteration). Once spent, the LLM operators fall back to their symbolic counterparts for the rest of the search rather than stalling. Read the tally with `budget_used(plugin.call_budget)`. |

### Search quality

| Keyword | Default | Description |
| --- | --- | --- |
| `amnesty_complexity` | `0` | Complexity amnesty: any population member at or above this complexity has its constants re-optimized at the end of a generation, before selection can cull it, so good structure is not lost to a bad constant fit. Pure constant optimization — it runs even with `use_llm=false`. `0` disables the pass. |

### Parsing and observability

| Keyword | Default | Description |
| --- | --- | --- |
| `parse_rules` | `NormalizationRule[]` | Your own normalization rules, appended in order after the built-in defaults. See [Extending the parser](#extending-the-parser). |
| `parse_failure_sink` | `nothing` | Supply your own `ParseFailureStore` to keep a live handle on the store a (serial) search records into. |
| `lasr_logger` | `nothing` | A `LaSRLogger(SRLogger(...))` that records each LLM call — prompt, raw output, chosen expression. |

## Prompt templates

LaSR ships its templates inside the package; `default_prompts_dir()` returns where they
live and `LaSRPlugin()` uses them by default. There is no `prompts.zip` to download any
more. On a `Pkg.add` install that directory is read-only (files land mode 444) and is
replaced on upgrade, so edit a copy rather than the originals:

```julia
dir = copy_prompts("~/my_lasr_prompts")   # writable copies of every .prompt
plugin = LaSRPlugin(; prompts_dir=dir)
```

`prompts_dir` is *joined* with template names, so a trailing slash is optional, and the
directory needs only the templates you changed — anything absent falls back to the packaged
copy (`prompt_path(dir, name)` resolves one file).

From Python (`juliacall`, e.g. under PySR):

```python
from juliacall import Main as jl
jl.seval("using SymbolicRegression, LibraryAugmentedSymbolicRegression")
LaSR = jl.LibraryAugmentedSymbolicRegression

print(LaSR.default_prompts_dir())                          # read the shipped defaults
prompts_dir = str(LaSR.copy_prompts("~/my_lasr_prompts"))  # edit these, then pass along
```

## MLJ

LaSR has no MLJ models of its own — `LaSRRegressor` and `MultitargetLaSRRegressor` are
gone. Use SymbolicRegression's `SRRegressor` and `MultitargetSRRegressor` and pass the
plugin through `plugins`, exactly as you would to `Options`:

```julia
using MLJ
using SymbolicRegression
using LibraryAugmentedSymbolicRegression

plugin = LaSRPlugin(; model="my-model", api_key="token-abc123", mutate_weight=0.01)
model = SRRegressor(;
    plugins=(plugin,),
    niterations=40,
    binary_operators=[+, -, *, /],
)
mach = machine(model, X_table, y)
fit!(mach)
report(mach)
predict(mach, X_table)
```

## Extending the parser

LLMs write a dialect of their own: `|x|`, `ln(x)`, `pow(a, b)`, `C1*x0 + C2*x1`, `x_1`,
`2x`. LaSR normalizes those before parsing, through an ordered pipeline of
`NormalizationRule`s. Register your own for domain notation — they run after the defaults:

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

## Debugging LLM output

Unparseable model output falls back to a constant-1 node, which keeps the search alive but
hides *why* the output was unusable. Every fallback is recorded in a bounded store. Hand
the plugin your own store to keep a live handle on it:

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

If one shape dominates the summary, fix the root cause with a `parse_rules` entry instead
of eating the fallback forever.

## Running with Ollama

LaSR works with any OpenAI-compatible server. Ollama is a free one geared towards commodity
laptops; download it [here](https://ollama.com/download), then:

```bash
$ ollama pull llama3.1
# This downloads a 4GB-ish file that contains the Llama3.1 8B model.
# Ollama runs on port 11434 by default. A debug query to make sure we can connect:
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

Then run the search with `model="llama3.1:latest"` and
`api_kwargs=Dict("url" => "http://localhost:11434/v1", "max_tokens" => 4096)`:

```julia
using SymbolicRegression
using LibraryAugmentedSymbolicRegression
import MLJ: machine, fit!, predict, report

X = randn(Float64, 5, 100)
y = 2 * cos.(X[4, :]) + X[1, :] .^ 2 .- 2
y = y .+ randn(100) .* 1e-3

p = 1e-4 # Reduce this even further if the model takes too long...
model = SRRegressor(;
    plugins=(
        LaSRPlugin(;
            model="llama3.1:latest",
            api_key="token-abc123",
            api_kwargs=Dict("url" => "http://localhost:11434/v1", "max_tokens" => 4096),
            use_concepts=true,
            use_concept_evolution=true,
            context="We believe the relationship between the theta and offset parameter is a function of the cosine of the theta variable and the square of the offset.",
            variable_names=Dict("x1" => "theta", "x2" => "offset"),
            mutate_weight=p,
            randomize_weight=p,
            crossover_probability=p,
            max_llm_calls=500,   # a hard ceiling on this run's LLM traffic
            verbose=true,
        ),
    ),
    niterations=40,
    binary_operators=[+, -, *, /, ^],
    unary_operators=[cos],
    populations=20,
)

mach = machine(model, transpose(X), y)
fit!(mach)
report(mach)
predict(mach, transpose(X))
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

The test suite injects a fixed-output function through `LaSRPlugin(; llm_generate=...)`, so
prompt rendering, parsing, mutation/crossover dispatch, constraints and complete searches
are all exercised without network access or credentials.

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
