# LaSR: Library-Augmented Symbolic Regression

LibraryAugmentedSymbolicRegression.jl (LaSR) guides
[SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl)
searches with LLM-generated mutations, random expressions, crossovers, and an
optional evolving concept library. The method is described in the
[LaSR paper](https://arxiv.org/abs/2409.09359).

This branch targets SymbolicRegression.jl v2's plugin API.

## Usage

Configure the LLM boundary, put it in a `LaSRPlugin`, and pass the plugin to
the ordinary SR `Options` constructor:

```julia
using LibraryAugmentedSymbolicRegression

llm = LLMOptions(;
    model="my-model",
    api_key="...",
    api_kwargs=Dict(
        "url" => "http://localhost:11440/v1",
        "max_tokens" => 1000,
    ),
)

plugin = LaSRPlugin(;
    llm_options=llm,
    use_concepts=true,
    use_concept_evolution=true,
    context="The response depends on an angle and an offset.",
)

options = Options(;
    binary_operators=[+, -, *, /],
    unary_operators=[cos],
    plugins=(plugin,),
    mutations=(LLMMutateMutation() => 0.01, LLMRandomizeMutation() => 0.001),
    crossovers=(LLMCrossover() => 0.01,),
)

X = randn(2, 100)
y = @. 2cos(X[1, :]) + X[2, :]^2
hall_of_fame = equation_search(X, y; options, niterations=40)
```

The operation weights are unnormalized, exactly like other entries in
`Options.mutations` and `Options.crossovers`. SR's `crossover_probability`
still controls how often the search selects the crossover path at all.

LaSR ships its prompt templates in `prompts/`; `LaSRPlugin()` uses that package
directory by default. Set `prompts_dir` to use domain-specific templates.

## MLJ

`LaSRRegressor` and `MultitargetLaSRRegressor` are small constructors around
SR's native MLJ models:

```julia
using MLJ
using LibraryAugmentedSymbolicRegression

plugin = LaSRPlugin(;
    llm_options=LLMOptions(; model="my-model", api_key="..."),
)
model = LaSRRegressor(;
    plugin,
    mutations=(LLMMutateMutation() => 0.01,),
    niterations=40,
    binary_operators=[+, -, *, /],
)
mach = machine(model, X_table, y)
fit!(mach)
```

## Configuration

`LaSRPlugin` controls search and prompt policy:

- `use_llm`, `use_concepts`, and `use_concept_evolution` enable the LLM and
  concept-library features.
- `num_pareto_context`, `num_generated_equations`,
  `num_generated_concepts`, `num_concept_crossover`, and `max_concepts`
  control prompt context and output counts.
- `context`, `variable_names`, `prompts_dir`, and `idea_database` provide
  domain context. If `variable_names` is omitted, LaSR uses the SR dataset's
  names.
- `Options.mutations` and `Options.crossovers` set how often the LLM
  operations run, using `LLMMutateMutation`, `LLMRandomizeMutation`, and
  `LLMCrossover`.

`LLMOptions` only carries settings for the language model itself:

- `api_key`, `model`, `api_kwargs`, and `http_kwargs` are forwarded to
  PromptingTools' OpenAI-compatible schema.
- `llm_generate` is the generation function. Its default is
  `PromptingTools.aigenerate`; tests can inject a deterministic local function.

`LaSROptions`, `LaSRMutationWeights`, and `LLMOperationWeights` remain as a
compatibility path for older Julia callers, but new code should use
the native `Options` operation lists with `LaSRPlugin`.

## Plugin mapping

LaSR uses SR v2 extension points without replacing the search loop:

- `LLMMutateMutation` and `LLMRandomizeMutation` are custom
  `AbstractMutation`s selected through `Options.mutations`.
- `LLMCrossover` is an `AbstractCrossover` selected through
  `Options.crossovers`; SR retains retries, constraint checks, evaluation,
  accounting, and replacement.
- logger initialization runs in `on_search_start!`.
- concept evolution runs serially on the head node in `on_generation_end!`.
- concept and logger state is copied to workers through the plugin-state
  lifecycle.

The previous full `_main_search_loop!` override and `AbstractOptions` wrapper
are no longer used.

## Testing without an LLM service

The focused tests inject a function returning fixed JSON through
`LLMOptions(; llm_generate=...)`. This exercises prompt rendering, parsing,
mutation/crossover dispatch, constraints, evaluation, and complete SR searches
without network access or credentials.

## Benchmark reproduction

The code used for the paper remains archived on the
[`lasr-experiments`](https://github.com/trishullab/LaSR.jl/tree/lasr-experiments)
branch.
