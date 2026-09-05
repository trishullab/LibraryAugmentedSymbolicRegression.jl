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

plugin = LaSRPlugin(;
    model="my-model",
    api_key="...",
    api_kwargs=Dict(
        "url" => "http://localhost:11440/v1",
        "max_tokens" => 1000,
    ),
    use_concepts=true,
    use_concept_evolution=true,
    context="The response depends on an angle and an offset.",
    mutate_weight=0.01,
    randomize_weight=0.001,
    crossover_probability=0.01,
)

options = Options(;
    binary_operators=[+, -, *, /],
    unary_operators=[cos],
    plugins=(plugin,),
)

X = randn(2, 100)
y = @. 2cos(X[1, :]) + X[2, :]^2
hall_of_fame = equation_search(X, y; options, niterations=40)
```

The mutation weights are unnormalized, exactly like other entries in
`Options.mutations`. `crossover_probability` is conditional on SR first
selecting crossover via `crossover_probability`.

## Prompt templates

LaSR ships its templates inside the package; `default_prompts_dir()` returns where
they live and `LaSRPlugin()` uses them by default. On a `Pkg.add` install that
directory is read-only (files land mode 444) and is replaced on upgrade, so edit a
copy rather than the originals:

```julia
dir = copy_prompts("~/my_lasr_prompts")   # writable copies of every .prompt
plugin = LaSRPlugin(; prompts_dir=dir)
```

`prompts_dir` is *joined* with template names, so a trailing slash is optional, and
the directory needs only the templates you changed -- anything absent falls back to
the packaged copy (`prompt_path(dir, name)` resolves one file). A `prompts_dir` that
does not exist is rejected when the plugin is constructed, not minutes into a search.

From Python (`juliacall`, e.g. under PySR):

```python
from juliacall import Main as jl
jl.seval("using LibraryAugmentedSymbolicRegression")
LaSR = jl.LibraryAugmentedSymbolicRegression

print(LaSR.default_prompts_dir())                        # read the shipped defaults
prompts_dir = str(LaSR.copy_prompts("~/my_lasr_prompts"))  # edit these, then pass along
```

## MLJ

`LaSRRegressor` and `MultitargetLaSRRegressor` are small constructors around
SR's native MLJ models:

```julia
using MLJ
using LibraryAugmentedSymbolicRegression

plugin = LaSRPlugin(; model="my-model", api_key="...", mutate_weight=0.01)
model = LaSRRegressor(;
    plugin,
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
- `mutate_weight`, `randomize_weight`, and `crossover_probability` set how
  often the LLM mutations and crossover run.

LLM client settings are set directly on `LaSRPlugin`:

- `api_key`, `model`, `api_kwargs`, and `http_kwargs` are forwarded to
  PromptingTools' OpenAI-compatible schema.
- `llm_generate` is the generation function. Its default is
  `PromptingTools.aigenerate`; tests can inject a deterministic local function.

`LaSRPlugin` is the sole entry point — `Options(; plugins=(LaSRPlugin(...),))`.
Structural mutation weights (`mutate_constant`, `add_node`, ...) are set with
SR's own `Options(; mutation_weights=...)`; LLM-specific weights
(`mutate_weight`, `randomize_weight`, `crossover_probability`,
`generate_weight`) live on the plugin.

## Plugin mapping

LaSR uses SR v2 extension points without replacing the search loop:

- `LLMMutateMutation` and `LLMRandomizeMutation` are custom
  `AbstractMutation`s contributed by the plugin.
- LLM crossover proposes child expressions through SR's crossover proposal
  hook; SR retains constraint checks, evaluation accounting, and fallback.
- logger initialization runs in `on_search_start!`.
- concept evolution runs serially on the head node in `on_generation_end!`.
- concept and logger state is copied to workers through the plugin-state
  lifecycle.

The previous full `_main_search_loop!` override and `AbstractOptions` wrapper
are no longer used.

## Testing without an LLM service

The focused tests inject a function returning fixed JSON through
`LaSRPlugin(; llm_generate=...)`. This exercises prompt rendering, parsing,
mutation/crossover dispatch, constraints, evaluation, and complete SR searches
without network access or credentials.

## Benchmark reproduction

The code used for the paper remains archived on the
[`lasr-experiments`](https://github.com/trishullab/LaSR.jl/tree/lasr-experiments)
branch.
