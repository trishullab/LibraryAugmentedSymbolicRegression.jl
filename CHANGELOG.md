# Changelog

## v0.4.0

LaSR is now a **plugin** for SymbolicRegression.jl v2 rather than a wrapper around it.

### Breaking

- `LaSROptions`, `LLMOptions`, `LaSRMutationWeights`, and `LLMOperationWeights` are gone.
  The entry point is now:

  ```julia
  Options(; plugins=(LaSRPlugin(; mutate_weight=0.05, ...),), mutation_weights=...)
  ```

  Structural mutation weights live on SymbolicRegression's own `mutation_weights`; the LLM
  operator weights (`mutate_weight`, `randomize_weight`, `generate_weight`,
  `crossover_probability`) are fields on `LaSRPlugin`.
- **`LaSRRegressor` and `MultitargetLaSRRegressor` are gone.** They were keyword sugar that
  returned a plain `SRRegressor`/`MultitargetSRRegressor` with the plugin appended. Attach the
  plugin yourself:

  ```julia
  SRRegressor(; plugins=(LaSRPlugin(...),), niterations=40, ...)
  ```
- `SymbolicRegression` is pinned to `v2.0.0-beta.2`. A registry release is blocked until
  SymbolicRegression 2.0 is final.
- Prompt templates are resolved at runtime from the package, with a per-file fallback. The
  `prompts.zip` download is gone; use `copy_prompts` to get an editable copy.
- **LaSR no longer re-exports SymbolicRegression.** As a fork it mirrored SR.jl's whole
  namespace so it could stand in for it; as a plugin it exports only its own surface. Import
  both:

  ```julia
  using SymbolicRegression, LibraryAugmentedSymbolicRegression
  ```

  `LaSRPlugin` comes from LaSR; `Options`, `SRRegressor`, `equation_search`,
  `calculate_pareto_frontier`, `string_tree` and the rest come from SymbolicRegression.
- **The bundled llamafile server is gone**, along with the `LLAMAFILE_MODEL`,
  `LLAMAFILE_PATH`, `LLAMAFILE_URL` and `LLM_PORT` exports and the `__init__` hook that could
  download and spawn it. Point `api_kwargs["url"]` at an endpoint you control. A plugin should
  not stand up its own inference server on import.
- The MLJ compatibility stress suite inherited from SymbolicRegression.jl is removed;
  SymbolicRegression tests its own MLJ interface upstream.

### Security

- `parse_msg_content` no longer calls `eval` on model output. LLM responses are untrusted
  input, and evaluating them was remote code execution. A literal-only reader
  (`safe_literal_parse`) replaces it.

### Fixed

- An empty `variable_names` Dict fell back to an empty name list, so every parse failed with
  ``Variable `x` not found`` and *every* LLM suggestion was silently discarded.
- `LaSRPlugin` now warns when `use_llm=true` but every operator weight is zero, instead of
  silently attaching a plugin that never fires.
- The default `max_tokens` is 4096. At 1000 a majority of responses from common reasoning
  models truncate before their closing fence and are discarded.

### Added

- **`LLMGenerateMutation`** — best-of-K structural generation: asks for K full expressions,
  constant-fits each, keeps the best. Configured by `generate_weight` and
  `num_generated_equations`.
- **Complexity amnesty** (`amnesty_complexity`) — re-optimizes the constants of complex
  population members before selection can cull them, so good structure is not lost to a bad
  constant fit.
- **Constant fitting for generated trees** — a freshly generated skeleton has its constants
  fit before it competes, mirroring generate-then-optimize.
- **`LLMCrossover`** — the crossover operator ported to SymbolicRegression's
  `AbstractCrossover` API.
- **A normalization pipeline** (`NormalizationRule`, `DEFAULT_RULES`) replacing the ad-hoc
  string munging: `|x|`→`abs`, `ln`→`log`, `pow(a,b)`→`a^b`, unary sign, indexed constant
  placeholders, subscript collapsing, and implicit multiplication/application inherited from
  SymPy's transformation set. Scientists can register their own rules via
  `LaSRPlugin(; parse_rules=[...])`.
- **A parse-failure sink** — unparseable model output is recorded in a bounded store instead
  of vanishing into a silent constant-1 fallback. Inspect with `parse_failures` and
  `parse_failure_summary`.
- **Pluggable concept libraries** — `AbstractIdeaStore`, with `WindowedIdeaStore` (the
  historical behavior) and `ScoredIdeaStore` (quality-weighted, query-aware).
- **A suggestion pool and call budget** — every LLM call already asks for N proposals and
  uses one; the rest are banked and served to later identical prompts. `max_llm_calls` puts
  a hard ceiling on a run's LLM traffic.
- Mock-server tests covering the full LLM round trip, so the LLM path is exercised in CI
  without a model or network access.

### Removed

- Everything the package carried only because it used to be a fork of SymbolicRegression.jl:
  a dead copy of SR.jl's `Utils.jl`, the `pipelines.sh` runner, an inert
  `LocalPreferences.toml`, an archived prompt set, and a `coverage.jl` script whose CI
  pipeline produced a report nothing consumed. With them go seven dependencies that had no
  remaining consumer: `Downloads`, `MacroTools`, and the test-only `MLJBase`,
  `MLJModelInterface`, `MLJTestInterface`, `Suppressor` and `SymbolicUtils`.
- Static analysis is back: `test_jet.jl` runs again, scoped to LaSR's own modules.
