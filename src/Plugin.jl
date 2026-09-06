module PluginModule

using PromptingTools: aigenerate
using SymbolicRegression
using SymbolicRegression:
    AbstractMutation, AbstractCrossover, AbstractOptions, AbstractPlugin, Options
using ..LaSRLoggerModule: LaSRLogger
using ..SuggestionCacheModule: SuggestionCache
using ..CallBudgetModule: CallBudget
using ..IdeaStoreModule: AbstractIdeaStore, WindowedIdeaStore
using ..NormalizationRulesModule: NormalizationRule
using ..ParseFailuresModule: ParseFailure
import ..ParseFailuresModule: ParseFailureStore, parse_failures, parse_failure_summary

"""
    default_prompts_dir()

The absolute path of the `prompts/` directory that ships with LaSR.

"""
function default_prompts_dir()::String
    root = pkgdir(parentmodule(@__MODULE__))
    root === nothing && error("cannot locate the LaSR package directory")
    return normpath(joinpath(root, "prompts"))
end

"""
    normalize_prompts_dir(dir) -> String

Expand `dir` to a canonical absolute path with no trailing separator, and check that the
directory exists.
"""
function normalize_prompts_dir(dir::AbstractString)::String
    raw = normpath(abspath(expanduser(String(dir))))
    # `normpath` keeps a trailing separator; drop it so `plugin.prompts_dir` is canonical
    # and "dir" and "dir/" are the same configuration.
    path = length(raw) > 1 ? String(rstrip(raw, ('/', '\\'))) : raw
    isdir(path) || throw(
        ArgumentError(
            "prompts_dir does not exist: $path -- pass a directory of `.prompt` " *
            "templates (see `copy_prompts`), or omit it to use `default_prompts_dir()`",
        ),
    )
    return path
end

struct LLMMutateMutation <: AbstractMutation end
struct LLMRandomizeMutation <: AbstractMutation end
struct LLMGenerateMutation <: AbstractMutation end
struct LLMCrossover <: AbstractCrossover end

"""
    LaSRPlugin(; kws...)

The library-augmented symbolic regression plugin. Pass it to SymbolicRegression as
`Options(; plugins=(LaSRPlugin(...),), ...)`.

# LLM operators

Each operator asks the LLM for expressions and then competes with the symbolic operators
of SR. All four are off by default. Set at least one weight, or the plugin makes no call.

- `LLMMutateMutation` (`mutate_weight`): show the LLM one expression and ask for a change
  to it.
- `LLMRandomizeMutation` (`randomize_weight`): ask for one new expression of random size,
  up to `curmaxsize` nodes, then fit its constants. This replaces a random restart.
- `LLMGenerateMutation` (`generate_weight`): ask for `num_generated_equations` complete
  expressions in one call, fit the constants of each one, and keep the best. The parent
  survives unchanged if no candidate meets the constraints.
- `LLMCrossover` (`crossover_probability`): show the LLM two parents and ask for two
  children.

The three weights are unnormalized, as with every entry in `Options.mutations`, and SR
normalizes the full set. `crossover_probability` is different: it is a probability in
`[0, 1]` that applies only after SR selects crossover, and LaSR gives the remaining
weight to `SubtreeCrossover`.

# Concepts

LaSR can hold a library of natural-language concepts and put them into its prompts.

`use_concepts=true` adds `num_pareto_context` concepts from the store to each operator
prompt. `use_concept_evolution=true` fills the store during the search: every
`populations` generations, LaSR shows the LLM the Pareto frontier and the worst members,
and asks for `num_generated_concepts` concepts. It adds `num_concept_crossover` of them
to the store, then runs `num_concept_crossover` merge steps. Each merge step sends the
`evolution_candidates` of the store and adds one merged concept to the front.

`use_concept_evolution` has an effect only with `use_concepts=true`. Alone, it generates
concepts that no prompt reads.

# Arguments

## Backend

LaSR uses `PromptingTools.jl` to reach an OpenAI-compatible server.

  * `model=nothing`: the model name on the server.
  * `api_key=nothing`: the API key for the server. A local server usually accepts any
    string.
  * `api_kwargs=Dict("max_tokens" => 4096)`: passed to the OpenAI schema of
    PromptingTools. It must hold a `"url"` entry.
  * `http_kwargs=Dict("retries" => 3, "readtimeout" => 3600)`: passed to the HTTP layer.
  * `llm_generate=aigenerate`: the function that makes the call. Replace it with a mock
    in a test.
  * `verbose=true`: print the token count and the elapsed time of each call.

## Search

  * `use_llm=true`: use the LLM operators. Set it to `false` for a plain SR run.
  * `mutate_weight=0.0`: the unnormalized weight of `LLMMutateMutation`.
  * `randomize_weight=0.0`: the unnormalized weight of `LLMRandomizeMutation`.
  * `generate_weight=0.0`: the unnormalized weight of `LLMGenerateMutation`.
  * `crossover_probability=0.0`: the probability of `LLMCrossover`, given that SR selects
    crossover.
  * `num_generated_equations=5`: the number of expressions that one call requests.
  * `context=""`: a description of the problem in natural language. It goes at the front
    of each operator prompt. Domain knowledge here is the strongest single control.
  * `variable_names=nothing`: a map from the dataset names to meaningful names, such as
    `Dict("x1" => "theta")`. The dataset names apply when this is `nothing`.
  * `prompts_dir=default_prompts_dir()`: the directory of the `.prompt` templates. See
    `copy_prompts`. The constructor rejects a path that does not exist.
  * `parse_rules=NormalizationRule[]`: extra rules for the expression dialect of the LLM.
    They run after `DEFAULT_RULES`, in the given order.
  * `amnesty_complexity=0`: refit the constants of each population member at or above
    this complexity at the end of a generation. This rescues an expression that has good
    structure and a bad constant fit. `0` turns it off. It also runs with `use_llm=false`.
  * `lasr_logger=nothing`: a `LaSRLogger` that records each LLM call. LaSR builds one
    from the `logger` that you give `equation_search`, so set this only when SR gets no
    logger.

## Concepts

  * `use_concepts=false`: put concepts from the store into the operator prompts.
  * `use_concept_evolution=false`: generate and merge concepts during the search.
  * `num_pareto_context=5`: how many concepts an operator prompt shows, and how many
    Pareto members and worst members a concept prompt shows.
  * `num_generated_concepts=5`: the number of concepts that one concept call requests.
  * `num_concept_crossover=2`: how many concepts LaSR adds per round, and how many merge
    steps it runs.

## Library

  * `idea_database=String[]`: the concepts that seed the default store.
  * `max_concepts=30`: the sampling window of the default store. Retrieval draws from the
    `max_concepts` newest refined concepts.
  * `idea_store=nothing`: an `AbstractIdeaStore` that replaces the default store and
    overrides `idea_database` and `max_concepts`. LaSR supplies `WindowedIdeaStore` and
    `ScoredIdeaStore`.

## Budget

  * `suggestion_cache=nothing`: a `SuggestionCache` that pools the proposals of a call
    that no operator used, and serves them to later requests. Read it with `cache_stats`.
  * `max_llm_calls=nothing`: a ceiling on the LLM calls of the full run. The LLM
    operators fall back to their symbolic counterparts after the ceiling. Read the count
    with `budget_used(plugin.call_budget)`.
  * `parse_failure_sink=nothing`: a `ParseFailureStore` that collects the LLM strings
    that the parser could not read. Read it with `parse_failures` and
    `parse_failure_summary`.
"""
struct LaSRPlugin <: AbstractPlugin
    api_key::Union{String,Nothing}
    model::Union{String,Nothing}
    api_kwargs::Dict
    http_kwargs::Dict
    llm_generate::Function
    verbose::Bool
    use_llm::Bool
    use_concepts::Bool
    use_concept_evolution::Bool
    num_pareto_context::Int
    num_generated_equations::Int
    num_generated_concepts::Int
    num_concept_crossover::Int
    max_concepts::Int
    context::String
    variable_names::Union{Dict,Nothing}
    prompts_dir::String
    idea_store::AbstractIdeaStore
    lasr_logger::Union{LaSRLogger,Nothing}
    mutate_weight::Float64
    randomize_weight::Float64
    crossover_probability::Float64
    generate_weight::Float64
    amnesty_complexity::Int
    parse_rules::Vector{NormalizationRule}
    suggestion_cache::Union{SuggestionCache,Nothing}
    call_budget::CallBudget
    parse_failure_sink::Union{Nothing,ParseFailureStore}
    function LaSRPlugin(;
        api_key::Union{String,Nothing}=nothing,
        model::Union{String,Nothing}=nothing,
        api_kwargs::Dict=Dict("max_tokens" => 4096),
        http_kwargs::Dict=Dict("retries" => 3, "readtimeout" => 3600),
        llm_generate::Function=aigenerate,
        verbose::Bool=true,
        use_llm::Bool=true,
        use_concepts::Bool=false,
        use_concept_evolution::Bool=false,
        num_pareto_context::Integer=5,
        num_generated_equations::Integer=5,
        num_generated_concepts::Integer=5,
        num_concept_crossover::Integer=2,
        max_concepts::Integer=30,
        context::AbstractString="",
        variable_names::Union{Dict,Nothing}=nothing,
        prompts_dir::AbstractString=default_prompts_dir(),
        idea_database::Vector{<:AbstractString}=AbstractString[],
        idea_store::Union{AbstractIdeaStore,Nothing}=nothing,
        lasr_logger::Union{LaSRLogger,Nothing}=nothing,
        mutate_weight::Real=0.0,
        randomize_weight::Real=0.0,
        crossover_probability::Real=0.0,
        generate_weight::Real=0.0,
        amnesty_complexity::Integer=0,
        parse_rules::Vector{NormalizationRule}=NormalizationRule[],
        suggestion_cache::Union{SuggestionCache,Nothing}=nothing,
        max_llm_calls::Union{Int,Nothing}=nothing,
        parse_failure_sink::Union{Nothing,ParseFailureStore}=nothing,
    )
        mutate_weight >= 0 || throw(ArgumentError("`mutate_weight` must be nonnegative."))
        randomize_weight >= 0 ||
            throw(ArgumentError("`randomize_weight` must be nonnegative."))
        0 <= crossover_probability <= 1 ||
            throw(ArgumentError("`crossover_probability` must be between 0 and 1."))
        generate_weight >= 0 ||
            throw(ArgumentError("`generate_weight` must be nonnegative."))
        amnesty_complexity >= 0 ||
            throw(ArgumentError("`amnesty_complexity` must be nonnegative."))
        if use_llm &&
            iszero(mutate_weight) &&
            iszero(randomize_weight) &&
            iszero(crossover_probability) &&
            iszero(generate_weight)
            @warn "`LaSRPlugin` has `use_llm=true` but every LLM operator weight is zero, " *
                "so no LLM call will ever be made. Set at least one of `mutate_weight`, " *
                "`randomize_weight`, `generate_weight`, or `crossover_probability`."
        end
        isnothing(max_llm_calls) ||
            max_llm_calls > 0 ||
            throw(ArgumentError("`max_llm_calls` must be positive, or `nothing`."))
        store = something(
            idea_store,
            WindowedIdeaStore(; window=Int(max_concepts), seed=String[idea_database...]),
        )
        return new(
            api_key,
            model,
            api_kwargs,
            http_kwargs,
            llm_generate,
            verbose,
            use_llm,
            use_concepts,
            use_concept_evolution,
            num_pareto_context,
            num_generated_equations,
            num_generated_concepts,
            num_concept_crossover,
            max_concepts,
            context,
            variable_names,
            normalize_prompts_dir(prompts_dir),
            store,
            lasr_logger,
            mutate_weight,
            randomize_weight,
            crossover_probability,
            generate_weight,
            amnesty_complexity,
            parse_rules,
            suggestion_cache,
            CallBudget(max_llm_calls),
            parse_failure_sink,
        )
    end
end

mutable struct LaSRPluginState
    idea_store::AbstractIdeaStore
    lasr_logger::Union{LaSRLogger,Nothing}
    variable_names::Dict
    generations::Int
    worst_members::Vector{Any}
    # Held BY REFERENCE across `fork_plugin_state`/`refresh_worker_plugin_state` (unlike
    # every other field above, which is deep/shallow-copied) so parse failures recorded by
    # any `:serial`/`:multithreading` worker aggregate into one shared, lock-guarded store.
    # See `src/SRInterface.jl` (`_copy_plugin_state`) and `src/ParseFailures.jl` (`ParseFailureStore`).
    parse_failures::ParseFailureStore
end

struct LaSRContext{O<:Options,S} <: AbstractOptions
    sr_options::O
    plugin::LaSRPlugin
    state::S
end

const _CLIENT_KEYS = (:api_key, :model, :api_kwargs, :http_kwargs, :llm_generate, :verbose)
const _PLUGIN_KEYS = fieldnames(LaSRPlugin)

function Base.getproperty(context::LaSRContext, key::Symbol)
    if key in (:sr_options, :plugin, :state)
        return getfield(context, key)
    elseif key === :idea_store && !isnothing(getfield(context, :state))
        return getfield(context, :state).idea_store
    elseif key === :lasr_logger && !isnothing(getfield(context, :state))
        return getfield(context, :state).lasr_logger
    elseif key === :variable_names && !isnothing(getfield(context, :state))
        return getfield(context, :state).variable_names
    elseif key in _CLIENT_KEYS
        return getproperty(getfield(context, :plugin), key)
    elseif key in _PLUGIN_KEYS && !hasproperty(getfield(context, :sr_options), key)
        # `hasproperty` guard: never shadow SR options (e.g. `crossover_probability`)
        return getproperty(getfield(context, :plugin), key)
    else
        return getproperty(getfield(context, :sr_options), key)
    end
end

"""
    parse_failures(ctx::LaSRContext) -> Vector{ParseFailure}

Return the parse failures that the search recorded, or no failures if `ctx` holds no
plugin state.
"""
function parse_failures(ctx::LaSRContext)
    state = getfield(ctx, :state)
    state isa LaSRPluginState || return ParseFailure[]
    return parse_failures(state.parse_failures)
end
function parse_failure_summary(ctx::LaSRContext; n::Int=10)
    state = getfield(ctx, :state)
    state isa LaSRPluginState || return Pair{String,Int}[]
    return parse_failure_summary(state.parse_failures; n=n)
end

lasr_context(context::LaSRContext, state=nothing) = context

function lasr_plugin(options::SymbolicRegression.Options)
    matches = filter(plugin -> plugin isa LaSRPlugin, options.plugins)
    length(matches) == 1 ||
        throw(ArgumentError("Expected exactly one LaSRPlugin in `options.plugins`."))
    return only(matches)::LaSRPlugin
end

function lasr_context(options::SymbolicRegression.Options, state=nothing)
    plugin = lasr_plugin(options)
    return LaSRContext(options, plugin, state)
end

function lasr_state(options::SymbolicRegression.Options, plugin_states::Tuple)
    index = findfirst(plugin -> plugin isa LaSRPlugin, options.plugins)
    isnothing(index) && throw(ArgumentError("LaSRPlugin is not active."))
    return plugin_states[index]::LaSRPluginState
end

end
