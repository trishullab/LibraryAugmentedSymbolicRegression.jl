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

Absolute path of the prompt templates shipped with LaSR (the package's `prompts/`
directory). Deliberately a function, not a `const`: a constant computed from
`pkgdir` is evaluated at *precompile* time and its absolute path is frozen into the
cache, so a depot that is later moved or copied (container image, restored CI
cache, depot built as one user and run as another) keeps a valid pkgimage pointing
at a path that no longer exists, and every prompt load fails. Resolving per call
tracks wherever the package actually lives.

On a `Pkg.add` install this directory is READ-ONLY (files land mode 444). To edit
the templates, materialize a writable copy with [`copy_prompts`](@ref) and pass it
as `prompts_dir`.
"""
function default_prompts_dir()::String
    root = pkgdir(parentmodule(@__MODULE__))
    root === nothing && error("cannot locate the LaSR package directory")
    return normpath(joinpath(root, "prompts"))
end

# `prompts_dir` is joined with template names (never concatenated), so a trailing
# separator is optional. A directory that does not exist is a typo: fail here rather
# than minutes into a search at the first LLM call.
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

Library-augmented symbolic regression plugin. Pass it through
`Options(; plugins=(LaSRPlugin(...),), ...)`. LLM client settings
(`model`, `api_key`, ...) and everything else are set directly on the
plugin. The mutation weights are unnormalized, like all entries in
`Options.mutations`; `crossover_probability` is conditional on SR selecting
crossover.

Every keyword is documented in the "Configuration" section of the README.
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
            Int(num_pareto_context),
            Int(num_generated_equations),
            Int(num_generated_concepts),
            Int(num_concept_crossover),
            Int(max_concepts),
            String(context),
            variable_names,
            normalize_prompts_dir(prompts_dir),
            store,
            lasr_logger,
            Float64(mutate_weight),
            Float64(randomize_weight),
            Float64(crossover_probability),
            Float64(generate_weight),
            Int(amnesty_complexity),
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

# `ctx.state` is `nothing` for a `LaSRContext` built directly from a bare `Options` (e.g.
# a parser unit test with no plugin state); return the empty-store answer rather than
# erroring, matching the guard used at the `record_parse_failure!` call sites in
# `src/ExpressionIO.jl`.
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
