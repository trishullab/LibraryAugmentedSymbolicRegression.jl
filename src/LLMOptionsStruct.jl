module LLMOptionsStructModule

using DispatchDoctor: @unstable
using PromptingTools: aigenerate
using SymbolicRegression:
    AbstractMutation, AbstractCrossover, AbstractOptions, AbstractPlugin, Options
using ..LoggingModule: LaSRLogger
using ..IdeaStoreModule: AbstractIdeaStore, WindowedIdeaStore
using ..NormalizeModule: NormalizationRule, ParseFailure
import ..NormalizeModule: ParseFailureStore, parse_failures, parse_failure_summary

const DEFAULT_PROMPTS_DIR = joinpath(pkgdir(parentmodule(@__MODULE__)), "prompts") * "/"

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
    is_parametric::Bool
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
    # Optional externally-owned parse-failure store. When set, `init_plugin_state` uses
    # THIS store instead of building a fresh one, so a caller (e.g. the PySR seam) holds a
    # live handle to the exact store the (serial) search records into. Default `nothing`.
    parse_failure_sink::Union{Nothing,ParseFailureStore}
    function LaSRPlugin(;
        api_key::Union{String,Nothing}=nothing,
        model::Union{String,Nothing}=nothing,
        api_kwargs::Dict=Dict("max_tokens" => 1000),
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
        is_parametric::Bool=false,
        context::AbstractString="",
        variable_names::Union{Dict,Nothing}=nothing,
        prompts_dir::AbstractString=DEFAULT_PROMPTS_DIR,
        idea_database::Vector{<:AbstractString}=AbstractString[],
        # Pass a custom `AbstractIdeaStore` (BM25, Scored, RAG, ...) to change how
        # concepts are retrieved. Defaults to a `WindowedIdeaStore` seeded from
        # `idea_database` and sized to `max_concepts`, reproducing the historical
        # uniform-random windowed sampling.
        idea_store::Union{AbstractIdeaStore,Nothing}=nothing,
        lasr_logger::Union{LaSRLogger,Nothing}=nothing,
        mutate_weight::Real=0.0,
        randomize_weight::Real=0.0,
        crossover_probability::Real=0.0,
        generate_weight::Real=0.0,
        # Structural amnesty: any population member whose complexity is at least
        # `amnesty_complexity` has its constants re-optimized in `on_generation_end!`
        # before selection can cull it, so good structure is not lost to a bad
        # constant fit. `0` disables the pass.
        amnesty_complexity::Integer=0,
        # Scientist-registerable string/expr normalization rules, appended after
        # `DEFAULT_RULES` (in order) by `parse_expr` when this plugin is active.
        parse_rules::Vector{NormalizationRule}=NormalizationRule[],
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
        store = something(
            idea_store,
            WindowedIdeaStore(;
                window=Int(max_concepts), seed=String[idea_database...]
            ),
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
            is_parametric,
            String(context),
            variable_names,
            String(prompts_dir),
            store,
            lasr_logger,
            Float64(mutate_weight),
            Float64(randomize_weight),
            Float64(crossover_probability),
            Float64(generate_weight),
            Int(amnesty_complexity),
            parse_rules,
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
    # See `src/Mutate.jl` (`_copy_plugin_state`) and `src/Normalize.jl` (`ParseFailureStore`).
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
# `src/Parse.jl`.
@unstable function parse_failures(ctx::LaSRContext)
    state = getfield(ctx, :state)
    state isa LaSRPluginState || return ParseFailure[]
    return parse_failures(state.parse_failures)
end
@unstable function parse_failure_summary(ctx::LaSRContext; n::Int=10)
    state = getfield(ctx, :state)
    state isa LaSRPluginState || return Pair{String,Int}[]
    return parse_failure_summary(state.parse_failures; n=n)
end

end
