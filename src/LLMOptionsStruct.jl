module LLMOptionsStructModule

using PromptingTools: aigenerate
using SymbolicRegression:
    AbstractMutation, AbstractCrossover, AbstractOptions, AbstractPlugin, Options
using ..LoggingModule: LaSRLogger
using ..IdeaStoreModule: AbstractIdeaStore, WindowedIdeaStore

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
        )
    end
end

mutable struct LaSRPluginState
    idea_store::AbstractIdeaStore
    lasr_logger::Union{LaSRLogger,Nothing}
    variable_names::Dict
    generations::Int
    worst_members::Vector{Any}
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

end
