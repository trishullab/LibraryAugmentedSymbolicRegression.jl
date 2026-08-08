module LLMOptionsStructModule

using PromptingTools: aigenerate
using SymbolicRegression:
    AbstractCrossover, AbstractMutation, AbstractOptions, AbstractPlugin, Options
using ..LaSRMutationWeightsModule: LaSRMutationWeights
using ..LoggingModule: LaSRLogger

const DEFAULT_PROMPTS_DIR = joinpath(pkgdir(parentmodule(@__MODULE__)), "prompts") * "/"

Base.@kwdef mutable struct LLMOperationWeights
    llm_crossover::Float64 = 0.0
    llm_mutate::Float64 = 0.0
    llm_randomize::Float64 = 0.0
end

"""
    LLMOptions(; kws...)

Options for the language model itself: which model to call, and how. All
LaSR search/prompt behavior is configured directly on [`LaSRPlugin`](@ref).
"""
Base.@kwdef mutable struct LLMOptions
    api_key::Union{String,Nothing} = nothing
    model::Union{String,Nothing} = nothing
    api_kwargs::Dict = Dict("max_tokens" => 1000)
    http_kwargs::Dict = Dict("retries" => 3, "readtimeout" => 3600)
    llm_generate::Function = aigenerate
    verbose::Bool = true
end

struct LLMMutateMutation <: AbstractMutation end
struct LLMRandomizeMutation <: AbstractMutation end
struct LLMCrossover <: AbstractCrossover end

"""
    LaSRPlugin(; kws...)

Library-augmented symbolic regression plugin. Pass it through
`Options(; plugins=(LaSRPlugin(...),), ...)`. LLM client settings
(`model`, `api_key`, ...) live in [`LLMOptions`](@ref); everything else is
set directly on the plugin. Select the LLM operations by adding
[`LLMMutateMutation`](@ref), [`LLMRandomizeMutation`](@ref), and
[`LLMCrossover`](@ref) to the native `Options` operation lists.
"""
struct LaSRPlugin <: AbstractPlugin
    llm_options::LLMOptions
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
    idea_database::Vector{AbstractString}
    lasr_logger::Union{LaSRLogger,Nothing}
    function LaSRPlugin(;
        llm_options::LLMOptions=LLMOptions(),
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
        lasr_logger::Union{LaSRLogger,Nothing}=nothing,
    )
        return new(
            llm_options,
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
            AbstractString[idea_database...],
            lasr_logger,
        )
    end
end

mutable struct LaSRPluginState
    idea_database::Vector{AbstractString}
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

const _LLM_OPTIONS_KEYS = fieldnames(LLMOptions)
const _PLUGIN_KEYS = fieldnames(LaSRPlugin)

function Base.getproperty(context::LaSRContext, key::Symbol)
    if key in (:sr_options, :plugin, :state)
        return getfield(context, key)
    elseif key === :idea_database && !isnothing(getfield(context, :state))
        return getfield(context, :state).idea_database
    elseif key === :lasr_logger && !isnothing(getfield(context, :state))
        return getfield(context, :state).lasr_logger
    elseif key === :variable_names && !isnothing(getfield(context, :state))
        return getfield(context, :state).variable_names
    elseif key in _LLM_OPTIONS_KEYS
        return getproperty(getfield(context, :plugin).llm_options, key)
    elseif key in _PLUGIN_KEYS && !hasproperty(getfield(context, :sr_options), key)
        # `hasproperty` guard: never shadow SR options (e.g. `crossover_probability`)
        return getproperty(getfield(context, :plugin), key)
    else
        return getproperty(getfield(context, :sr_options), key)
    end
end

end
