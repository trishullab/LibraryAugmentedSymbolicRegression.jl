module LLMOptionsModule

using DispatchDoctor: @unstable
using StatsBase: StatsBase
using Base: isvalid
using SymbolicRegression
using ..LaSRMutationWeightsModule: LaSRMutationWeights

"""
    LLMOperationWeights(;kws...)

Defines the probability of different LLM-based mutation operations.
NOTE: The LLM operations can be significantly slower than their symbolic counterparts,
so higher probabilities will result in slower operations. By default, we set all probs to 0.0.
The maximum value for these parameters is 1.0 (100% of the time).
# Arguments
- `llm_crossover::Float64`: Probability of calling LLM version of crossover.
"""
Base.@kwdef mutable struct LLMOperationWeights
    llm_crossover::Float64 = 0.0
end

"""
    LLMOptions(;kws...)

This defines how to call the LLM inference functions. LLM inference is managed by PromptingTools.jl but
this module serves as the entry point to define new options for the LLM inference.
# Arguments
- `use_llm::Bool`: Whether to use LLM inference or not. (default: false)
- `use_concepts::Bool`: Whether to summarize programs into concepts and use the concepts to guide the search. (default: false)
    NOTE: If `use_llm` is false, then `use_concepts` will be ignored.
- `use_concept_evolution::Bool`: Whether to evolve the concepts after every iteration. (default: false)
    NOTE: If `use_concepts` is false, then `use_concept_evolution` will be ignored.
- `lasr_weights::LLMWeights`: lasr_weights for different LLM operations.
- `num_pareto_context::Int64`: Number of equations to sample from pareto frontier.
- `use_concepts::Bool`: Use natural language concepts in the LLM prompts. 
- `use_concept_evolution::Bool`: Evolve natural language concepts through succesive LLM
    calls.
- api_key::AbstractString: OpenAI API key. Required.
- model::AbstractString: OpenAI model to use. Required.
- api_kwargs::Dict: Additional keyword arguments to pass to the OpenAI API.
    - url::AbstractString: URL to send the request to. Required.
    - max_tokens::Int: Maximum number of tokens to generate. (default: 1000)
- http_kwargs::Dict: Additional keyword arguments for the HTTP request.
    - retries::Int: Number of retries to attempt. (default: 3)
    - readtimeout::Int: Read timeout for the HTTP request (in seconds; default is 1 hour).
- `llm_recorder_dir::AbstractString`: File to save LLM logs to. Useful for debugging.
- `llm_context::AbstractString`: Context AbstractString for LLM.
- `variable_names::Union{Dict,Nothing}`: Variable order for LLM. (default: nothing)
- `max_concepts::UInt32`: Number of concepts to keep track of. (default: 30)
- `verbose::Bool`: Output LLM generation query statistics. (default: true)
"""
Base.@kwdef mutable struct LLMOptions
    # LaSR Ablation Modifiers
    use_llm::Bool = false
    use_concepts::Bool = false
    use_concept_evolution::Bool = false
    lasr_mutation_weights::LaSRMutationWeights = LaSRMutationWeights()
    llm_operation_weights::LLMOperationWeights = LLMOperationWeights()

    # LaSR Performance Modifiers
    num_pareto_context::Int64 = 5
    num_generated_equations::Int64 = 5
    num_generated_concepts::Int64 = 5
    max_concepts::Int64 = 30
    # This is a cheeky hack to not have to deal with parametric types in LLMFunctions.jl. TODO: High priority rectify.
    is_parametric::Bool = false
    llm_context::AbstractString = ""

    # LaSR Bookkeeping Utilities
    # llm_logger::Union{SymbolicRegression.AbstractSRLogger, Nothing} = nothing
    llm_recorder_dir::AbstractString = "lasr_runs/"
    variable_names::Union{Dict,Nothing} = nothing
    prompts_dir::AbstractString = "prompts/"
    idea_database::Vector{AbstractString} = []

    # LaSR LLM API Options
    api_key::AbstractString = ""
    model::AbstractString = ""
    api_kwargs::Dict = Dict("max_tokens" => 1000)
    http_kwargs::Dict = Dict("retries" => 3, "readtimeout" => 3600)
    verbose::Bool = true
end

const llm_mutations = fieldnames(LLMOperationWeights)
const v_llm_mutations = Symbol[llm_mutations...]

# Validate some options are set correctly.
"""Validate some options are set correctly.
Specifically, need to check
- If `use_llm` is true, then `api_key` and `model` must be set.
- If `use_llm` is true, then `api_kwargs` must have a `url` key and it must be a valid URL.
- If `use_llm` is true, then `llm_recorder_dir` must be a valid directory.
"""
function validate_llm_options(options::LLMOptions)
    if options.use_llm
        if options.api_key == ""
            throw(ArgumentError("api_key must be set if LLM is use_llm."))
        end
        if options.model == ""
            throw(ArgumentError("model must be set if LLM is use_llm."))
        end
        if !haskey(options.api_kwargs, "url")
            throw(ArgumentError("api_kwargs must have a 'url' key."))
        end
        if !isdir(options.prompts_dir)
            throw(ArgumentError("options.prompts_dir not found."))
        end
    end
end

"""
    LaSROptions(;kws...)

This defines the options for the LibraryAugmentedSymbolicRegression module. It is a composite
type that contains both the LLMOptions and the SymbolicRegression.Options.
# Arguments
- `llm_options::LLMOptions`: Options for the LLM inference.
- `sr_options::SymbolicRegression.Options`: Options for the SymbolicRegression module.

# Example
```julia
llm_options = LLMOptions(;
    ...
)

options = Options(;
    binary_operators = (+, *, -, /, ^),
    unary_operators = (cos, log),
    nested_constraints = [(^) => [(^) => 0, cos => 0, log => 0], (/) => [(/) => 1], (cos) => [cos => 0, log => 0], log => [log => 0, cos => 0, (^) => 0]],
    constraints = [(^) => (3, 1), log => 5, cos => 7],
    populations=20,
)

lasr_options = LaSROptions(llm_options, options)
```

"""
struct LaSROptions{O<:SymbolicRegression.Options} <: SymbolicRegression.AbstractOptions
    llm_options::LLMOptions
    sr_options::O
end
const LLM_OPTIONS_KEYS = fieldnames(LLMOptions)

# Constructor with both sets of parameters:
@unstable function LaSROptions(; kws...)
    llm_options_keys = filter(k -> k in LLM_OPTIONS_KEYS, keys(kws))
    llm_options = LLMOptions(;
        NamedTuple(llm_options_keys .=> Tuple(kws[k] for k in llm_options_keys))...
    )
    sr_options_keys = filter(k -> !(k in LLM_OPTIONS_KEYS), keys(kws))
    sr_options = SymbolicRegression.Options(;
        NamedTuple(sr_options_keys .=> Tuple(kws[k] for k in sr_options_keys))...
    )
    return LaSROptions(llm_options, sr_options)
end

# Make all `Options` available while also making `llm_options` accessible
function Base.getproperty(options::LaSROptions, k::Symbol)
    if k in LLM_OPTIONS_KEYS
        return getproperty(getfield(options, :llm_options), k)
    else
        return getproperty(getfield(options, :sr_options), k)
    end
end

# Add setproperty! for `Options` and `llm_options`
function Base.setproperty!(options::LaSROptions, k::Symbol, v)
    if k in LLM_OPTIONS_KEYS
        return setproperty!(getfield(options, :llm_options), k, v)
    else
        return setproperty!(getfield(options, :sr_options), k, v)
    end
end

function Base.propertynames(options::LaSROptions)
    return (LLM_OPTIONS_KEYS..., fieldnames(SymbolicRegression.Options)...)
end

end # module
