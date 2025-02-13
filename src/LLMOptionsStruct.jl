module LLMOptionsStructModule

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
"""
Base.@kwdef mutable struct LLMOptions
    # LaSR Ablation Modifiers
    use_llm::Bool
    use_concepts::Bool
    use_concept_evolution::Bool
    lasr_mutation_weights::Union{LaSRMutationWeights,Nothing}
    llm_operation_weights::Union{LLMOperationWeights,Nothing}
    # LaSR Performance Modifiers
    num_pareto_context::Integer
    num_generated_equations::Integer
    num_generated_concepts::Integer
    max_concepts::Integer
    # This is a cheeky hack to not have to deal with parametric types in LLMFunctions.jl. TODO: High priority rectify.
    is_parametric::Bool
    llm_context::Union{String,Nothing}

    # LaSR Bookkeeping Utilities
    # llm_logger::Union{SymbolicRegression.AbstractSRLogger, Nothing}
    llm_recorder_dir::Union{String,Nothing}
    variable_names::Union{Dict,Nothing}
    prompts_dir::Union{String,Nothing}
    idea_database::Union{Vector{AbstractString},Nothing}

    # LaSR LLM API Options
    api_key::Union{String,Nothing}
    model::Union{String,Nothing}
    api_kwargs::Union{Dict,Nothing}
    http_kwargs::Union{Dict,Nothing}
    verbose::Bool
end

const llm_mutations = fieldnames(LLMOperationWeights)
const v_llm_mutations = Symbol[llm_mutations...]

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

end # module
