module LLMOptionsStructModule

using DispatchDoctor: @unstable
using StatsBase: StatsBase
using Base: isvalid
using SymbolicRegression
using ..LaSRMutationWeightsModule: LaSRMutationWeights
using ..LoggingModule: LaSRLogger

"""
    LLMOperationWeights(;kws...)

Defines the probability of different LLM-based mutation operations.
NOTE: The LLM operations can be significantly slower than their symbolic counterparts,
so higher probabilities will result in slower operations. By default, we set all probs to 0.0.
The maximum value for these parameters is 1.0 (100% of the time).
# Arguments
- `llm_crossover::Float64`: Probability of calling LLM version of crossover.
- `llm_mutate::Float64`: Probability of calling LLM version of mutation.
- `llm_gen_random::Float64`: Probability of calling LLM version of gen_random.
"""
Base.@kwdef mutable struct LLMOperationWeights
    llm_crossover::Float64 = 0.0
    llm_mutate::Float64 = 0.0
    llm_randomize::Float64 = 0.0
end

"""
    set_llm_mutation_weights(mutation_weights::LaSRMutationWeights, llm_operation_weights::LLMOperationWeights)

Set the mutation weights for a LaSR model based on the provided LLM operation weights.

# Arguments
- `mutation_weights::LaSRMutationWeights`: An instance of `LaSRMutationWeights` containing the current mutation weights.
- `llm_operation_weights::LLMOperationWeights`: An instance of `LLMOperationWeights` containing the weights for LLM operations.

# Returns
- `mutation_weights::LaSRMutationWeights`: The updated `mutation_weights` with modified values based on the LLM operation weights.

# Description
This function adjusts the mutation weights of a LaSR model by incorporating the weights from LLM operations. It performs the following steps:
1. Extracts the `randomize` weight from `mutation_weights`.
2. Creates a dictionary of other mutation weights excluding those starting with `llm_` or `randomize`.
3. Computes new values for `llm_randomize` and `llm_mutate` based on the LLM operation weights and the current mutation weights.
4. Updates the original mutation weights by scaling them with the complementary LLM operation weights.
5. Sets the new values for `llm_randomize` and `llm_mutate` in the `mutation_weights`.

The function returns the updated `mutation_weights` with the new values.

We give special consideration to the `randomize` weight because it is independently sampled from the other mutation weights.
"""
function set_llm_mutation_weights(
    mutation_weights::LaSRMutationWeights, llm_operation_weights::LLMOperationWeights
)::LaSRMutationWeights
    randomize_w = mutation_weights.randomize
    oth_mutations = Dict([
        sym => getproperty(mutation_weights, sym) for
        sym in fieldnames(typeof(mutation_weights)) if
        !startswith(string(sym), r"llm_|randomize")
    ])

    llm_randomize_w = llm_operation_weights.llm_randomize * randomize_w
    llm_mutate_w =
        llm_operation_weights.llm_mutate * sum(values(oth_mutations)) /
        length(oth_mutations)

    # modify the original values of the mutation weights
    mutation_weights.randomize = (1 - llm_operation_weights.llm_randomize) * randomize_w
    for (sym, val) in oth_mutations
        setproperty!(mutation_weights, sym, (1 - llm_operation_weights.llm_mutate) * val)
    end

    # if llm_randomize or llm_mutate are not 0.0, we should respect user values.
    mutation_weights.llm_randomize = if mutation_weights.llm_randomize == 0.0
        llm_randomize_w
    else
        mutation_weights.llm_mutate
    end
    mutation_weights.llm_mutate =
        mutation_weights.llm_mutate == 0.0 ? llm_mutate_w : mutation_weights.llm_mutate

    return mutation_weights
end

function set_llm_mutation_weights(
    mutation_weights::NamedTuple, llm_operation_weights::NamedTuple
)::LaSRMutationWeights
    return set_llm_mutation_weights(
        LaSRMutationWeights(; mutation_weights...),
        LLMOperationWeights(; llm_operation_weights...),
    )
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
    mutation_weights::Union{LaSRMutationWeights,Nothing}
    llm_operation_weights::Union{LLMOperationWeights,Nothing}
    # LaSR Performance Modifiers
    num_pareto_context::Integer
    num_generated_equations::Integer
    num_generated_concepts::Integer
    num_concept_crossover::Integer
    max_concepts::Integer
    # This is a cheeky hack to not have to deal with parametric types in LLMFunctions.jl. TODO: High priority rectify.
    is_parametric::Bool
    llm_context::Union{String,Nothing}

    # LaSR Bookkeeping Utilities
    # llm_logger::Union{SymbolicRegression.AbstractSRLogger, Nothing}
    variable_names::Union{Dict,Nothing}
    prompts_dir::Union{String,Nothing}
    idea_database::Union{Vector{AbstractString},Nothing}

    # LaSR LLM API Options
    api_key::Union{String,Nothing}
    model::Union{String,Nothing}
    api_kwargs::Union{Dict,Nothing}
    http_kwargs::Union{Dict,Nothing}
    lasr_logger::Union{LaSRLogger,Nothing}
    verbose::Bool
    tracking::Bool
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
