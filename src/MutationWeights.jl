module LaSRMutationWeightsModule

using DispatchDoctor: @unstable
using Base
using SymbolicRegression

"""
    LLMMutationProbabilities(;kws...)

Defines the probability of different LLM-based mutation operations.
NOTE:
    - These must sum up to 1.0.
    - The LLM operations can be significantly slower than their symbolic counterparts,
   so higher probabilities will result in slower operations. By default, we set all probs to 0.0.

# Arguments
- `llm_mutate::Float64`: Probability of calling LLM version of mutation.
- `llm_gen_random::Float64`: Probability of calling LLM version of gen_random.

TODO: Implement more prompts so we can make specialized mutation operators like
llm_mutate_const, llm_mutate_operation. 
"""
Base.@kwdef mutable struct LLMMutationProbabilities
    llm_mutate::Float64 = 0.0
    llm_randomize::Float64 = 0.0
end

"""
    LaSRMutationWeights{W<:SymbolicRegression.MutationWeights}(mutation_weights::W, llm_weights::LLMMutationProbabilities)

Defines the composite weights for all the mutation operations in the LaSR module.
"""
mutable struct LaSRMutationWeights{W<:SymbolicRegression.MutationWeights} <:
               SymbolicRegression.AbstractMutationWeights
    sr_weights::W
    llm_weights::LLMMutationProbabilities
end
const LLM_MUTATION_WEIGHTS_KEYS = fieldnames(LLMMutationProbabilities)

@unstable function LaSRMutationWeights(; kws...)
    sr_weights_keys = filter(k -> !(k in LLM_MUTATION_WEIGHTS_KEYS), keys(kws))
    sr_weights = SymbolicRegression.MutationWeights(;
        NamedTuple(sr_weights_keys .=> Tuple(kws[k] for k in sr_weights_keys))...
    )
    sr_weights_vec = [getfield(sr_weights, f) for f in fieldnames(typeof(sr_weights))]

    llm_weights_keys = filter(k -> k in LLM_MUTATION_WEIGHTS_KEYS, keys(kws))
    llm_weights = LLMMutationProbabilities(;
        NamedTuple(llm_weights_keys .=> Tuple(kws[k] for k in llm_weights_keys))...
    )
    llm_weights_vec = [getfield(llm_weights, f) for f in fieldnames(typeof(llm_weights))]

    norm_sr_weights = SymbolicRegression.MutationWeights(
        sr_weights_vec * (1 - sum(llm_weights_vec))...
    )
    norm_llm_weights = LLMMutationProbabilities(llm_weights_vec * sum(sr_weights_vec)...)

    return LaSRMutationWeights(norm_sr_weights, norm_llm_weights)
end

function Base.getproperty(weights::LaSRMutationWeights, k::Symbol)
    if k in LLM_MUTATION_WEIGHTS_KEYS
        return getproperty(getfield(weights, :llm_weights), k)
    else
        return getproperty(getfield(weights, :sr_weights), k)
    end
end

function Base.propertynames(weights::LaSRMutationWeights)
    return (LLM_MUTATION_WEIGHTS_KEYS..., SymbolicRegression.MUTATION_WEIGHTS_KEYS...)
end

end
