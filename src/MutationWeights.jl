module LaSRMutationWeightsModule

using DispatchDoctor: @unstable
using Base: Base
using StatsBase: StatsBase
using SymbolicRegression: AbstractMutationWeights
import SymbolicRegression: sample_mutation

"""
    LaSRMutationWeights <: AbstractMutationWeights

Defines the composite weights for all the mutation operations in the LaSR module. In addition to the mutation weights
    defined in [`MutationWeights`](@ref SymbolicRegression.CoreModule.MutationWeightsModule.MutationWeights), this struct
    also includes the weights for the LLM mutation operations. These LLM weights can either be manually set or we will
    set them programmatically based on llm_operation_weights. The weights are normalized to sum to 1.0.

# See Also

- [`AbstractMutationWeights`](@ref SymbolicRegression.CoreModule.MutationWeightsModule.AbstractMutationWeights): Use to define custom mutation weight types.
- [`MutationWeights`](@ref SymbolicRegression.CoreModule.MutationWeightsModule.MutationWeights): The default mutation weights for the SR module.
"""
Base.@kwdef mutable struct LaSRMutationWeights <: AbstractMutationWeights
    # Mutation weights imported from SR.jl
    mutate_constant::Float64 = 0.0353
    mutate_operator::Float64 = 3.63
    mutate_feature::Float64 = 0.1
    swap_operands::Float64 = 0.00608
    rotate_tree::Float64 = 1.42
    add_node::Float64 = 0.0771
    insert_node::Float64 = 2.44
    delete_node::Float64 = 0.369
    simplify::Float64 = 0.00148
    randomize::Float64 = 0.00695
    do_nothing::Float64 = 0.431
    optimize::Float64 = 0.0
    form_connection::Float64 = 0.5
    break_connection::Float64 = 0.1

    # Mutation weights specific to LaSR
    # Set programmatically based on the SR.jl mutation weights
    llm_mutate::Float64 = 0.0
    llm_randomize::Float64 = 0.0
end

const lasr_mutations = fieldnames(LaSRMutationWeights)
const v_lasr_mutations = Symbol[lasr_mutations...]

# For some reason it's much faster to write out the fields explicitly:
let contents = [Expr(:., :w, QuoteNode(field)) for field in lasr_mutations]
    @eval begin
        function Base.convert(::Type{Vector}, w::LaSRMutationWeights)::Vector{Float64}
            return $(Expr(:vect, contents...))
        end
        function Base.copy(w::LaSRMutationWeights)
            return $(Expr(:call, :LaSRMutationWeights, contents...))
        end
    end
end

"""
    sample_mutation(w::LaSRMutationWeights)

Sample a mutation operation from the LaSR mutation weights. The weights are normalized to sum to 1.0 before
    sampling.

"""
function sample_mutation(w::P) where {P<:LaSRMutationWeights}
    weights = [getproperty(w, sym) for sym in fieldnames(P)]
    norm_weights = weights ./ sum(weights)
    return StatsBase.sample(v_lasr_mutations, StatsBase.Weights(norm_weights))
end

end
