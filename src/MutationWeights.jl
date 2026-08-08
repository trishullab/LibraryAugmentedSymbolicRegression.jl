module LaSRMutationWeightsModule

"""
    LaSRMutationWeights(; kws...)

Compatibility container for LaSR's weighted mutations. New code should set
the LLM weights on [`LaSRPlugin`](@ref); `LaSROptions` still accepts this type.
"""
Base.@kwdef mutable struct LaSRMutationWeights
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
    llm_mutate::Float64 = 0.0
    llm_randomize::Float64 = 0.0
end

function Base.copy(w::LaSRMutationWeights)
    return LaSRMutationWeights(; (name => getfield(w, name) for name in fieldnames(typeof(w)))...)
end

end
