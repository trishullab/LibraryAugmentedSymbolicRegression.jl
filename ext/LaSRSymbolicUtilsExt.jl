module LaSRSymbolicUtilsExt

using SymbolicUtils: Symbolic
using LibraryAugmentedSymbolicRegression:
    AbstractExpressionNode, AbstractExpression, Node, Options
using LibraryAugmentedSymbolicRegression.MLJInterfaceModule:
    AbstractSRRegressor, get_options
using DynamicExpressions: get_tree, get_operators

import LibraryAugmentedSymbolicRegression: node_to_symbolic, symbolic_to_node

"""
    node_to_symbolic(tree::AbstractExpressionNode, options::Options; kws...)

Convert an expression to SymbolicUtils.jl form.
"""
function node_to_symbolic(
    tree::Union{AbstractExpressionNode,AbstractExpression}, options::Options; kws...
)
    return node_to_symbolic(get_tree(tree), get_operators(tree, options); kws...)
end
function node_to_symbolic(
    tree::Union{AbstractExpressionNode,AbstractExpression}, m::AbstractSRRegressor; kws...
)
    return node_to_symbolic(tree, get_options(m); kws...)
end

"""
    symbolic_to_node(eqn::Symbolic, options::Options; kws...)

Convert a SymbolicUtils.jl expression to LibraryAugmentedSymbolicRegression.jl's `Node` type.
"""
function symbolic_to_node(eqn::Symbolic, options::Options; kws...)
    return symbolic_to_node(eqn, options.operators; kws...)
end
function symbolic_to_node(eqn::Symbolic, m::AbstractSRRegressor; kws...)
    return symbolic_to_node(eqn, get_options(m); kws...)
end

function Base.convert(
    ::Type{Symbolic},
    tree::Union{AbstractExpressionNode,AbstractExpression},
    options::Union{Options,Nothing}=nothing;
    kws...,
)
    return convert(Symbolic, get_tree(tree), get_operators(tree, options); kws...)
end
function Base.convert(
    ::Type{Symbolic},
    tree::Union{AbstractExpressionNode,AbstractExpression},
    m::AbstractSRRegressor;
    kws...,
)
    return convert(Symbolic, tree, get_options(m); kws...)
end

function Base.convert(
    ::Type{N}, x::Union{Number,Symbolic}, options::Options; kws...
) where {N<:Union{AbstractExpressionNode,AbstractExpression}}
    return convert(N, x, options.operators; kws...)
end
function Base.convert(
    ::Type{N}, x::Union{Number,Symbolic}, m::AbstractSRRegressor; kws...
) where {N<:Union{AbstractExpressionNode,AbstractExpression}}
    return convert(N, x, get_options(m); kws...)
end

end