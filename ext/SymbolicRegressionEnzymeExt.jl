module SymbolicRegressionEnzymeExt

using SymbolicRegression.LossFunctionsModule: eval_loss
using DynamicExpressions:
    AbstractExpression,
    AbstractExpressionNode,
    get_constants,
    set_constants!,
    extract_gradient,
    with_contents,
    get_contents
using ADTypes: AutoEnzyme
using Enzyme: autodiff, Reverse, Active, Const, Duplicated

import SymbolicRegression.ConstantOptimizationModule: GradEvaluator

# We prepare a copy of the tree and all arrays
function GradEvaluator(f::F, backend::AE) where {F,AE<:AutoEnzyme}
    storage_tree = copy(f.tree)
    _, storage_refs = get_constants(storage_tree)
    return GradEvaluator(f, backend, (; storage_tree, storage_refs))
end

function evaluator(tree, dataset, options, idx, output)
    output[] = eval_loss(tree, dataset, options; regularization=false, idx=idx)
    return nothing
end

function (g::GradEvaluator{<:Any,<:AutoEnzyme})(_, G, x::AbstractVector{T}) where {T}
    set_constants!(g.f.tree, x, g.f.refs)
    set_constants!(g.extra.storage_tree, zero(x), g.extra.storage_refs)

    output = fill(zero(T), 1)
    doutput = fill(one(T), 1)

    autodiff(
        Reverse,
        evaluator,
        Duplicated(g.f.tree, g.extra.storage_tree),
        Const(g.f.dataset),
        Const(g.f.options),
        Const(g.f.idx),
        Duplicated(output, doutput),
    )
    # ^ The seed value of 1.0 propagates from `doutput` to `g.extra.storage_tree`

    if G !== nothing
        # TODO: This is redundant since we already have the references.
        # Should just be able to extract from the references directly.
        G .= first(get_constants(g.extra.storage_tree))
    end
    return output[]
end

end