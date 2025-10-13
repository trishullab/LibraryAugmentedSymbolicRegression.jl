module TrackedPopMemberModule
using Base
using DispatchDoctor: @unstable
using DynamicExpressions: DynamicExpressions
using DynamicExpressions: AbstractExpression, string_tree
using SymbolicRegression
import SymbolicRegression: AbstractPopMember
import SymbolicRegression.PopMemberModule: create_child
using ..LaSRMutationWeightsModule: v_lasr_mutations

const LLM_OPS = filter(x -> startswith(String(x), "llm_"), v_lasr_mutations)
@inline is_llm_op(x) = (x !== nothing) && (x in LLM_OPS)

mutable struct TrackedPopMember{T,L,N} <: AbstractPopMember{T,L,N}
    tree::N
    cost::L
    loss::L
    birth::Int
    complexity::Int
    ref::Int
    parent::Int
    llm_contribution::Int  # Custom fields to track LLM calls
    sr_contribution::Int
    total_contribution::Int
end

# Direct constructor that matches field order
function TrackedPopMember(
    tree::N,
    cost::L,
    loss::L,
    birth::Int,
    complexity::Int,
    ref::Int,
    parent::Int,
    llm_contribution::Int,
    sr_contribution::Int,
    total_contribution::Int,
) where {T,L,N<:DynamicExpressions.AbstractExpression{T}}
    return TrackedPopMember{T,L,N}(
        tree,
        cost,
        loss,
        birth,
        complexity,
        ref,
        parent,
        llm_contribution,
        sr_contribution,
        total_contribution,
    )
end

function TrackedPopMember(
    tree::N, cost::L, loss::L, options, complexity::Int; parent=-1, deterministic=nothing
) where {T,L,N<:DynamicExpressions.AbstractExpression{T}}
    return TrackedPopMember(
        tree,
        cost,
        loss,
        SymbolicRegression.get_birth_order(; deterministic=deterministic),
        complexity,
        abs(rand(Int)),
        parent,
        0,
        0,
        0,
    )
end

# Constructor for Population initialization (dataset, tree, options)
function TrackedPopMember(
    dataset::SymbolicRegression.Dataset, tree, options; parent=-1, deterministic=nothing
)
    ex = SymbolicRegression.create_expression(tree, options, dataset)
    complexity = SymbolicRegression.compute_complexity(ex, options)
    cost, loss = SymbolicRegression.eval_cost(dataset, ex, options; complexity=complexity)

    return TrackedPopMember(
        ex,
        cost,
        loss,
        SymbolicRegression.get_birth_order(; deterministic=deterministic),
        complexity,
        abs(rand(Int)),
        parent,
        0,
        0,
        0,
    )
end

@unstable DynamicExpressions.constructorof(::Type{<:TrackedPopMember}) = TrackedPopMember

# Define with_type_parameters for TrackedPopMember
@unstable function DynamicExpressions.with_type_parameters(
    ::Type{<:TrackedPopMember}, ::Type{T}, ::Type{L}, ::Type{N}
) where {T,L,N}
    return TrackedPopMember{T,L,N}
end

# Define copy for TrackedPopMember
function Base.copy(p::TrackedPopMember)
    return TrackedPopMember(
        copy(p.tree),
        copy(p.cost),
        copy(p.loss),
        copy(p.birth),
        copy(getfield(p, :complexity)),
        copy(p.ref),
        copy(p.parent),
        copy(p.llm_contribution),
        copy(p.sr_contribution),
        copy(p.total_contribution),
    )
end

function create_child(
    parent::TrackedPopMember{T,L},
    tree::DynamicExpressions.AbstractExpression{T},
    cost::L,
    loss::L,
    options;
    complexity::Union{Int,Nothing}=nothing,
    mutation_choice::Union{Symbol,Nothing}=nothing,
    parent_ref,
) where {T,L}
    actual_complexity = @something complexity SymbolicRegression.compute_complexity(
        tree, options
    )
    llm_contribution = parent.llm_contribution
    sr_contribution = parent.sr_contribution
    total_contribution = parent.total_contribution + 1
    if is_llm_op(mutation_choice)
        llm_contribution += 1
    else
        sr_contribution += 1
    end
    return TrackedPopMember(
        tree,
        cost,
        loss,
        SymbolicRegression.get_birth_order(; deterministic=options.deterministic),
        actual_complexity,
        abs(rand(Int)),
        parent_ref,
        llm_contribution,
        sr_contribution,
        total_contribution,
    )
end

function create_child(
    parents::Tuple{<:TrackedPopMember,<:TrackedPopMember},
    tree::N,
    cost::L,
    loss::L,
    options;
    complexity::Union{Int,Nothing}=nothing,
    mutation_choice::Union{Symbol,Nothing}=nothing,
    parent_ref,
) where {T,L,N<:DynamicExpressions.AbstractExpression{T}}
    actual_complexity = @something complexity SymbolicRegression.compute_complexity(
        tree, options
    )
    llm_contribution = parents[1].llm_contribution + parents[2].llm_contribution
    sr_contribution = parents[1].sr_contribution + parents[2].sr_contribution
    if is_llm_op(mutation_choice)
        llm_contribution += 1
    else
        sr_contribution += 1
    end
    total_contribution = llm_contribution + sr_contribution

    return TrackedPopMember(
        tree,
        cost,
        loss,
        SymbolicRegression.CoreModule.UtilsModule.get_birth_order(;
            deterministic=options.deterministic
        ),
        actual_complexity,
        abs(rand(Int)),
        parent_ref,
        llm_contribution,
        sr_contribution,
        total_contribution,
    )
end

end
