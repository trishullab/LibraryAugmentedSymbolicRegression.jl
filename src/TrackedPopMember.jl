module TrackedPopMemberModule

using Base
using DispatchDoctor: @unstable
using DynamicExpressions: AbstractExpression, string_tree
using SymbolicRegression: AbstractOptions, DATA_TYPE, LOSS_TYPE, Dataset
import SymbolicRegression.PopMemberModule: AbstractPopMember, PopMember, compute_complexity
import SymbolicRegression.ExpressionBuilderModule: strip_metadata, embed_metadata

# TrackedPopMember is a PopMember with additional tracking for LLM_contribution, SR_contribution, total_contribution
mutable struct TrackedPopMember{T<:DATA_TYPE,L<:LOSS_TYPE,N<:AbstractExpression{T}} <:
               AbstractPopMember{T,L,N}
    pm::PopMember{T,L,N}
    llm_contribution::Float64
    sr_contribution::Float64
    total_contribution::Float64
end
const TRACKED_KEYS = [:llm_contribution, :sr_contribution, :total_contribution]

@unstable @inline function Base.getproperty(member::TrackedPopMember, field::Symbol)
    if field == :pm
        return getfield(member, :pm)
    elseif field in TRACKED_KEYS
        return getfield(member, field)
    else
        return getproperty(getfield(member, :pm), field)
    end
end

@inline function Base.setproperty!(member::TrackedPopMember, field::Symbol, value)
    if field == :pm
        return setfield!(member, :pm, value)
    elseif field in TRACKED_KEYS
        return setfield!(member, field, value)
    else
        return setproperty!(getfield(member, :pm), field, value)
    end
end

function Base.show(
    io::IO, p::TrackedPopMember{T,L,N}
) where {T<:DATA_TYPE,L<:LOSS_TYPE,N<:AbstractExpression{T}}
    shower(x) = sprint(show, x)
    print(io, "PopMember(")
    print(io, "tree = (", string_tree(p.tree), "), ")
    print(io, "loss = ", shower(p.loss), ", ")
    print(io, "cost = ", shower(p.cost))
    for k in TRACKED_KEYS
        print(io, ", ", k, " = ", shower(getproperty(p, k)))
    end
    print(io, ")")
    return nothing
end

function TrackedPopMember(;
    llm_contribution::Float64=0.0,
    sr_contribution::Float64=0.0,
    total_contribution::Float64=0.0,
    kws...,
)
    # Create a new TrackedPopMember with the given contributions
    pm = PopMember(kws...)
    return TrackedPopMember(pm, llm_contribution, sr_contribution, total_contribution)
end

function Base.copy(p::TrackedPopMember)
    pm = copy(p.pm)
    llm_contribution = copy(p.llm_contribution)
    sr_contribution = copy(p.sr_contribution)
    total_contribution = copy(p.total_contribution)
    return TrackedPopMember(pm, llm_contribution, sr_contribution, total_contribution)
end

function strip_metadata(
    tracked_member::TrackedPopMember, options::AbstractOptions, dataset::Dataset{T,L}
) where {T,L}
    member = tracked_member.pm
    new_tm = copy(tracked_member)
    new_tm.pm = PopMember(
        strip_metadata(member.tree, options, dataset),
        member.cost,
        member.loss,
        nothing;
        member.ref,
        member.parent,
        deterministic=options.deterministic,
    )
    return new_tm
end

@unstable begin
    function embed_metadata(
        tracked_member::TrackedPopMember, options::AbstractOptions, dataset::Dataset{T,L}
    ) where {T,L}
    return TrackedPopMember(
        PopMember(
            embed_metadata(tracked_member.pm.tree, options, dataset),
            tracked_member.pm.cost,
            tracked_member.pm.loss,
            nothing;
            tracked_member.pm.ref,
            tracked_member.pm.parent,
            deterministic=options.deterministic,
        ),
        tracked_member.llm_contribution,
        tracked_member.sr_contribution,
        tracked_member.total_contribution,
    )
    end
end

function compute_complexity(
    member::TrackedPopMember, options::AbstractOptions; break_sharing=Val(false)
)::Int
    return compute_complexity(member.pm, options; break_sharing)
end

function recompute_complexity!(
    member::TrackedPopMember, options::AbstractOptions; break_sharing=Val(false)
)::Int
    return recompute_complexity!(member.pm, options; break_sharing)
end

end
