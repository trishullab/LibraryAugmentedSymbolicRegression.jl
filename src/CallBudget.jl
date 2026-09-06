module CallBudgetModule

using Base.Threads: Atomic, atomic_add!

export CallBudget, claim_call!, budget_used

"""
    CallBudget(limit)

A ceiling on how many LLM calls one search may make. When it is exhausted the LLM operators fall back to their symbolic counterparts for the rest of the run, so the search continues at full speed rather than stalling.
"""
struct CallBudget
    limit::Union{Int,Nothing}
    used::Atomic{Int}
    denied::Atomic{Int}
end

function CallBudget(limit::Union{Int,Nothing}=nothing)
    return CallBudget(limit, Atomic{Int}(0), Atomic{Int}(0))
end

"""
    claim_call!(budget) -> Bool

Reserve one call. Returns `false` once the run's allowance is spent.
"""
function claim_call!(budget::CallBudget)
    n = atomic_add!(budget.used, 1)
    budget.limit === nothing && return true
    n < budget.limit && return true
    atomic_add!(budget.denied, 1)
    return false
end

budget_used(budget::CallBudget) = (used=budget.used[], denied=budget.denied[])

end # module
