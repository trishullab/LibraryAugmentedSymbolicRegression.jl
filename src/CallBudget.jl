module CallBudgetModule

using Base.Threads: Atomic, atomic_add!

export CallBudget, claim_call!, budget_used

"""
    CallBudget(limit)

A ceiling on how many LLM calls one search may make.

Without this, LaSR's cost is set only indirectly. `llm_operation_weights` is a
probability applied per *mutation*, so the resulting call count depends on
`populations`, `population_size` and `ncycles_per_iteration` as well -- a user who
raises any of those silently buys more LLM traffic. Measured at the defaults with
`p=0.01`, a search issues roughly 25 calls per iteration, or about a thousand over a
40-iteration run; wall-clock is then set by how fast the endpoint can retire those.

A budget turns that into a number the user picks directly. When it is exhausted the
LLM operators fall back to their symbolic counterparts for the rest of the run,
so the search continues at full speed rather than stalling, and the worst case is
bounded no matter how the other knobs are set. `nothing` restores unbounded behaviour.
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
    # Count first and unconditionally: `used` is also the diagnostic for how much LLM
    # traffic a search actually generated, which is exactly what an unbounded run needs
    # reported. `atomic_add!` returns the previous value, so this reserves and tests in
    # one step and stays correct with populations running on separate threads.
    n = atomic_add!(budget.used, 1)
    budget.limit === nothing && return true
    n < budget.limit && return true
    atomic_add!(budget.denied, 1)
    return false
end

budget_used(budget::CallBudget) = (used=budget.used[], denied=budget.denied[])

end # module
