module HallOfFameModule

__precompile__(false)

using DispatchDoctor: @unstable
using SymbolicRegression:
    DATA_TYPE,
    LOSS_TYPE,
    PopMember,
    Dataset,
    create_expression,
    AbstractVector,
    calculate_pareto_frontier,
    Population,
    init_value
using SymbolicRegression
using SymbolicRegression.LoggingModule: pareto_volume, string_tree, compute_complexity
import SymbolicRegression.HallOfFameModule: HallOfFame, format_hall_of_fame

using ..CoreModule: LaSROptions
using ..TrackedPopMemberModule: TrackedPopMember

function _log_scalars(;
    @nospecialize(pops::AbstractVector{<:Population}),
    @nospecialize(hall_of_fame::HallOfFame{T,L}),
    dataset::Dataset{T,L},
    options::LaSROptions,
) where {T,L}
    out = Dict{String,Any}()

    #### Population diagnostics
    out["population"] = Dict([
        "complexities" => let
            complexities = Int[]
            for pop in pops, member in pop.members
                push!(complexities, compute_complexity(member, options))
            end
            complexities
        end,
        "llm_usages" => let
            llm_usages = Float64[]
            for pop in pops, member in pop.members
                llm_contribution = get(member, :llm_contribution, 0.0)
                total_contribution = get(member, :total_contribution, 1.0)
                push!(llm_usages, llm_contribution / total_contribution)
            end
            llm_usages
        end,
    ])

    #### Summaries
    dominating = calculate_pareto_frontier(hall_of_fame)
    trees = [member.tree for member in dominating]
    losses = L[member.loss for member in dominating]
    complexities = Int[compute_complexity(member, options) for member in dominating]

    out["summaries"] = Dict([
        "min_loss" => length(dominating) > 0 ? dominating[end].loss : L(Inf),
        "pareto_volume" => pareto_volume(
            losses, complexities, options.maxsize, options.loss_scale == :linear
        ),
        "llm_usage" => if length(dominating) > 0
            get(dominating[end], :llm_contribution, 0.0) /
            get(dominating[end], :total_contribution, 1.0) # TODO: check if this makes sense.
        else
            L(0)
        end,
    ])

    #### Full Pareto front
    out["equations"] = let
        equations = String[
            string_tree(member.tree, options; variable_names=dataset.variable_names) for
            member in dominating
        ]
        Dict([
            "complexity=" * string(complexities[i_eqn]) =>
                Dict("loss" => losses[i_eqn], "equation" => equations[i_eqn]) for
            i_eqn in eachindex(complexities, losses, equations)
        ])
    end
    return out
end

end
