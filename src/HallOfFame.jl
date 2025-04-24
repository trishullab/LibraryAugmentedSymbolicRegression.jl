module HallOfFameModule

using DispatchDoctor: @unstable
using SymbolicRegression: DATA_TYPE, LOSS_TYPE, PopMember, Dataset, create_expression, AbstractVector, calculate_pareto_frontier, Population
using SymbolicRegression
using SymbolicRegression.LoggingModule: pareto_volume, string_tree, compute_complexity
import SymbolicRegression.HallOfFameModule: HallOfFame, format_hall_of_fame
import SymbolicRegression.LoggingModule: _log_scalars

using ..CoreModule: LaSROptions
using ..TrackedPopMemberModule: TrackedPopMember

function HallOfFame(
    options::LaSROptions, dataset::Dataset{T,L}
) where {T<:DATA_TYPE,L<:LOSS_TYPE}
    base_tree = create_expression(zero(T), options, dataset)
    PM = if options.tracking
        TrackedPopMember
    else
        PopMember
    end
    return HallOfFame{T,L,typeof(base_tree),PM{T,L,typeof(base_tree)}}(
        [
            if options.tracking
                TrackedPopMember(
                    PopMember(
                        dataset,
                        base_tree,
                        options;
                        parent=-1,
                        deterministic=options.deterministic,
                    ),
                    0.0,
                    0.0,
                    0.0,
                )
            else
                PopMember(
                    dataset,
                    base_tree,
                    options;
                    parent=-1,
                    deterministic=options.deterministic,
                )
            end for _ in 1:(options.maxsize)
        ],
        [false for i in 1:(options.maxsize)],
    )
end

@unstable function format_hall_of_fame(
    hof::HallOfFame{T,L,TreeType,PM}, options::LaSROptions
) where {T<:DATA_TYPE,L<:LOSS_TYPE,TreeType,PM}
    default_columns = [:losses, :complexities, :scores, :trees]
    if options.tracking
        columns = [
            default_columns...
            [:llm_contribution, :sr_contribution, :total_contribution]...
        ]
    end
    all_hall_of_fame = format_hall_of_fame(hof, options; columns=columns)

    if options.tracking && !isempty(all_hall_of_fame)
        # remove the llm_contribution, sr_contribution, and total_contribution columns
        # add a new column called llm_usage = llm_contribution / total_contribution
        default_dict = Dict(k => all_hall_of_fame[k] for k in default_columns)
        new_dict = Dict(
            :llm_usage =>
                all_hall_of_fame.llm_contribution ./ all_hall_of_fame.total_contribution,
            :sr_usage =>
                all_hall_of_fame.sr_contribution ./ all_hall_of_fame.total_contribution,
        )

        merged_dict = merge(default_dict, new_dict)
        all_hall_of_fame = NamedTuple(merged_dict)
    end
    return all_hall_of_fame
end

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
                push!(llm_usages, member.llm_contribution / member.total_contribution)
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
        "pareto_volume" => pareto_volume(losses, complexities, options.maxsize),
        "llm_usage" => length(dominating) > 0 ? dominating[end].llm_contribution / dominating[end].total_contribution : L(0),
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
