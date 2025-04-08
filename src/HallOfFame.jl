module HallOfFameModule

using SymbolicRegression: DATA_TYPE, LOSS_TYPE, PopMember, Dataset, create_expression
import SymbolicRegression.HallOfFameModule: HallOfFame

using ..CoreModule: LaSROptions
using ..TrackedPopMemberModule: TrackedPopMember

function HallOfFame(
    options::LaSROptions, dataset::Dataset{T,L}
) where {T<:DATA_TYPE,L<:LOSS_TYPE}
    base_tree = create_expression(zero(T), options, dataset)
    return HallOfFame{T,L,typeof(base_tree)}(
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

end
