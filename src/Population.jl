module PopulationModule

using SymbolicRegression:
    DATA_TYPE,
    LOSS_TYPE,
    gen_random_tree,
    AbstractExpression,
    AbstractPopMember,
    PopMember,
    Dataset,
    RecordType,
    AbstractOptions
import SymbolicRegression.PopulationModule: Population, generate_record

using ..CoreModule: LaSROptions
using ..LLMFunctionsModule: _gen_llm_random_tree
using ..TrackedPopMemberModule: TrackedPopMember

function Population(
    dataset::Dataset{T,L};
    options::LaSROptions,
    population_size=nothing,
    nlength::Int=3,
    nfeatures::Int,
    npop=nothing,
) where {T,L}
    @assert (population_size !== nothing) ⊻ (npop !== nothing)
    population_size = if npop === nothing
        population_size
    else
        npop
    end

    generations = []
    for _ in 1:population_size
        if options.use_llm && (rand() < options.llm_operation_weights.llm_randomize)
            tree = _gen_llm_random_tree(nlength, options, nfeatures, T)
            llm_used = true
        else
            tree = gen_random_tree(nlength, options, nfeatures, T)
            llm_used = false
        end
        push!(generations, (tree, llm_used))
    end
    return Population(
        [
            if options.tracking
                TrackedPopMember(
                    PopMember(
                        dataset,
                        tree,
                        options;
                        parent=-1,
                        deterministic=options.deterministic,
                    ),
                    convert(Float64, llm_used),
                    convert(Float64, !llm_used),
                    1.0,
                )
            else
                PopMember(
                    dataset, tree, options; parent=-1, deterministic=options.deterministic
                )
            end for (tree, llm_used) in generations
        ],
        population_size,
    )
end

function generate_record(member::TrackedPopMember, options::AbstractOptions)::RecordType
    return RecordType(
        "tree" => string_tree(member.tree, options; pretty=false),
        "loss" => member.loss,
        "cost" => member.cost,
        "complexity" => compute_complexity(member, options),
        "birth" => member.birth,
        "ref" => member.ref,
        "parent" => member.parent,
        "llm_contribution" => member.llm_contribution,
        "sr_contribution" => member.sr_contribution,
        "total_contribution" => member.total_contribution,
    )
end

end
