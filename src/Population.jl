module PopulationModule

__precompile__(false)

using SymbolicRegression:
    DATA_TYPE,
    LOSS_TYPE,
    gen_random_tree,
    AbstractExpression,
    AbstractPopMember,
    PopMember,
    Dataset,
    RecordType,
    Population,
    AbstractOptions
import SymbolicRegression.PopulationModule: record_population

using ..CoreModule: LaSROptions
using ..LLMFunctionsModule: _gen_llm_random_tree
using ..TrackedPopMemberModule: TrackedPopMember

function record_population(pop::Population, options::AbstractOptions)::RecordType
    return RecordType(
        "population" => [
            RecordType(
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
            ) for member in pop.members
        ],
        "time" => time(),
    )
end

end
