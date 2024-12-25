module CoreModule

function create_expression end

include("Utils.jl")
include("MutationWeights.jl")
include("LLMOptions.jl")

using .LaSRMutationWeightsModule: LLMMutationProbabilities, LaSRMutationWeights
using .LLMOptionsModule:
    LLMOperationWeights,
    LLMOptions,
    LaSROptions

end
