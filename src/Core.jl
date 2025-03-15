module CoreModule

function create_expression end

include("Utils.jl")
include("MutationWeights.jl")
include("Logging.jl")
include("LLMOptionsStruct.jl")
include("LLMOptions.jl")

using .LaSRMutationWeightsModule: LaSRMutationWeights
using .LoggingModule: LaSRLogger
using .LLMOptionsStructModule: LLMOperationWeights, LLMOptions, LaSROptions
using .LLMOptionsModule: LaSROptions, LASR_DEFAULT_OPTIONS

end
