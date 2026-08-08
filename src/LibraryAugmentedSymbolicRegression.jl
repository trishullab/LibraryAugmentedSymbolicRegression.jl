module LibraryAugmentedSymbolicRegression

export LaSRPlugin,
    LLMMutateMutation,
    LLMRandomizeMutation,
    LLMCrossover,
    LLMOptions,
    LLMOperationWeights,
    LaSRMutationWeights,
    LaSROptions,
    LaSRRegressor,
    MultitargetLaSRRegressor,
    llm_randomize_tree,
    llm_crossover_trees,
    llm_mutate_tree,
    concept_evolution,
    generate_concepts,
    LaSRLogger,
    render_expr,
    parse_expr,
    parse_msg_content,
    construct_prompt,
    load_prompt,
    LLAMAFILE_MODEL,
    LLAMAFILE_PATH,
    LLAMAFILE_URL,
    LLM_PORT

using Reexport
using DispatchDoctor: @stable
@reexport using SymbolicRegression

@stable default_mode = "disable" begin
    include("Utils.jl")
    include("LLMServe.jl")
    include("MutationWeights.jl")
    include("Logging.jl")
    include("LLMOptionsStruct.jl")
    include("LLMOptions.jl")
    include("Parse.jl")
    include("LLMUtils.jl")
    include("LLMFunctions.jl")
    include("Mutate.jl")
end

using .LaSRMutationWeightsModule: LaSRMutationWeights
using .LoggingModule: LaSRLogger
using .LLMOptionsStructModule:
    LLMOperationWeights,
    LLMOptions,
    LaSRPlugin,
    LLMMutateMutation,
    LLMRandomizeMutation,
    LLMCrossover
using .LLMOptionsModule: LaSROptions
using .LLMServeModule:
    async_run_llm_server, LLAMAFILE_MODEL, LLAMAFILE_PATH, LLAMAFILE_URL, LLM_PORT
using .LLMFunctionsModule:
    llm_randomize_tree,
    llm_mutate_tree,
    llm_crossover_trees,
    concept_evolution,
    parse_msg_content,
    generate_concepts
using .LLMUtilsModule: load_prompt, construct_prompt
using .ParseModule: render_expr, parse_expr

include("MLJInterface.jl")
using .MLJInterfaceModule: LaSRRegressor, MultitargetLaSRRegressor

function __init__()
    should_start_llamafile =
        get(ENV, "START_LLAMASERVER", "false") == "true" ||
        get(ENV, "SYMBOLIC_REGRESSION_TEST_SUITE", "") == "online_llamafile"
    should_start_llamafile && async_run_llm_server(LLAMAFILE_URL, LLAMAFILE_PATH, LLM_PORT)
    return nothing
end

end
