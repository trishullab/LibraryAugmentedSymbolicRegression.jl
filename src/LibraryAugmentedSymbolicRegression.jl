module LibraryAugmentedSymbolicRegression

export LaSRPlugin,
    LLMMutateMutation,
    LLMRandomizeMutation,
    LLMGenerateMutation,
    LLMCrossover,
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
    default_prompts_dir,
    prompt_path,
    copy_prompts,
    LLAMAFILE_MODEL,
    LLAMAFILE_PATH,
    LLAMAFILE_URL,
    LLM_PORT,
    AbstractIdeaStore,
    WindowedIdeaStore,
    ScoredIdeaStore,
    add_idea!,
    retrieve_ideas,
    update_idea_value!,
    evolution_candidates,
    NormalizationRule,
    parse_failures,
    parse_failure_summary,
    SuggestionCache,
    CallBudget,
    cache_stats,
    budget_used

using Reexport
using DispatchDoctor: @stable
@reexport using SymbolicRegression

@stable default_mode = "disable" begin
    include("LLMServe.jl")
    include("Logging.jl")
    include("LLMCache.jl")
    include("IdeaStore.jl")
    include("Normalize.jl")
    include("LLMOptionsStruct.jl")
    include("LLMOptions.jl")
    include("Parse.jl")
    include("LLMUtils.jl")
    include("LLMFunctions.jl")
    include("Mutate.jl")
end

using .LoggingModule: LaSRLogger
using .LLMCacheModule:
    SuggestionCache,
    CallBudget,
    cache_stats,
    reset_cache!,
    reset_budget!,
    budget_used
using .IdeaStoreModule:
    AbstractIdeaStore,
    WindowedIdeaStore,
    ScoredIdeaStore,
    add_idea!,
    retrieve_ideas,
    update_idea_value!,
    evolution_candidates
using .LLMOptionsStructModule:
    LaSRPlugin,
    LLMMutateMutation,
    LLMRandomizeMutation,
    LLMGenerateMutation,
    LLMCrossover,
    default_prompts_dir
using .LLMServeModule:
    async_run_llm_server, LLAMAFILE_MODEL, LLAMAFILE_PATH, LLAMAFILE_URL, LLM_PORT
using .LLMFunctionsModule:
    llm_randomize_tree,
    llm_mutate_tree,
    llm_crossover_trees,
    concept_evolution,
    parse_msg_content,
    generate_concepts
using .LLMUtilsModule: load_prompt, construct_prompt, prompt_path, copy_prompts
using .NormalizeModule: NormalizationRule
using .LLMOptionsStructModule: parse_failures, parse_failure_summary
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
