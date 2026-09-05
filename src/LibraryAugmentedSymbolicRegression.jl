module LibraryAugmentedSymbolicRegression

export LaSRPlugin,
    LLMMutateMutation,
    LLMRandomizeMutation,
    LLMGenerateMutation,
    LLMCrossover,
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

using DispatchDoctor: @stable

@stable default_mode = "disable" begin
    # Leaves: no intra-package dependencies.
    include("LaSRLogger.jl")
    include("SuggestionCache.jl")
    include("CallBudget.jl")
    include("IdeaStore.jl")
    include("NormalizationRules.jl")
    include("ParseFailures.jl")

    # The plugin's own types: what a user configures, and the runtime view of it.
    include("Plugin.jl")

    # Text <-> expression.
    include("ExpressionIO.jl")
    include("Prompts.jl")

    # What the LLM does.
    include("Client.jl")
    include("LLMOperators.jl")
    include("Concepts.jl")

    # Every method SymbolicRegression dispatches into LaSR.
    include("SRInterface.jl")
end

using .LaSRLoggerModule: LaSRLogger
using .SuggestionCacheModule: SuggestionCache, cache_stats, reset_cache!
using .CallBudgetModule: CallBudget, budget_used
using .IdeaStoreModule:
    AbstractIdeaStore,
    WindowedIdeaStore,
    ScoredIdeaStore,
    add_idea!,
    retrieve_ideas,
    update_idea_value!,
    evolution_candidates
using .PluginModule:
    LaSRPlugin,
    LLMMutateMutation,
    LLMRandomizeMutation,
    LLMGenerateMutation,
    LLMCrossover,
    default_prompts_dir
using .LLMOperatorsModule: llm_randomize_tree, llm_mutate_tree, llm_crossover_trees
using .ConceptsModule: concept_evolution, generate_concepts
using .ClientModule: parse_msg_content
using .PromptsModule: load_prompt, construct_prompt, prompt_path, copy_prompts
using .NormalizationRulesModule: NormalizationRule
using .PluginModule: parse_failures, parse_failure_summary
using .ExpressionIOModule: render_expr, parse_expr

end
