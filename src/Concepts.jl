module ConceptsModule

using Random: rand, shuffle
using DispatchDoctor: @unstable
using SymbolicRegression: AbstractOptions
using ..PluginModule: lasr_context
using ..PromptsModule:
    load_prompt, prompt_path, get_vars, get_ops, construct_prompt, format_pareto
using ..IdeaStoreModule: add_idea!, evolution_candidates
using ..LaSRLoggerModule: log_generation!
using ..ClientModule: request_suggestions, _clean, render_conversation
using PromptingTools: SystemMessage, UserMessage

@unstable function concept_evolution(options::AbstractOptions)
    options = lasr_context(options)
    # The store decides which ideas are eligible for distillation: for the windowed store
    # this is the overflow beyond its sampling window; for the scored store it is the
    # below-median-value ideas.
    candidates = evolution_candidates(options.idea_store)
    isempty(candidates) && return nothing

    ideas = shuffle(candidates)
    conversation = [
        SystemMessage(
            load_prompt(prompt_path(options.prompts_dir, "concept_evolution_system.prompt"))
        ),
        UserMessage(
            construct_prompt(
                load_prompt(
                    prompt_path(options.prompts_dir, "concept_evolution_user.prompt")
                ),
                ideas,
                "idea",
            ),
        ),
    ]

    rendered_msg = render_conversation(
        conversation, options; N=options.num_generated_concepts
    )

    idea_options, gen_id = request_suggestions(
        options,
        "concept_evolution",
        conversation,
        options.num_generated_concepts;
        rendered_msg=rendered_msg,
    )
    if isempty(idea_options)
        return nothing
    end

    N = min(size(idea_options)[1], options.num_generated_concepts)

    chosen_idea = _clean(idea_options[rand(1:N)])

    log_generation!(
        options.lasr_logger; id=gen_id, mode="concept_evolution", chosen=chosen_idea
    )

    return chosen_idea
end

function generate_concepts(dominating, worst_members, options::AbstractOptions)
    options = lasr_context(options)
    # turn dominating pareto curve into ideas as strings
    if isnothing(dominating)
        return nothing
    end

    gexpr = format_pareto(dominating, options, options.num_pareto_context)
    bexpr = format_pareto(worst_members, options, options.num_pareto_context)

    conversation = [
        SystemMessage(
            load_prompt(prompt_path(options.prompts_dir, "generate_concepts_system.prompt"))
        ),
        UserMessage(
            construct_prompt(
                construct_prompt(
                    load_prompt(
                        prompt_path(options.prompts_dir, "generate_concepts_user.prompt")
                    ),
                    gexpr,
                    "gexpr",
                ),
                bexpr,
                "bexpr",
            ),
        ),
    ]
    rendered_msg = render_conversation(
        conversation, options; N=options.num_generated_concepts
    )

    idea_options, gen_id = request_suggestions(
        options,
        "generate_concepts",
        conversation,
        options.num_generated_concepts;
        rendered_msg=rendered_msg,
        variables=get_vars(options),
        operators=get_ops(options),
        no_system_message=false,
        verbose=options.verbose,
    )
    if isempty(idea_options)
        return nothing
    end

    N = min(size(idea_options)[1], options.num_generated_concepts)

    for _ in 1:(options.num_concept_crossover)
        a = rand(1:N)
        chosen_idea = _clean(idea_options[a])
        log_generation!(
            options.lasr_logger; id=gen_id, mode="generate_concepts", chosen=chosen_idea
        )
        add_idea!(options.idea_store, chosen_idea)
    end

    for _ in 1:(options.num_concept_crossover)
        out = concept_evolution(options)
        if !isnothing(out)
            add_idea!(options.idea_store, out; refined=true)
        end
    end
end

end # module
