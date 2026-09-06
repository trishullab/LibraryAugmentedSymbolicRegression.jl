module ConceptsModule

using Random: rand, shuffle
using DispatchDoctor: @unstable
using SymbolicRegression: AbstractOptions
using ..PluginModule: lasr_context
using ..PromptsModule: format_pareto
using ..IdeaStoreModule: add_idea!, evolution_candidates
using ..LaSRLoggerModule: log_generation!
using ..ClientModule: ask, _clean

@unstable function concept_evolution(options::AbstractOptions)
    options = lasr_context(options)
    candidates = evolution_candidates(options.idea_store)
    isempty(candidates) && return nothing

    idea_options, gen_id = ask(
        options,
        "concept_evolution",
        options.num_generated_concepts,
        shuffle(candidates) => "idea",
    )
    isempty(idea_options) && return nothing

    N = min(length(idea_options), options.num_generated_concepts)
    chosen_idea = _clean(idea_options[rand(1:N)])
    log_generation!(
        options.lasr_logger; id=gen_id, mode="concept_evolution", chosen=chosen_idea
    )
    return chosen_idea
end

function generate_concepts(dominating, worst_members, options::AbstractOptions)
    options = lasr_context(options)
    isnothing(dominating) && return nothing

    n_context = options.num_pareto_context
    idea_options, gen_id = ask(
        options,
        "generate_concepts",
        options.num_generated_concepts,
        format_pareto(dominating, options, n_context) => "gexpr",
        format_pareto(worst_members, options, n_context) => "bexpr",
    )
    isempty(idea_options) && return nothing

    N = min(length(idea_options), options.num_generated_concepts)
    for _ in 1:(options.num_concept_crossover)
        chosen_idea = _clean(idea_options[rand(1:N)])
        log_generation!(
            options.lasr_logger; id=gen_id, mode="generate_concepts", chosen=chosen_idea
        )
        add_idea!(options.idea_store, chosen_idea)
    end

    for _ in 1:(options.num_concept_crossover)
        out = concept_evolution(options)
        isnothing(out) || add_idea!(options.idea_store, out; refined=true)
    end
    return nothing
end

end # module
