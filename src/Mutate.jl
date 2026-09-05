module MutateModule

using Random: rand
using UUIDs: uuid1
using DynamicExpressions:
    AbstractExpression, get_tree, with_contents, simplify_tree!, combine_operators
using SymbolicRegression:
    AbstractOptions,
    AbstractPopMember,
    AbstractCrossover,
    CrossoverResult,
    MutationResult,
    calculate_pareto_frontier,
    check_constraints,
    compute_complexity,
    gen_random_tree_fixed_size
import SymbolicRegression:
    mutate!,
    crossover,
    init_plugin_state,
    fork_plugin_state,
    on_search_start!,
    on_generation_end!,
    refresh_worker_plugin_state
using ..LLMOptionsStructModule:
    LaSRPlugin, LaSRPluginState, LLMMutateMutation, LLMRandomizeMutation, LLMCrossover
using ..LLMOptionsModule: lasr_context, lasr_state
using ..LLMFunctionsModule:
    llm_mutate_tree, llm_crossover_trees, llm_randomize_tree, generate_concepts
using ..LoggingModule: LaSRLogger, log_generation!
using ..ParseModule: render_expr

function mutate!(
    tree::N,
    member::P,
    ::LLMMutateMutation,
    options::AbstractOptions;
    plugin_states::Tuple,
    kws...,
) where {N<:AbstractExpression,P<:AbstractPopMember}
    context = lasr_context(options, lasr_state(options, plugin_states))
    return MutationResult{N,P}(; tree=llm_mutate_tree(tree, context))
end

function mutate!(
    tree::N,
    member::P,
    ::LLMRandomizeMutation,
    options::AbstractOptions;
    plugin_states::Tuple,
    curmaxsize::Int,
    nfeatures::Int,
    kws...,
) where {N<:AbstractExpression,P<:AbstractPopMember}
    context = lasr_context(options, lasr_state(options, plugin_states))
    return MutationResult{N,P}(;
        tree=llm_randomize_tree(tree, curmaxsize, context, nfeatures)
    )
end

function init_plugin_state(plugin::LaSRPlugin, options, dataset)
    variable_names = if isnothing(plugin.variable_names)
        Dict(index => name for (index, name) in enumerate(dataset.variable_names))
    else
        copy(plugin.variable_names)
    end
    return LaSRPluginState(
        copy(plugin.idea_database), plugin.lasr_logger, variable_names, 0, Any[]
    )
end

function on_search_start!(state::LaSRPluginState, ::LaSRPlugin, dataset, options, ropt)
    if !isnothing(ropt.logger)
        state.lasr_logger = LaSRLogger(ropt.logger)
    end
    return nothing
end

function _copy_plugin_state(state::LaSRPluginState)
    return LaSRPluginState(
        copy(state.idea_database),
        state.lasr_logger,
        copy(state.variable_names),
        state.generations,
        copy(state.worst_members),
    )
end

fork_plugin_state(state::LaSRPluginState, ::LaSRPlugin, dataset) = _copy_plugin_state(state)

function refresh_worker_plugin_state(
    worker_state::LaSRPluginState,
    head_state::LaSRPluginState,
    ::LaSRPlugin,
    dataset,
)
    return _copy_plugin_state(head_state)
end

function on_generation_end!(
    state::LaSRPluginState,
    plugin::LaSRPlugin,
    search_state,
    dataset,
    options,
    ropt,
    returned_pop,
)
    config = plugin
    config.use_llm && config.use_concept_evolution || return nothing

    state.generations += 1
    worst = nothing
    for member in returned_pop.members
        (isnothing(worst) || member.loss > worst.loss) && (worst = member)
    end
    !isnothing(worst) && push!(state.worst_members, worst)

    state.generations % options.populations == 0 || return nothing
    output = findfirst(search_state.plugin_states) do states
        any(candidate -> candidate === state, states)
    end
    isnothing(output) && return nothing
    dominating = calculate_pareto_frontier(search_state.halls_of_fame[output])
    if !isempty(dominating)
        filter!(member -> member.loss > last(dominating).loss, state.worst_members)
    end
    generate_concepts(
        dominating, state.worst_members, lasr_context(options, state)
    )
    empty!(state.worst_members)
    return nothing
end

_is_constant(expression) = let tree = get_tree(expression)
    tree.degree == 0 && tree.constant
end

function crossover(
    member1::P,
    member2::P,
    ::LLMCrossover,
    options::AbstractOptions;
    dataset,
    curmaxsize,
    plugin_states::Tuple,
    attempt::Int=1,
    kws...,
) where {T,L,N<:AbstractExpression,P<:AbstractPopMember{T,L,N}}
    # On a constraint-retry the engine re-invokes this; don't burn another LLM call —
    # hand back the parents so it can fall back cheaply (per the expensive-crossover contract).
    if attempt > 1
        return CrossoverResult{N}(; child1=copy(member1.tree), child2=copy(member2.tree))
    end
    state = lasr_state(options, plugin_states)
    context = lasr_context(options, state)
    child1 = combine_operators(
        simplify_tree!(copy(member1.tree), options.operators), options.operators
    )
    child2 = combine_operators(
        simplify_tree!(copy(member2.tree), options.operators), options.operators
    )
    if _is_constant(child1)
        child1 = with_contents(
            child1,
            gen_random_tree_fixed_size(rand(1:curmaxsize), options, dataset.nfeatures, T),
        )
    end
    if _is_constant(child2)
        child2 = with_contents(
            child2,
            gen_random_tree_fixed_size(rand(1:curmaxsize), options, dataset.nfeatures, T),
        )
    end

    child1, child2 = llm_crossover_trees(child1, child2, context)
    child1 = combine_operators(simplify_tree!(child1, options.operators), options.operators)
    child2 = combine_operators(simplify_tree!(child2, options.operators), options.operators)

    generation_id = uuid1()
    rendered = render_expr(child1, context) * " && " * render_expr(child2, context)
    log_generation!(state.lasr_logger; id=generation_id, mode="crossover", chosen=rendered)
    # The engine owns constraint checking, retry, evaluation and replacement.
    return CrossoverResult{N}(; child1=child1, child2=child2)
end

end
