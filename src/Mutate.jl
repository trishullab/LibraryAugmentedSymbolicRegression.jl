module MutateModule

using Random: rand
using DynamicExpressions:
    AbstractExpression, get_tree, with_contents, simplify_tree!, combine_operators
using SymbolicRegression:
    AbstractOptions,
    AbstractPopMember,
    CrossoverResult,
    MutationResult,
    calculate_pareto_frontier,
    gen_random_tree_fixed_size
import SymbolicRegression:
    crossover,
    mutate!,
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
using ..LoggingModule: LaSRLogger

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
    worker_state::LaSRPluginState, head_state::LaSRPluginState, ::LaSRPlugin, dataset
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
    generate_concepts(dominating, state.worst_members, lasr_context(options, state))
    empty!(state.worst_members)
    return nothing
end

_is_constant(expression) =
    let tree = get_tree(expression)
        tree.degree == 0 && tree.constant
    end

function crossover(
    member1::P,
    member2::P,
    ::LLMCrossover,
    options::AbstractOptions;
    plugin_states::Tuple,
    dataset,
    curmaxsize::Int,
    attempt::Int,
    kws...,
) where {T,N<:AbstractExpression{T},P<:AbstractPopMember{T,<:Any,N}}
    parent1, parent2 = member1.tree, member2.tree
    # SymbolicRegression retries crossovers which violate constraints. Do not
    # repeat an expensive model call on those retries.
    if attempt > 1
        return CrossoverResult{N}(; child1=copy(parent1), child2=copy(parent2))
    end

    state = lasr_state(options, plugin_states)
    context = lasr_context(options, state)
    context.use_llm ||
        return CrossoverResult{N}(; child1=copy(parent1), child2=copy(parent2))

    child1 = combine_operators(
        simplify_tree!(copy(parent1), options.operators), options.operators
    )
    child2 = combine_operators(
        simplify_tree!(copy(parent2), options.operators), options.operators
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
    if _is_constant(child1) || _is_constant(child2)
        return CrossoverResult{N}(; child1=copy(parent1), child2=copy(parent2))
    end
    return CrossoverResult{N}(; child1, child2)
end

end
