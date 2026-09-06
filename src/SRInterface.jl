module SRInterfaceModule

using Random: rand
using UUIDs: uuid1
using DynamicExpressions:
    AbstractExpression, get_tree, with_contents, simplify_tree!, combine_operators
using SymbolicRegression
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
using SymbolicRegression.PopMemberModule: PopMember
using SymbolicRegression.ConstantOptimizationModule: optimize_constants
using SymbolicRegression.HallOfFameModule: update_hall_of_fame!
import SymbolicRegression:
    mutate!,
    crossover,
    init_plugin_state,
    fork_plugin_state,
    on_search_start!,
    on_generation_end!,
    refresh_worker_plugin_state,
    plugin_mutations,
    plugin_crossovers
using ..PluginModule:
    LaSRPlugin,
    LaSRPluginState,
    LLMMutateMutation,
    LLMRandomizeMutation,
    LLMGenerateMutation,
    LLMCrossover,
    lasr_context,
    lasr_state
using ..ParseFailuresModule: ParseFailureStore
using ..LLMOperatorsModule:
    llm_mutate_tree, llm_crossover_trees, llm_randomize_tree, llm_generate_candidates
using ..ConceptsModule: generate_concepts
using ..LaSRLoggerModule: LaSRLogger, log_generation!
using ..ExpressionIOModule: render_expr

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
    dataset,
    plugin_states::Tuple,
    curmaxsize::Int,
    nfeatures::Int,
    kws...,
) where {N<:AbstractExpression,P<:AbstractPopMember}
    context = lasr_context(options, lasr_state(options, plugin_states))
    tree = llm_randomize_tree(tree, curmaxsize, context, nfeatures)
    # Fit the freshly generated skeleton's constants before it competes, mirroring
    # LLM-SR's generate-then-optimize step. Without this a generated tree enters the
    # population with unfit (often literal-1) constants, scores poorly, and is culled
    # before its structure can prove out -- which is why a generation-heavy config
    # underperformed. If the optimized candidate is constraint-valid we accept it
    # immediately (generate-and-evaluate); otherwise we hand the tree back to the normal
    # flow so SR's constraint retry/annealing still applies. Constant fitting uses the
    # genuine SR `options` (not the LaSR `context`).
    new_member = PopMember(dataset, tree, options; deterministic=options.deterministic)
    opt_member, num_evals = optimize_constants(dataset, new_member, options)
    if check_constraints(opt_member.tree, options, curmaxsize)
        return MutationResult{N,P}(;
            member=opt_member, num_evals=num_evals, return_immediately=true
        )
    end
    return MutationResult{N,P}(; tree=opt_member.tree, num_evals=num_evals)
end

function mutate!(
    tree::N,
    member::P,
    ::LLMGenerateMutation,
    options::AbstractOptions;
    dataset,
    plugin_states::Tuple,
    curmaxsize::Int,
    nfeatures::Int,
    kws...,
) where {T,L,N<:AbstractExpression,P<:AbstractPopMember{T,L,N}}
    # Ask the LLM for a batch of full-expression skeletons, constant-fit each, and keep the
    # best-scoring constraint-valid candidate. This "generate K, evaluate all, keep best"
    # step lets the search adopt a complex structure wholesale instead of building it up one
    # node at a time -- the primary fix for LaSR under-building complex expressions. When
    # the LLM returns nothing usable we hand the parent tree back unchanged (a no-op
    # mutation). Constant fitting uses the genuine SR `options`, not the LaSR `context`.
    context = lasr_context(options, lasr_state(options, plugin_states))
    candidates = llm_generate_candidates(context, curmaxsize, nfeatures, T)
    best_member = nothing
    total_evals = 0.0
    for cand in candidates
        ex = with_contents(copy(tree), cand)
        new_member = PopMember(dataset, ex, options; deterministic=options.deterministic)
        opt_member, num_evals = optimize_constants(dataset, new_member, options)
        total_evals += num_evals
        if check_constraints(opt_member.tree, options, curmaxsize) &&
            (best_member === nothing || opt_member.loss < best_member.loss)
            best_member = opt_member
        end
    end
    if best_member === nothing
        # Hand the untouched parent back (not a fresh failed candidate) so SR's constraint
        # retry does NOT re-invoke this expensive batched-K LLM call -- the same
        # expensive-op guard `LLMCrossover` applies via its `attempt` check. This is a
        # deliberate asymmetry with `LLMRandomizeMutation`, whose single cheap draw hands its
        # failed candidate back to trigger SR's constraint retry/annealing.
        return MutationResult{N,P}(; tree=tree, num_evals=total_evals)
    end
    return MutationResult{N,P}(;
        member=best_member, num_evals=total_evals, return_immediately=true
    )
end

function init_plugin_state(plugin::LaSRPlugin, options, dataset)
    variable_names = if isnothing(plugin.variable_names)
        Dict(index => name for (index, name) in enumerate(dataset.variable_names))
    else
        copy(plugin.variable_names)
    end
    return LaSRPluginState(
        deepcopy(plugin.idea_store),
        plugin.lasr_logger,
        variable_names,
        0,
        Any[],
        something(plugin.parse_failure_sink, ParseFailureStore()),
    )
end

function on_search_start!(state::LaSRPluginState, ::LaSRPlugin, dataset, options, ropt)
    if !isnothing(ropt.logger)
        state.lasr_logger = LaSRLogger(ropt.logger)
    end
    return nothing
end

function _copy_plugin_state(state::LaSRPluginState)
    # `state.parse_failures` is passed through BY REFERENCE (not deep-copied like every
    # other field above) so that `:serial`/`:multithreading` workers all record into the
    # same lock-guarded `ParseFailureStore` and their fallback counts aggregate into one
    # place instead of being scattered/lost across per-worker copies. `:multiprocessing`
    # workers run in separate address spaces, so this reference-sharing cannot reach them
    # -- per-process aggregation there is a documented follow-up (the `lasr_logger` route
    # already covers cross-process observability in the meantime).
    return LaSRPluginState(
        deepcopy(state.idea_store),
        state.lasr_logger,
        copy(state.variable_names),
        state.generations,
        copy(state.worst_members),
        state.parse_failures,
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

    # Complexity amnesty: re-fit the constants of structurally-rich members before
    # selection can cull them, so good structure is not lost to a bad constant fit.

    if config.amnesty_complexity > 0
        improved = false
        for member in returned_pop.members
            if compute_complexity(member, options) >= config.amnesty_complexity
                optimize_constants(dataset, member, options)
                improved = true
            end
        end
        if improved && !isnothing(search_state)
            output = findfirst(search_state.plugin_states) do states
                any(candidate -> candidate === state, states)
            end
            isnothing(output) || update_hall_of_fame!(
                search_state.halls_of_fame[output], returned_pop.members, options
            )
        end
    end

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

function plugin_mutations(plugin::LaSRPlugin)
    plugin.use_llm || return Pair{SymbolicRegression.AbstractMutation,Float64}[]
    pairs = Pair{SymbolicRegression.AbstractMutation,Float64}[]
    plugin.mutate_weight > 0 && push!(pairs, LLMMutateMutation() => plugin.mutate_weight)
    plugin.randomize_weight > 0 &&
        push!(pairs, LLMRandomizeMutation() => plugin.randomize_weight)
    plugin.generate_weight > 0 &&
        push!(pairs, LLMGenerateMutation() => plugin.generate_weight)
    return pairs
end

# Crossover is now an `AbstractCrossover` sampled by weight from `options.crossovers`
# (parallel to `plugin_mutations`), replacing the old `propose_crossover` hook. The plugin
# injects an `LLMCrossover` weighted by `crossover_probability`; the built-in
# `SubtreeCrossover` carries the remaining weight.
function plugin_crossovers(plugin::LaSRPlugin)
    plugin.use_llm || return Pair{SymbolicRegression.AbstractCrossover,Float64}[]
    plugin.crossover_probability > 0 || return Pair{SymbolicRegression.AbstractCrossover,Float64}[]
    # Make `crossover_probability` a true conditional probability. SR merges crossovers by
    # type, so pinning SubtreeCrossover to (1 - p) overrides its default weight of 1.0.
    # Without this, LLM crossover competes p against a fixed 1.0 and can never exceed 50%
    # (p = 1.0 gave only 0.5), contradicting the parameter's [0, 1] probability contract.
    p = plugin.crossover_probability
    return Pair{SymbolicRegression.AbstractCrossover,Float64}[
        LLMCrossover() => p,
        SymbolicRegression.SubtreeCrossover() => (1 - p),
    ]
end

end
