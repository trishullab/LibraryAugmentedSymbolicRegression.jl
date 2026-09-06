module LLMOperatorsModule

using Random: default_rng, AbstractRNG, rand, randperm
using DispatchDoctor: @unstable
using DynamicExpressions:
    AbstractExpressionNode,
    AbstractExpression,
    get_contents,
    with_contents,
    count_nodes,
    filter_map
using SymbolicRegression: DATA_TYPE, gen_random_tree_fixed_size, AbstractOptions
using SymbolicRegression.MutationFunctionsModule: with_contents_for_mutation
using ..PluginModule: lasr_context
using ..IdeaStoreModule: retrieve_ideas
using ..ExpressionIOModule: render_expr, parse_expr
using ..LaSRLoggerModule: log_generation!
using ..ClientModule: ask, _clean

_is_one_constant(expression) =
    let tree = get_contents(expression)
        tree.constant && tree.val == one(tree.val)
    end

"""
    _first_usable(::Type{T}, candidates, options)

Read the candidates in random order and return the contents of the first one that parses
to something other than the constant-1 fallback. Return `nothing` if no candidate is
usable.
"""
@unstable function _first_usable(::Type{T}, candidates, options) where {T}
    for i in randperm(length(candidates))
        t = parse_expr(T, _clean(candidates[i]), options)
        _is_one_constant(t) || return get_contents(t)
    end
    return nothing
end

function _assumptions(options; query=nothing)
    a = if options.use_concepts
        retrieve_ideas(options.idea_store, options.num_pareto_context; query=query)
    else
        String[]
    end
    options.context == "" || pushfirst!(a, options.context)
    return a
end

function llm_randomize_tree(
    ex::E,
    curmaxsize::Int,
    options::AbstractOptions,
    nfeatures::Int,
    rng::AbstractRNG=default_rng(),
)::E where {E<:AbstractExpression}
    options = lasr_context(options)
    return with_contents_for_mutation(
        ex,
        llm_randomize_tree(get_contents(ex), curmaxsize, options, nfeatures, rng),
        nothing,
    )
end

"""
    llm_randomize_tree(node, curmaxsize, options, nfeatures, rng)

Ask the LLM for a fresh expression and keep the first proposal that parses to something
other than the constant-1 fallback. Fall back to `gen_random_tree_fixed_size` when the
call fails or no proposal is usable.
"""
function llm_randomize_tree(
    ::NT,
    curmaxsize::Int,
    options::AbstractOptions,
    nfeatures::Int,
    rng::AbstractRNG=default_rng(),
)::NT where {T<:DATA_TYPE,NT<:AbstractExpressionNode{T}}
    options = lasr_context(options)
    node_count = rand(rng, 1:curmaxsize)
    candidates, gen_id = ask(
        options,
        "gen_random",
        options.num_generated_equations,
        _assumptions(options) => "assump",
    )
    chosen = isempty(candidates) ? nothing : _first_usable(T, candidates, options)
    if chosen === nothing
        return gen_random_tree_fixed_size(node_count, options, nfeatures, T)
    end
    log_generation!(
        options.lasr_logger;
        id=gen_id,
        mode="gen_random",
        chosen=render_expr(chosen, options),
    )
    return chosen
end

"""
    _is_usable_candidate(tree, nfeatures, curmaxsize)

A generated skeleton is usable if it fits within `curmaxsize` nodes and references at
least one valid input feature (features `1:nfeatures`). Pure-constant proposals (e.g.
`"1.0"`) reference no feature and are rejected.
"""
function _is_usable_candidate(
    tree::AbstractExpressionNode, nfeatures::Int, curmaxsize::Int
)::Bool
    count_nodes(tree) <= curmaxsize || return false
    features = filter_map(
        node -> node.degree == 0 && !node.constant, node -> Int(node.feature), tree, Int
    )
    return !isempty(features) && all(f -> 1 <= f <= nfeatures, features)
end

"""
    llm_generate_candidates(options, curmaxsize, nfeatures, ::Type{T})

Ask the LLM for a batch of full expressions in one call, using the same prompts as
`llm_randomize_tree`, and return every proposal that parses and is usable. A usable
proposal refers to at least one feature and has `curmaxsize` nodes at most.

`llm_randomize_tree` samples one tree. This function keeps every usable candidate instead,
so the caller can fit the constants of each one and keep the best. Return an empty vector
if the call fails, or if no proposal is usable.
"""
function llm_generate_candidates(
    options::AbstractOptions, curmaxsize::Int, nfeatures::Int, ::Type{T}
)::Vector{<:AbstractExpressionNode{T}} where {T<:DATA_TYPE}
    options = lasr_context(options)
    # `use_cache=false`: the pool hands back a single suggestion, but this operator wants
    # the whole batch so it can fit and score each proposal.
    raw, _ = ask(
        options,
        "gen_candidates",
        options.num_generated_equations,
        _assumptions(options) => "assump";
        prompt="gen_random",
        use_cache=false,
    )
    candidates = AbstractExpressionNode{T}[]
    for r in raw
        local tree
        try
            tree = get_contents(parse_expr(T, _clean(r), options))
        catch
            continue
        end
        _is_usable_candidate(tree, nfeatures, curmaxsize) && push!(candidates, tree)
    end
    return candidates
end

function llm_mutate_tree(
    ex::E, options::AbstractOptions
)::E where {T<:DATA_TYPE,E<:AbstractExpression{T}}
    options = lasr_context(options)
    return with_contents(ex, llm_mutate_tree(get_contents(ex), options))
end

"""LLM Mutation on a tree"""
function llm_mutate_tree(
    tree::NT, options::AbstractOptions
)::NT where {T<:DATA_TYPE,NT<:AbstractExpressionNode{T}}
    options = lasr_context(options)
    expr = render_expr(tree, options)
    candidates, gen_id = ask(
        options,
        "mutate",
        options.num_generated_equations,
        _assumptions(options; query=expr) => "assump";
        expr=expr,
    )
    isempty(candidates) && return tree

    chosen = _first_usable(T, candidates, options)
    chosen === nothing && return tree   # fall back to the parent, never a constant
    log_generation!(
        options.lasr_logger; id=gen_id, mode="mutate", chosen=render_expr(chosen, options)
    )
    return chosen
end

function llm_crossover_trees(
    ex1::E, ex2::E, options::AbstractOptions
)::Tuple{E,E} where {T,E<:AbstractExpression{T}}
    options = lasr_context(options)
    tree1, tree2 = llm_crossover_trees(get_contents(ex1), get_contents(ex2), options)
    return with_contents(ex1, tree1), with_contents(ex2, tree2)
end

"""LLM Crossover between two expressions"""
function llm_crossover_trees(
    tree1::NT1, tree2::NT2, options::AbstractOptions
)::Tuple{
    NT1,NT2
} where {T<:DATA_TYPE,NT1<:AbstractExpressionNode{T},NT2<:AbstractExpressionNode{T}}
    options = lasr_context(options)
    expr1 = render_expr(tree1, options)
    expr2 = render_expr(tree2, options)
    candidates, gen_id = ask(
        options,
        "crossover",
        options.num_generated_equations,
        _assumptions(options; query=expr1 * " " * expr2) => "assump";
        expr1=expr1,
        expr2=expr2,
    )
    isempty(candidates) && return tree1, tree2

    # Pick up to two distinct usable candidates in random order. A missing child falls back
    # to a parent, never a constant-1 tree.
    usable = AbstractExpressionNode{T}[]
    for i in randperm(length(candidates))
        t = parse_expr(T, _clean(candidates[i]), options)
        _is_one_constant(t) && continue
        push!(usable, get_contents(t))
        length(usable) == 2 && break
    end

    cross_tree1 = length(usable) >= 1 ? usable[1] : tree1
    cross_tree2 = length(usable) >= 2 ? usable[2] : tree2

    recording_str =
        render_expr(cross_tree1, options) * " && " * render_expr(cross_tree2, options)
    log_generation!(options.lasr_logger; id=gen_id, mode="crossover", chosen=recording_str)

    return cross_tree1, cross_tree2
end

end # module
