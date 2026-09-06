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
using Compat: Returns
using SymbolicRegression: DATA_TYPE, gen_random_tree_fixed_size, AbstractOptions
using SymbolicRegression.MutationFunctionsModule: with_contents_for_mutation
using ..PluginModule: lasr_context
using ..PromptsModule: load_prompt, prompt_path, get_vars, get_ops, construct_prompt
using ..IdeaStoreModule: retrieve_ideas
using ..ExpressionIOModule: render_expr, parse_expr
using ..LaSRLoggerModule: log_generation!
using ..ClientModule: request_suggestions, _clean, render_conversation
using PromptingTools: SystemMessage, UserMessage

_is_one_constant(expression) =
    let tree = get_contents(expression)
        tree.constant && tree.val == one(tree.val)
    end

# Return the first parsed candidate that is not the constant-1 fallback, in random order,
# or `nothing` if every candidate is unusable.
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
    tree = get_contents(ex)
    context = nothing
    ex = with_contents_for_mutation(
        ex, llm_randomize_tree(tree, curmaxsize, options, nfeatures, rng), context
    )
    return ex
end

function llm_randomize_tree(
    ::NT,
    curmaxsize::Int,
    options::AbstractOptions,
    nfeatures::Int,
    rng::AbstractRNG=default_rng(),
)::NT where {T<:DATA_TYPE,NT<:AbstractExpressionNode{T}}
    options = lasr_context(options)
    tree_size_to_generate = rand(rng, 1:curmaxsize)
    return _gen_llm_random_tree(tree_size_to_generate, options, nfeatures, T)
end

@unstable function _gen_llm_random_tree(
    node_count::Int, options::AbstractOptions, nfeatures::Int, ::Type{T}
)::AbstractExpressionNode{T} where {T<:DATA_TYPE}
    options = lasr_context(options)
    assumptions = _assumptions(options)

    conversation = [
        SystemMessage(
            load_prompt(prompt_path(options.prompts_dir, "gen_random_system.prompt"))
        ),
        UserMessage(
            construct_prompt(
                load_prompt(prompt_path(options.prompts_dir, "gen_random_user.prompt")),
                assumptions,
                "assump",
            ),
        ),
    ]
    rendered_msg = render_conversation(
        conversation, options; N=options.num_generated_equations
    )

    gen_tree_options, gen_id = request_suggestions(
        options,
        "gen_random",
        conversation,
        options.num_generated_equations;
        rendered_msg=rendered_msg,
        variables=get_vars(options),
        operators=get_ops(options),
        no_system_message=false,
        verbose=options.verbose,
    )
    if isempty(gen_tree_options)
        return gen_random_tree_fixed_size(node_count, options, nfeatures, T)
    end

    chosen = _first_usable(T, gen_tree_options, options)
    chosen === nothing &&
        return gen_random_tree_fixed_size(node_count, options, nfeatures, T)
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
`"1.0"`) reference no feature and are rejected, mirroring the `_is_one_constant` guard the
single-shot generators apply, but generalized to any constant-only tree.
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

Ask the LLM for a batch of full-expression proposals (a single batched call) and return
every proposal that parses and is usable (feature-referencing and within `curmaxsize`).
Unlike `_gen_llm_random_tree`, which samples a single tree, this keeps all usable
candidates so the caller can constant-fit each and keep the best. Returns an empty vector
if the call fails or nothing usable was produced.
"""
function llm_generate_candidates(
    options::AbstractOptions, curmaxsize::Int, nfeatures::Int, ::Type{T}
)::Vector{<:AbstractExpressionNode{T}} where {T<:DATA_TYPE}
    options = lasr_context(options)
    assumptions = _assumptions(options)

    conversation = [
        SystemMessage(
            load_prompt(prompt_path(options.prompts_dir, "gen_random_system.prompt"))
        ),
        UserMessage(
            construct_prompt(
                load_prompt(prompt_path(options.prompts_dir, "gen_random_user.prompt")),
                assumptions,
                "assump",
            ),
        ),
    ]

    # `request_suggestions` returns the raw proposal strings; parse each into a tree the
    # same way the single-shot generators do, then keep the usable ones.
    gen_tree_options, gen_id = request_suggestions(
        options,
        "gen_candidates",
        conversation,
        options.num_generated_equations;
        variables=get_vars(options),
        operators=get_ops(options),
        no_system_message=false,
        verbose=options.verbose,
    )
    if isempty(gen_tree_options)
        return AbstractExpressionNode{T}[]
    end

    candidates = AbstractExpressionNode{T}[]
    for raw in gen_tree_options
        local tree
        try
            tree = get_contents(parse_expr(T, _clean(raw), options))
        catch
            continue
        end
        if _is_usable_candidate(tree, nfeatures, curmaxsize)
            push!(candidates, tree)
        end
    end
    return candidates
end

function llm_mutate_tree(
    ex::E, options::AbstractOptions
)::E where {T<:DATA_TYPE,E<:AbstractExpression{T}}
    options = lasr_context(options)
    tree = get_contents(ex)
    ex = with_contents(ex, llm_mutate_tree(tree, options))
    return ex
end

"""LLM Mutation on a tree"""
function llm_mutate_tree(
    tree::NT, options::AbstractOptions
)::NT where {T<:DATA_TYPE,NT<:AbstractExpressionNode{T}}
    options = lasr_context(options)
    expr = render_expr(tree, options)

    assumptions = _assumptions(options; query=expr)

    conversation = [
        SystemMessage(
            load_prompt(prompt_path(options.prompts_dir, "mutate_system.prompt"))
        ),
        UserMessage(
            construct_prompt(
                load_prompt(prompt_path(options.prompts_dir, "mutate_user.prompt")),
                assumptions,
                "assump",
            ),
        ),
    ]
    rendered_msg = render_conversation(
        conversation, options; N=options.num_generated_equations, expr=expr
    )

    mut_tree_options, gen_id = request_suggestions(
        options,
        "mutate",
        conversation,
        options.num_generated_equations;
        rendered_msg=rendered_msg,
        variables=get_vars(options),
        operators=get_ops(options),
        expr=expr,
        no_system_message=false,
        verbose=options.verbose,
    )
    if isempty(mut_tree_options)
        return tree
    end

    chosen = _first_usable(T, mut_tree_options, options)
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
    tree1 = get_contents(ex1)
    tree2 = get_contents(ex2)
    tree1, tree2 = llm_crossover_trees(tree1, tree2, options)
    ex1 = with_contents(ex1, tree1)
    ex2 = with_contents(ex2, tree2)
    return ex1, ex2
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

    assumptions = _assumptions(options; query=expr1 * " " * expr2)

    conversation = [
        SystemMessage(
            load_prompt(prompt_path(options.prompts_dir, "crossover_system.prompt"))
        ),
        UserMessage(
            construct_prompt(
                load_prompt(prompt_path(options.prompts_dir, "crossover_user.prompt")),
                assumptions,
                "assump",
            ),
        ),
    ]

    rendered_msg = render_conversation(
        conversation, options; N=options.num_generated_equations, expr1=expr1, expr2=expr2
    )

    cross_tree_options, gen_id = request_suggestions(
        options,
        "crossover",
        conversation,
        options.num_generated_equations;
        rendered_msg=rendered_msg,
        variables=get_vars(options),
        operators=get_ops(options),
        expr1=expr1,
        expr2=expr2,
        no_system_message=false,
        verbose=options.verbose,
    )
    if isempty(cross_tree_options)
        return tree1, tree2
    end

    # Pick up to two distinct usable candidates in random order. A missing child falls back
    # to a parent, never a constant-1 tree (the old N==1 and fill paths skipped that check).
    usable = AbstractExpressionNode{T}[]
    for i in randperm(length(cross_tree_options))
        t = parse_expr(T, _clean(cross_tree_options[i]), options)
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
