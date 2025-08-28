module LLMFunctionsModule

using Random: default_rng, AbstractRNG, rand, randperm
using DispatchDoctor: @unstable
using DynamicExpressions:
    Node,
    AbstractExpressionNode,
    AbstractExpression,
    ParametricExpression,
    ParametricNode,
    AbstractNode,
    NodeSampler,
    get_contents,
    with_contents,
    constructorof,
    set_node!,
    count_nodes,
    has_constants,
    has_operators,
    string_tree,
    AbstractOperatorEnum
using Compat: Returns, @inline
using SymbolicRegression:
    Options, DATA_TYPE, gen_random_tree_fixed_size, crossover_trees, AbstractOptions
using SymbolicRegression.MutationFunctionsModule: with_contents_for_mutation
using ..LLMOptionsModule: LaSROptions
using ..LLMUtilsModule:
    load_prompt,
    convertDict,
    get_vars,
    get_ops,
    construct_prompt,
    format_pareto,
    sample_context
using ..ParseModule: render_expr, parse_expr
using ..LoggingModule: log_generation!
using PromptingTools:
    SystemMessage,
    UserMessage,
    AIMessage,
    aigenerate,
    render,
    CustomOpenAISchema,
    OllamaSchema,
    OpenAISchema
using JSON: parse
using UUIDs: uuid1

@unstable function llm_randomize_tree(
    ex::AbstractExpression,
    curmaxsize::Int,
    options::AbstractOptions,
    nfeatures::Int,
    rng::AbstractRNG=default_rng(),
)
    tree = get_contents(ex)
    context = nothing
    ex = with_contents_for_mutation(
        ex, llm_randomize_tree(tree, curmaxsize, options, nfeatures, rng), context
    )
    return ex
end

@unstable function llm_randomize_tree(
    ::AbstractExpressionNode{T},
    curmaxsize::Int,
    options::AbstractOptions,
    nfeatures::Int,
    rng::AbstractRNG=default_rng(),
) where {T<:DATA_TYPE}
    tree_size_to_generate = rand(rng, 1:curmaxsize)
    return _gen_llm_random_tree(tree_size_to_generate, options, nfeatures, T)
end

function _gen_llm_random_tree(
    node_count::Int, options::AbstractOptions, nfeatures::Int, ::Type{T}
)::AbstractExpressionNode{T} where {T<:DATA_TYPE}
    if isnothing(options.idea_database)
        assumptions = []
    else
        assumptions = sample_context(
            options.idea_database,
            min(options.num_pareto_context, length(options.idea_database)),
            options.max_concepts,
        )
    end

    if options.llm_context != ""
        pushfirst!(assumptions, options.llm_context)
    end

    if !options.use_concepts
        assumptions = []
    end

    conversation = [
        SystemMessage(load_prompt(options.prompts_dir * "gen_random_system.prompt")),
        UserMessage(
            construct_prompt(
                load_prompt(options.prompts_dir * "gen_random_user.prompt"),
                assumptions,
                "assump",
            ),
        ),
    ]
    rendered_msg = join(
        [
            x["content"] for x in render(
                CustomOpenAISchema(),
                conversation;
                variables=get_vars(options),
                operators=get_ops(options),
                N=options.num_generated_equations,
                no_system_message=false,
            )
        ],
        "\n",
    )

    # Instantiate ID once for all subsequent log calls in this function
    gen_id = uuid1()
    log_generation!(
        options.lasr_logger; id=gen_id, mode="gen_random", llm_input=rendered_msg
    )

    msg = nothing
    try
        msg = aigenerate(
            CustomOpenAISchema(),
            conversation;
            variables=get_vars(options),
            operators=get_ops(options),
            N=options.num_generated_equations,
            api_key=options.api_key,
            model=options.model,
            api_kwargs=convertDict(options.api_kwargs),
            http_kwargs=convertDict(options.http_kwargs),
            no_system_message=false,
            verbose=options.verbose,
        )
    catch e
        log_generation!(
            options.lasr_logger; id=gen_id, mode="gen_random", failed="None." * string(e)
        )
        return gen_random_tree_fixed_size(node_count, options, nfeatures, T)
    end

    log_generation!(
        options.lasr_logger; id=gen_id, mode="gen_random", llm_output=string(msg.content)
    )

    gen_tree_options = parse_msg_content(String(msg.content), options)

    N = min(size(gen_tree_options)[1], options.num_generated_equations)

    if N == 0
        log_generation!(options.lasr_logger; id=gen_id, mode="gen_random", failed="None")
        return gen_random_tree_fixed_size(node_count, options, nfeatures, T)
    end

    for i in 1:N
        l = rand(1:N)
        t = parse_expr(
            T,
            String(strip(gen_tree_options[l], [' ', '\n', '"', ',', '.', '[', ']'])),
            options,
        )
        if t.val == 1 && t.constant
            continue
        end
        log_generation!(
            options.lasr_logger;
            id=gen_id,
            mode="gen_random",
            chosen=render_expr(t, options),
        )
        return t
    end

    out = parse_expr(
        T, String(strip(gen_tree_options[1], [' ', '\n', '"', ',', '.', '[', ']'])), options
    )
    log_generation!(
        options.lasr_logger; id=gen_id, mode="gen_random", chosen=render_expr(out, options)
    )

    if out.val == 1 && out.constant
        return gen_random_tree_fixed_size(node_count, options, nfeatures, T)
    end

    return out
end

function concept_evolution(idea_database, options::AbstractOptions)
    num_ideas = size(idea_database)[1]
    if num_ideas <= options.max_concepts
        return nothing
    end

    ideas = [idea_database[rand((options.max_concepts + 1):num_ideas)] for _ in 1:num_ideas]
    conversation = [
        SystemMessage(load_prompt(options.prompts_dir * "concept_evolution_system.prompt")),
        UserMessage(
            construct_prompt(
                load_prompt(options.prompts_dir * "concept_evolution_user.prompt"),
                ideas,
                "idea",
            ),
        ),
    ]

    rendered_msg = join(
        [
            x["content"] for x in render(
                CustomOpenAISchema(),
                conversation;
                variables=get_vars(options),
                operators=get_ops(options),
                N=options.num_generated_concepts,
                no_system_message=false,
            )
        ],
        "\n",
    )

    # Instantiate ID once for this function
    gen_id = uuid1()
    log_generation!(
        options.lasr_logger; id=gen_id, mode="concept_evolution", llm_input=rendered_msg
    )

    msg = nothing
    try
        msg = aigenerate(
            CustomOpenAISchema(),
            conversation;
            N=options.num_generated_concepts,
            api_key=options.api_key,
            model=options.model,
            api_kwargs=convertDict(options.api_kwargs),
            http_kwargs=convertDict(options.http_kwargs),
        )
    catch e
        log_generation!(
            options.lasr_logger;
            id=gen_id,
            mode="concept_evolution",
            failed="None." * string(e),
        )
        return nothing
    end

    log_generation!(
        options.lasr_logger;
        id=gen_id,
        mode="concept_evolution",
        llm_output=string(msg.content),
    )

    idea_options = parse_msg_content(String(msg.content), options)

    N = min(size(idea_options)[1], options.num_generated_concepts)

    if N == 0
        log_generation!(
            options.lasr_logger; id=gen_id, mode="concept_evolution", failed="None"
        )
        return nothing
    end

    chosen_idea = String(
        strip(idea_options[rand(1:N)], [' ', '\n', '"', ',', '.', '[', ']'])
    )

    log_generation!(
        options.lasr_logger; id=gen_id, mode="concept_evolution", chosen=chosen_idea
    )

    return chosen_idea
end

@unstable function try_capture(pattern::Regex, text::String)::Union{Nothing,AbstractString}
    m = match(pattern, text)
    return m === nothing ? nothing : get(m.captures, 1, nothing)
end

function parse_msg_content(msg_content::String, options::AbstractOptions)::Vector{String}
    # Attempt extraction with several patterns in order
    patterns = [r"```json(.*?)```"s, r"```(.*?)```"s, r"(\[.*?\])"s]

    content = nothing
    for pat in patterns
        content = try_capture(pat, msg_content)
        content !== nothing && break
    end

    content = content === nothing ? msg_content : content

    out = nothing
    try
        out = parse(content)
    catch
        if options.verbose
            @debug "Failed to parse content: $content"
        end
    end

    try
        out = eval(Meta.parse(msg_content))
    catch
        if options.verbose
            @debug "Failed to eval content: $content"
        end
    end

    if out isa Dict && all(x -> isa(x, String), values(out))
        return collect(values(out))
    elseif out isa Vector && all(x -> isa(x, String), out)
        return out
    end
    return String[]
end

function generate_concepts(dominating, worst_members, options::AbstractOptions)
    # turn dominating pareto curve into ideas as strings
    if isnothing(dominating)
        return nothing
    end

    gexpr = format_pareto(dominating, options, options.num_pareto_context)
    bexpr = format_pareto(worst_members, options, options.num_pareto_context)

    conversation = [
        SystemMessage(load_prompt(options.prompts_dir * "generate_concepts_system.prompt")),
        UserMessage(
            construct_prompt(
                construct_prompt(
                    load_prompt(options.prompts_dir * "generate_concepts_user.prompt"),
                    gexpr,
                    "gexpr",
                ),
                bexpr,
                "bexpr",
            ),
        ),
    ]
    rendered_msg = join(
        [
            x["content"] for x in render(
                CustomOpenAISchema(),
                conversation;
                variables=get_vars(options),
                operators=get_ops(options),
                N=options.num_generated_concepts,
                no_system_message=false,
            )
        ],
        "\n",
    )

    # Instantiate ID for this function
    gen_id = uuid1()
    log_generation!(
        options.lasr_logger; id=gen_id, mode="generate_concepts", llm_input=rendered_msg
    )

    msg = nothing
    try
        msg = aigenerate(
            CustomOpenAISchema(),
            conversation;
            variables=get_vars(options),
            operators=get_ops(options),
            N=options.num_generated_concepts,
            api_key=options.api_key,
            model=options.model,
            api_kwargs=convertDict(options.api_kwargs),
            http_kwargs=convertDict(options.http_kwargs),
            no_system_message=false,
            verbose=options.verbose,
        )
    catch e
        log_generation!(
            options.lasr_logger;
            id=gen_id,
            mode="generate_concepts",
            failed="None." * string(e),
        )
        return nothing
    end

    log_generation!(
        options.lasr_logger;
        id=gen_id,
        mode="generate_concepts",
        llm_output=string(msg.content),
    )

    idea_options = parse_msg_content(String(msg.content), options)

    N = min(size(idea_options)[1], options.num_generated_concepts)

    if N == 0
        log_generation!(
            options.lasr_logger; id=gen_id, mode="generate_concepts", failed="None"
        )
        return nothing
    end

    for _ in 1:(options.num_concept_crossover)
        a = rand(1:N)
        chosen_idea = String(strip(idea_options[a], [' ', '\n', '"', ',', '.', '[', ']']))
        log_generation!(
            options.lasr_logger; id=gen_id, mode="generate_concepts", chosen=chosen_idea
        )
        push!(options.idea_database, chosen_idea)
    end

    for _ in 1:(options.num_concept_crossover)
        out = concept_evolution(options.idea_database, options)
        if !isnothing(out)
            pushfirst!(options.idea_database, out)
        end
    end
end

function llm_mutate_tree(
    ex::AbstractExpression{T}, options::AbstractOptions
)::AbstractExpression{T} where {T<:DATA_TYPE}
    tree = get_contents(ex)
    ex = with_contents(ex, llm_mutate_tree(tree, options))
    return ex
end

"""LLM Mutation on a tree"""
function llm_mutate_tree(
    tree::AbstractExpressionNode{T}, options::AbstractOptions
)::AbstractExpressionNode{T} where {T<:DATA_TYPE}
    expr = render_expr(tree, options)

    if isnothing(options.idea_database)
        assumptions = []
    else
        assumptions = sample_context(
            options.idea_database, options.num_pareto_context, options.max_concepts
        )
    end

    if !options.use_concepts
        assumptions = []
    end
    if options.llm_context != ""
        pushfirst!(assumptions, options.llm_context)
    end

    conversation = [
        SystemMessage(load_prompt(options.prompts_dir * "mutate_system.prompt")),
        UserMessage(
            construct_prompt(
                load_prompt(options.prompts_dir * "mutate_user.prompt"),
                assumptions,
                "assump",
            ),
        ),
    ]
    rendered_msg = join(
        [
            x["content"] for x in render(
                CustomOpenAISchema(),
                conversation;
                variables=get_vars(options),
                operators=get_ops(options),
                N=options.num_generated_equations,
                expr=expr,
                no_system_message=false,
            )
        ],
        "\n",
    )

    # Instantiate ID for this function
    gen_id = uuid1()
    log_generation!(options.lasr_logger; id=gen_id, mode="mutate", llm_input=rendered_msg)

    msg = nothing
    try
        msg = aigenerate(
            CustomOpenAISchema(),
            conversation; #OllamaSchema(), conversation;
            variables=get_vars(options),
            operators=get_ops(options),
            N=options.num_generated_equations,
            expr=expr,
            api_key=options.api_key,
            model=options.model,
            api_kwargs=convertDict(options.api_kwargs),
            http_kwargs=convertDict(options.http_kwargs),
            no_system_message=false,
            verbose=options.verbose,
        )
    catch e
        log_generation!(
            options.lasr_logger; id=gen_id, mode="mutate", failed="None." * string(e)
        )
        return tree
    end

    log_generation!(
        options.lasr_logger; id=gen_id, mode="mutate", llm_output=string(msg.content)
    )

    mut_tree_options = parse_msg_content(String(msg.content), options)

    N = min(size(mut_tree_options)[1], options.num_generated_equations)

    if N == 0
        log_generation!(options.lasr_logger; id=gen_id, mode="mutate", failed="None")
        return tree
    end

    for i in 1:N
        l = rand(1:N)
        t = parse_expr(
            T,
            String(strip(mut_tree_options[l], [' ', '\n', '"', ',', '.', '[', ']'])),
            options,
        )
        if t.val == 1 && t.constant
            continue
        end

        log_generation!(
            options.lasr_logger; id=gen_id, mode="mutate", chosen=render_expr(t, options)
        )
        return t
    end

    out = parse_expr(
        T, String(strip(mut_tree_options[1], [' ', '\n', '"', ',', '.', '[', ']'])), options
    )

    log_generation!(
        options.lasr_logger; id=gen_id, mode="mutate", chosen=render_expr(out, options)
    )
    return out
end

function llm_crossover_trees(
    ex1::E, ex2::E, options::AbstractOptions
)::Tuple{E,E} where {T,E<:AbstractExpression{T}}
    tree1 = get_contents(ex1)
    tree2 = get_contents(ex2)
    tree1, tree2 = llm_crossover_trees(tree1, tree2, options)
    ex1 = with_contents(ex1, tree1)
    ex2 = with_contents(ex2, tree2)
    return ex1, ex2
end

"""LLM Crossover between two expressions"""
function llm_crossover_trees(
    tree1::AbstractExpressionNode{T},
    tree2::AbstractExpressionNode{T},
    options::AbstractOptions,
)::Tuple{AbstractExpressionNode{T},AbstractExpressionNode{T}} where {T<:DATA_TYPE}
    expr1 = render_expr(tree1, options)
    expr2 = render_expr(tree2, options)

    if isnothing(options.idea_database)
        assumptions = []
    else
        assumptions = sample_context(
            options.idea_database,
            min(options.num_pareto_context, length(options.idea_database)),
            options.max_concepts,
        )
    end

    if !options.use_concepts
        assumptions = []
    end

    if options.llm_context != ""
        pushfirst!(assumptions, options.llm_context)
    end

    conversation = [
        SystemMessage(load_prompt(options.prompts_dir * "crossover_system.prompt")),
        UserMessage(
            construct_prompt(
                load_prompt(options.prompts_dir * "crossover_user.prompt"),
                assumptions,
                "assump",
            ),
        ),
    ]

    rendered_msg = join(
        [
            x["content"] for x in render(
                CustomOpenAISchema(),
                conversation;
                variables=get_vars(options),
                operators=get_ops(options),
                N=options.num_generated_equations,
                expr1=expr1,
                expr2=expr2,
                no_system_message=false,
            )
        ],
        "\n",
    )

    # Instantiate ID for crossover
    gen_id = uuid1()
    log_generation!(
        options.lasr_logger; id=gen_id, mode="crossover", llm_input=rendered_msg
    )

    msg = nothing
    try
        msg = aigenerate(
            CustomOpenAISchema(),
            conversation;
            variables=get_vars(options),
            operators=get_ops(options),
            N=options.num_generated_equations,
            expr1=expr1,
            expr2=expr2,
            api_key=options.api_key,
            model=options.model,
            api_kwargs=convertDict(options.api_kwargs),
            http_kwargs=convertDict(options.http_kwargs),
            no_system_message=false,
            verbose=options.verbose,
        )
    catch e
        log_generation!(
            options.lasr_logger; id=gen_id, mode="crossover", failed="None." * string(e)
        )
        return tree1, tree2
    end

    log_generation!(
        options.lasr_logger; id=gen_id, mode="crossover", llm_output=string(msg.content)
    )

    cross_tree_options = parse_msg_content(String(msg.content), options)

    cross_tree1 = nothing
    cross_tree2 = nothing

    N = min(size(cross_tree_options)[1], options.num_generated_equations)

    if N == 0
        log_generation!(options.lasr_logger; id=gen_id, mode="crossover", failed="None")
        return tree1, tree2
    end

    if N == 1
        t = parse_expr(
            T,
            String(strip(cross_tree_options[1], [' ', '\n', '"', ',', '.', '[', ']'])),
            options,
        )

        log_generation!(
            options.lasr_logger; id=gen_id, mode="crossover", chosen=render_expr(t, options)
        )
        return t, tree2
    end

    for i in 1:(2 * N)
        l = rand(1:N)
        t = parse_expr(
            T,
            String(strip(cross_tree_options[l], [' ', '\n', '"', ',', '.', '[', ']'])),
            options,
        )
        if t.val == 1 && t.constant
            continue
        end

        if isnothing(cross_tree1)
            cross_tree1 = t
        elseif isnothing(cross_tree2)
            cross_tree2 = t
            break
        end
    end

    if isnothing(cross_tree1)
        cross_tree1 = parse_expr(
            T,
            String(strip(cross_tree_options[1], [' ', '\n', '"', ',', '.', '[', ']'])),
            options,
        )
    end

    if isnothing(cross_tree2)
        cross_tree2 = parse_expr(
            T,
            String(strip(cross_tree_options[2], [' ', '\n', '"', ',', '.', '[', ']'])),
            options,
        )
    end

    recording_str =
        render_expr(cross_tree1, options) * " && " * render_expr(cross_tree2, options)
    log_generation!(options.lasr_logger; id=gen_id, mode="crossover", chosen=recording_str)

    return cross_tree1, cross_tree2
end

end
