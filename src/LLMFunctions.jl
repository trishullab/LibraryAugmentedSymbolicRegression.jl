module LLMFunctionsModule

using Random: default_rng, AbstractRNG, rand, randperm
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
    copy_node,
    set_node!,
    count_nodes,
    has_constants,
    has_operators,
    string_tree,
    AbstractOperatorEnum
using Compat: Returns, @inline
using SymbolicRegression:
    Options, DATA_TYPE, gen_random_tree_fixed_size, random_node_and_parent, AbstractOptions
using ..LLMOptionsModule: LaSROptions

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

"""LLM Recoder records the LLM calls for debugging purposes."""
function llm_recorder(options::LaSROptions, expr::String, mode::String="debug")
    if options.use_llm
        if !isdir(options.llm_recorder_dir)
            mkdir(options.llm_recorder_dir)
        end
        recorder = open(joinpath(options.llm_recorder_dir, "llm_calls.txt"), "a")
        write(recorder, string("[", mode, "]\n", expr, "\n[/", mode, "]\n"))
        close(recorder)
    end
end

function load_prompt(path::String)::String
    # load prompt file 
    f = open(path, "r")
    s = read(f, String)
    s = strip(s)
    close(f)
    return s
end

function convertDict(d)::NamedTuple
    return (; Dict(Symbol(k) => v for (k, v) in d)...)
end

function get_vars(options::LaSROptions)::String
    variable_names = get_variable_names(options.variable_names)
    return join(variable_names, ", ")
end

function get_ops(options::LaSROptions)::String
    binary_operators = map(v -> string(v), options.operators.binops)
    unary_operators = map(v -> string(v), options.operators.unaops)
    # Binary Ops: +, *, -, /, safe_pow (^)
    # Unary Ops: exp, safe_log, safe_sqrt, sin, cos
    return replace(
        replace(
            "binary operators: " *
            join(binary_operators, ", ") *
            ", and unary operators: " *
            join(unary_operators, ", "),
            "safe_" => "",
        ),
        "pow" => "^",
    )
end

"""
Constructs a prompt by replacing the element_id_tag with the corresponding element in the element_list.
If the element_list is longer than the number of occurrences of the element_id_tag, the missing elements are added after the last occurrence.
If the element_list is shorter than the number of occurrences of the element_id_tag, the extra ids are removed.
"""
function construct_prompt(
    user_prompt::String, element_list::Vector, element_id_tag::String
)::String
    # Split the user prompt into lines
    lines = split(user_prompt, "\n")

    # Filter lines that match the pattern "... : {{element_id_tag[1-9]}}
    pattern = r"^.*: \{\{" * element_id_tag * r"\d+\}\}$"

    # find all occurrences of the element_id_tag
    n_occurrences = count(x -> occursin(pattern, x), lines)

    # if n_occurrences is less than |element_list|, add the missing elements after the last occurrence
    if n_occurrences < length(element_list)
        last_occurrence = findlast(x -> occursin(pattern, x), lines)
        @assert last_occurrence !== nothing "No occurrences of the element_id_tag found in the user prompt."
        for i in reverse((n_occurrences + 1):length(element_list))
            new_line = replace(lines[last_occurrence], string(n_occurrences) => string(i))
            insert!(lines, last_occurrence + 1, new_line)
        end
    end

    new_prompt = ""
    idx = 1
    for line in lines
        # if the line matches the pattern
        if occursin(pattern, line)
            if idx > length(element_list)
                continue
            end
            # replace the element_id_tag with the corresponding element
            new_prompt *=
                replace(line, r"\{\{" * element_id_tag * r"\d+\}\}" => element_list[idx]) *
                "\n"
            idx += 1
        else
            new_prompt *= line * "\n"
        end
    end
    return new_prompt
end

function llm_randomize_tree(
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
function llm_randomize_tree(
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
    node_count::Int, options::LaSROptions, nfeatures::Int, ::Type{T}
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
        SystemMessage(load_prompt(options.prompts_dir * "gen_random_system.txt")),
        UserMessage(
            construct_prompt(
                load_prompt(options.prompts_dir * "gen_random_user.txt"),
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

    llm_recorder(options.llm_options, rendered_msg, "llm_input|gen_random")

    msg = nothing
    try
        msg = aigenerate(
            CustomOpenAISchema(),
            conversation;
            variables=get_vars(options),
            operaotrs=get_ops(options),
            N=options.num_generated_equations,
            api_key=options.api_key,
            model=options.model,
            api_kwargs=convertDict(options.api_kwargs),
            http_kwargs=convertDict(options.http_kwargs),
            no_system_message=true,
            verbose=false,
        )
    catch e
        llm_recorder(options.llm_options, "None " * string(e), "gen_random|failed")
        return gen_random_tree_fixed_size(node_count, options, nfeatures, T)
    end
    llm_recorder(options.llm_options, string(msg.content), "llm_output|gen_random")

    gen_tree_options = parse_msg_content(String(msg.content))

    N = min(size(gen_tree_options)[1], N)

    if N == 0
        llm_recorder(options.llm_options, "None", "gen_random|failed")
        return gen_random_tree_fixed_size(node_count, options, nfeatures, T)
    end

    for i in 1:N
        l = rand(1:N)
        t = expr_to_tree(
            T,
            String(strip(gen_tree_options[l], [' ', '\n', '"', ',', '.', '[', ']'])),
            options,
        )
        if t.val == 1 && t.constant
            continue
        end
        llm_recorder(options.llm_options, tree_to_expr(t, options), "gen_random")

        return t
    end

    out = expr_to_tree(
        T, String(strip(gen_tree_options[1], [' ', '\n', '"', ',', '.', '[', ']'])), options
    )

    llm_recorder(options.llm_options, tree_to_expr(out, options), "gen_random")

    if out.val == 1 && out.constant
        return gen_random_tree_fixed_size(node_count, options, nfeatures, T)
    end

    return out
end

"""Crossover between two expressions"""
function crossover_trees(
    tree1::AbstractExpressionNode{T}, tree2::AbstractExpressionNode{T}
)::Tuple{AbstractExpressionNode{T},AbstractExpressionNode{T}} where {T<:DATA_TYPE}
    tree1 = copy_node(tree1)
    tree2 = copy_node(tree2)

    node1, parent1, side1 = random_node_and_parent(tree1)
    node2, parent2, side2 = random_node_and_parent(tree2)

    node1 = copy_node(node1)

    if side1 == 'l'
        parent1.l = copy_node(node2)
        # tree1 now contains this.
    elseif side1 == 'r'
        parent1.r = copy_node(node2)
        # tree1 now contains this.
    else # 'n'
        # This means that there is no parent2.
        tree1 = copy_node(node2)
    end

    if side2 == 'l'
        parent2.l = node1
    elseif side2 == 'r'
        parent2.r = node1
    else # 'n'
        tree2 = node1
    end
    return tree1, tree2
end

function sketch_const(val)
    does_not_need_brackets = (typeof(val) <: Union{Real,AbstractArray})

    if does_not_need_brackets
        if isinteger(val) && (abs(val) < 5) # don't abstract integer constants from -4 to 4, useful for exponents
            string(val)
        else
            "C"
        end
    else
        if isinteger(val) && (abs(val) < 5) # don't abstract integer constants from -4 to 4, useful for exponents
            "(" * string(val) * ")"
        else
            "(C)"
        end
    end
end

function tree_to_expr(
    ex::AbstractExpression{T}, options::LaSROptions
)::String where {T<:DATA_TYPE}
    return tree_to_expr(get_contents(ex), options)
end

function tree_to_expr(tree::AbstractExpressionNode{T}, options)::String where {T<:DATA_TYPE}
    variable_names = get_variable_names(options.variable_names)
    return string_tree(
        tree, options.operators; f_constant=sketch_const, variable_names=variable_names
    )
end

function get_variable_names(variable_names::Dict)::Vector{String}
    return [variable_names[key] for key in sort(collect(keys(variable_names)))]
end

function get_variable_names(variable_names::Nothing)::Vector{String}
    return ["x", "y", "z", "k", "j", "l", "m", "n", "p", "a", "b"]
end

function handle_not_expr(::Type{T}, x, var_names)::Node{T} where {T<:DATA_TYPE}
    if x isa Real
        Node{T}(; val=convert(T, x)) # old:  Node(T, 0, true, convert(T,x))
    elseif x isa Symbol
        if x === :C # constant that got abstracted
            Node{T}(; val=convert(T, 1)) # old: Node(T, 0, true, convert(T,1))
        else
            feature = findfirst(isequal(string(x)), var_names)
            if isnothing(feature) # invalid var name, just assume its x0
                feature = 1
            end
            Node{T}(; feature=feature) # old: Node(T, 0, false, nothing, feature)
        end
    else
        Node{T}(; val=convert(T, 1)) # old: Node(T, 0, true, convert(T,1)) # return a constant being 0
    end
end

function expr_to_tree_recurse(
    ::Type{T}, node::Expr, op::AbstractOperatorEnum, var_names
)::Node{T} where {T<:DATA_TYPE}
    args = node.args
    x = args[1]
    degree = length(args)

    if degree == 1
        handle_not_expr(T, x, var_names)
    elseif degree == 2
        unary_operators = map(v -> string(v), map(unaopmap, op.unaops))
        idx = findfirst(isequal(string(x)), unary_operators)
        if isnothing(idx) # if not used operator, make it the first one
            idx = findfirst(isequal("safe_" * string(x)), unary_operators)
            if isnothing(idx)
                idx = 1
            end
        end

        left = if (args[2] isa Expr)
            expr_to_tree_recurse(T, args[2], op, var_names)
        else
            handle_not_expr(T, args[2], var_names)
        end

        Node(; op=idx, l=left) # old: Node(1, false, nothing, 0, idx, left)
    elseif degree == 3
        if x === :^
            x = :pow
        end
        binary_operators = map(v -> string(v), map(binopmap, op.binops))
        idx = findfirst(isequal(string(x)), binary_operators)
        if isnothing(idx) # if not used operator, make it the first one
            idx = findfirst(isequal("safe_" * string(x)), binary_operators)
            if isnothing(idx)
                idx = 1
            end
        end

        left = if (args[2] isa Expr)
            expr_to_tree_recurse(T, args[2], op, var_names)
        else
            handle_not_expr(T, args[2], var_names)
        end
        right = if (args[3] isa Expr)
            expr_to_tree_recurse(T, args[3], op, var_names)
        else
            handle_not_expr(T, args[3], var_names)
        end

        Node(; op=idx, l=left, r=right) # old: Node(2, false, nothing, 0, idx, left, right)
    else
        Node{T}(; val=convert(T, 1))  # old: Node(T, 0, true, convert(T,1)) # return a constant being 1
    end
end

function expr_to_tree_run(::Type{T}, x::String, options)::Node{T} where {T<:DATA_TYPE}
    try
        expr = Meta.parse(x)
        variable_names = ["x", "y", "z", "k", "j", "l", "m", "n", "p", "a", "b"]
        if !isnothing(options.variable_names)
            variable_names = [
                options.variable_names[key] for
                key in sort(collect(keys(options.variable_names)))
            ]
        end
        if expr isa Expr
            expr_to_tree_recurse(T, expr, options.operators, variable_names)
        else
            handle_not_expr(T, expr, variable_names)
        end
    catch
        Node{T}(; val=convert(T, 1)) # old: Node(T, 0, true, convert(T,1)) # return a constant being 1
    end
end

function expr_to_tree(::Type{T}, x::String, options) where {T<:DATA_TYPE}
    if options.is_parametric
        out = ParametricNode{T}(expr_to_tree_run(T, x, options))
    else
        out = Node{T}(expr_to_tree_run(T, x, options))
    end
    return out
end

function format_pareto(dominating, options, num_pareto_context::Int)::Vector{String}
    pareto = Vector{String}()
    if !isnothing(dominating) && size(dominating)[1] > 0
        idx = randperm(size(dominating)[1])
        for i in 1:min(size(dominating)[1], num_pareto_context)
            push!(pareto, tree_to_expr(dominating[idx[i]].tree, options))
        end
    end
    while size(pareto)[1] < num_pareto_context
        push!(pareto, "None")
    end
    return pareto
end

function sample_one_context(idea_database, max_concepts)::String
    if isnothing(idea_database)
        return "None"
    end

    N = size(idea_database)[1]
    if N == 0
        return "None"
    end

    try
        idea_database[rand(1:min(max_concepts, N))]
    catch e
        "None"
    end
end

function sample_context(idea_database, N, max_concepts)::Vector{String}
    assumptions = Vector{String}()
    if isnothing(idea_database)
        for _ in 1:N
            push!(assumptions, "None")
        end
        return assumptions
    end

    if size(idea_database)[1] < N
        for i in 1:(size(idea_database)[1])
            push!(assumptions, idea_database[i])
        end
        for i in (size(idea_database)[1] + 1):N
            push!(assumptions, "None")
        end
        return assumptions
    end

    while size(assumptions)[1] < N
        chosen_idea = sample_one_context(idea_database, max_concepts)
        if chosen_idea in assumptions
            continue
        end
        push!(assumptions, chosen_idea)
    end
    return assumptions
end

function concept_evolution(idea_database, options::LaSROptions)
    num_ideas = size(idea_database)[1]
    if num_ideas <= options.max_concepts
        return nothing
    end

    ideas = [idea_database[rand((options.idea_threshold + 1):num_ideas)] for _ in 1:n_ideas]
    conversation = [
        SystemMessage(load_prompt(options.prompts_dir * "prompt_evol_system.txt")),
        UserMessage(
            construct_prompt(
                load_prompt(options.prompts_dir * "prompt_evol_user.txt"), ideas, "idea"
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
    llm_recorder(options.llm_options, rendered_msg, "llm_input|ideas")

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
        llm_recorder(options.llm_options, "None " * string(e), "ideas|failed")
        return nothing
    end
    llm_recorder(options.llm_options, string(msg.content), "llm_output|ideas")

    idea_options = parse_msg_content(String(msg.content))

    N = min(size(idea_options)[1], N)

    if N == 0
        llm_recorder(options.llm_options, "None", "ideas|failed")
        return nothing
    end

    # only choose one, merging ideas not really crossover
    chosen_idea = String(
        strip(idea_options[rand(1:N)], [' ', '\n', '"', ',', '.', '[', ']'])
    )

    llm_recorder(options.llm_options, chosen_idea, "ideas")

    return chosen_idea
end

function parse_msg_content(msg_content)
    content = msg_content
    try
        content = match(r"```json(.*?)```"s, msg_content).captures[1]
    catch e
        try
            content = match(r"```(.*?)```"s, msg_content).captures[1]
        catch e2
            try
                content = match(r"\[(.*?)\]"s, msg_content).match
            catch e3
                content = msg_content
            end
        end
    end

    try
        out = parse(content) # json parse
        if out === nothing
            return []
        end
        if out isa Dict
            return [out[key] for key in keys(out)]
        end

        if out isa Vector && all(x -> isa(x, String), out)
            return out
        end
    catch e
        try
            content = strip(content, [' ', '\n', '"', ',', '.', '[', ']'])
            content = replace(content, "\n" => " ")
            out_list = split(content, "\", \"")
            return out_list
        catch e2
            return []
        end
    end

    try
        content = strip(content, [' ', '\n', '"', ',', '.', '[', ']'])
        content = replace(content, "\n" => " ")
        out_list = split(content, "\", \"")
        return out_list
    catch e3
        return []
    end
    # old method:
    # find first JSON list
    # first_idx = findfirst('[', content)
    # last_idx = findfirst(']', content)
    # content = chop(content, head=first_idx, tail=length(content) - last_idx + 1)

    # out_list = split(content, ",")
    # for i in 1:length(out_list)
    #     out_list[i] = replace(out_list[i], "//.*" => "") # filter comments
    # end

    # new method (for Llama since it follows directions better):
end

function update_idea_database(dominating, worst_members, options::LaSROptions)
    # turn dominating pareto curve into ideas as strings
    if isnothing(dominating)
        return nothing
    end

    gexpr = format_pareto(dominating, options, options.num_pareto_context)
    bexpr = format_pareto(worst_members, options, options.num_pareto_context)

    conversation = [
        SystemMessage(load_prompt(options.prompts_dir * "extract_idea_system.txt")),
        UserMessage(
            construct_prompt(
                construct_prompt(
                    load_prompt(options.prompts_dir * "extract_idea_user.txt"),
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

    llm_recorder(options.llm_options, rendered_msg, "llm_input|gen_random")

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
            no_system_message=true,
            verbose=false,
        )
    catch e
        llm_recorder(options.llm_options, "None " * string(e), "ideas|failed")
        return nothing
    end

    llm_recorder(options.llm_options, string(msg.content), "llm_output|ideas")

    idea_options = parse_msg_content(String(msg.content))

    N = min(size(idea_options)[1], N)

    if N == 0
        llm_recorder(options.llm_options, "None", "ideas|failed")
        return nothing
    end

    a = rand(1:N)

    chosen_idea1 = String(strip(idea_options[a], [' ', '\n', '"', ',', '.', '[', ']']))

    llm_recorder(options.llm_options, chosen_idea1, "ideas")
    pushfirst!(options.idea_database, chosen_idea1)

    if N > 1
        b = rand(1:(N - 1))
        if a == b
            b += 1
        end
        chosen_idea2 = String(strip(idea_options[b], [' ', '\n', '"', ',', '.', '[', ']']))

        llm_recorder(options.llm_options, chosen_idea2, "ideas")

        pushfirst!(options.idea_database, chosen_idea2)
    end

    num_add = 2
    for _ in 1:num_add
        out = concept_evolution(options.idea_database, options)
        if !isnothing(out)
            pushfirst!(options.idea_database, out)
        end
    end
end

function llm_mutate_tree(
    ex::AbstractExpression{T}, options::LaSROptions
)::AbstractExpression{T} where {T<:DATA_TYPE}
    tree = get_contents(ex)
    ex = with_contents(ex, llm_mutate_tree(tree, options))
    return ex
end

"""LLM Mutation on a tree"""
function llm_mutate_tree(
    tree::AbstractExpressionNode{T}, options::LaSROptions
)::AbstractExpressionNode{T} where {T<:DATA_TYPE}
    expr = tree_to_expr(tree, options)

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
        SystemMessage(load_prompt(options.prompts_dir * "mutate_system.txt")),
        UserMessage(
            construct_prompt(
                load_prompt(options.prompts_dir * "mutate_user.txt"), assumptions, "assump"
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

    llm_recorder(options.llm_options, rendered_msg, "llm_input|mutate")

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
            no_system_message=true,
            verbose=false,
        )
    catch e
        llm_recorder(options.llm_options, "None " * string(e), "mutate|failed")
        # log error in llm_recorder
        return tree
    end

    llm_recorder(options.llm_options, string(msg.content), "llm_output|mutate")

    mut_tree_options = parse_msg_content(String(msg.content))

    N = min(size(mut_tree_options)[1], N)

    if N == 0
        llm_recorder(options.llm_options, "None", "mutate|failed")
        return tree
    end

    for i in 1:N
        l = rand(1:N)
        t = expr_to_tree(
            T,
            String(strip(mut_tree_options[l], [' ', '\n', '"', ',', '.', '[', ']'])),
            options,
        )
        if t.val == 1 && t.constant
            continue
        end

        llm_recorder(options.llm_options, tree_to_expr(t, options), "mutate")

        return t
    end

    out = expr_to_tree(
        T, String(strip(mut_tree_options[1], [' ', '\n', '"', ',', '.', '[', ']'])), options
    )

    llm_recorder(options.llm_options, tree_to_expr(out, options), "mutate")

    return out
end

function llm_crossover_trees(
    ex1::E, ex2::E, options::LaSROptions
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
    tree1::AbstractExpressionNode{T}, tree2::AbstractExpressionNode{T}, options::LaSROptions
)::Tuple{AbstractExpressionNode{T},AbstractExpressionNode{T}} where {T<:DATA_TYPE}
    expr1 = tree_to_expr(tree1, options)
    expr2 = tree_to_expr(tree2, options)

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
        SystemMessage(load_prompt(options.prompts_dir * "crossover_system.txt")),
        UserMessage(
            construct_prompt(
                load_prompt(options.prompts_dir * "crossover_user.txt"),
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

    llm_recorder(options.llm_options, rendered_msg, "llm_input|crossover")

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
            no_system_message=true,
            verbose=false,
        )
    catch e
        llm_recorder(options.llm_options, "None " * string(e), "crossover|failed")
        return tree1, tree2
    end

    llm_recorder(options.llm_options, string(msg.content), "llm_output|crossover")

    cross_tree_options = parse_msg_content(String(msg.content))

    cross_tree1 = nothing
    cross_tree2 = nothing

    N = min(size(cross_tree_options)[1], N)

    if N == 0
        llm_recorder(options.llm_options, "None", "crossover|failed")
        return tree1, tree2
    end

    if N == 1
        t = expr_to_tree(
            T,
            String(strip(cross_tree_options[1], [' ', '\n', '"', ',', '.', '[', ']'])),
            options,
        )

        llm_recorder(options.llm_options, tree_to_expr(t, options), "crossover")

        return t, tree2
    end

    for i in 1:(2 * N)
        l = rand(1:N)
        t = expr_to_tree(
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
        cross_tree1 = expr_to_tree(
            T,
            String(strip(cross_tree_options[1], [' ', '\n', '"', ',', '.', '[', ']'])),
            options,
        )
    end

    if isnothing(cross_tree2)
        cross_tree2 = expr_to_tree(
            T,
            String(strip(cross_tree_options[2], [' ', '\n', '"', ',', '.', '[', ']'])),
            options,
        )
    end

    recording_str =
        tree_to_expr(cross_tree1, options) * " && " * tree_to_expr(cross_tree2, options)
    llm_recorder(options.llm_options, recording_str, "crossover")

    return cross_tree1, cross_tree2
end

end
