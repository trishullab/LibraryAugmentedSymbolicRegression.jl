module LLMUtilsModule

using Random: rand, randperm
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
using SymbolicRegression: DATA_TYPE
using ..LLMOptionsModule: LaSROptions
using ..ParseModule: render_expr
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

function format_pareto(dominating, options, num_pareto_context::Int)::Vector{String}
    pareto = Vector{String}()
    if !isnothing(dominating) && size(dominating)[1] > 0
        idx = randperm(size(dominating)[1])
        for i in 1:min(size(dominating)[1], num_pareto_context)
            push!(pareto, render_expr(dominating[idx[i]].tree, options))
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

end
