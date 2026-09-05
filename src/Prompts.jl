module PromptsModule

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
using SymbolicRegression: DATA_TYPE, AbstractOptions
using DispatchDoctor: @unstable
using ..PluginModule: lasr_context, default_prompts_dir
using ..ExpressionIOModule: render_expr, get_variable_names
using JSON: parse

"""
    prompt_path(prompts_dir, name)

Path of prompt template `name` for a plugin configured with `prompts_dir`. A
template present in `prompts_dir` wins; anything the user did not override falls
back to the packaged copy, so a custom directory may hold just the one template it
changes. `prompts_dir` is *joined* with `name`, so a trailing separator is optional
(concatenating it silently produced `.../my_promptsmutate_user.prompt` before).
"""
function prompt_path(prompts_dir::AbstractString, name::AbstractString)::String
    path = joinpath(prompts_dir, name)
    isfile(path) && return path
    packaged = joinpath(default_prompts_dir(), name)
    isfile(packaged) && return packaged
    return throw(
        ArgumentError(
            "prompt template \"$name\" not found in $prompts_dir, and no packaged " *
            "template of that name exists in $(default_prompts_dir())",
        ),
    )
end

"""
    copy_prompts(dest; force=false)

Copy the packaged `.prompt` templates into `dest` (created if needed) as *writable*
files, and return `dest`. The installed originals are read-only (mode 444 after
`Pkg.add`) and are replaced wholesale on upgrade, so editing prompts means editing a
copy and passing it as `prompts_dir`. Existing files in `dest` are left alone unless
`force=true`, so re-running this never discards edits.
"""
function copy_prompts(dest::AbstractString; force::Bool=false)::String
    target = normpath(abspath(expanduser(String(dest))))
    packaged = default_prompts_dir()
    mkpath(target)
    for name in readdir(packaged)
        endswith(name, ".prompt") || continue
        out = joinpath(target, name)
        (isfile(out) && !force) && continue
        cp(joinpath(packaged, name), out; force=true)
        chmod(out, 0o644)
    end
    return target
end

function load_prompt(path::String)::String
    return String(strip(read(path, String)))
end

# A NamedTuple built from a runtime `Dict` has a value-dependent concrete type, so its
# return type is genuinely `NamedTuple` (abstract). Mark `@unstable` — this is a per-call
# argument-marshalling helper, so the inference boundary here is harmless.
@unstable function convertDict(d)::NamedTuple
    return (; Dict(Symbol(k) => v for (k, v) in d)...)
end

function get_vars(options::AbstractOptions)::String
    options = lasr_context(options)
    variable_names = get_variable_names(options.variable_names)
    return join(variable_names, ", ")
end

function get_ops(options::AbstractOptions)::String
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
    # Remove all None elements from the element_list
    element_list = filter(x -> x != "None", element_list)
    # Split the user prompt into lines
    lines = split(user_prompt, r"\n|\r\n")

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

    output_lines = String[]
    idx = 1
    for line in lines
        # if the line matches the pattern
        if occursin(pattern, line)
            if idx > length(element_list)
                continue
            end
            # replace the element_id_tag with the corresponding element
            push!(
                output_lines,
                replace(line, r"\{\{" * element_id_tag * r"\d+\}\}" => element_list[idx]),
            )
            idx += 1
        else
            push!(output_lines, line)
        end
    end
    return join(output_lines, "\n")
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

end
