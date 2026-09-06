module PromptsModule

using Random: randperm
using SymbolicRegression: AbstractOptions
using DispatchDoctor: @unstable
using ..PluginModule: lasr_context, default_prompts_dir
using ..ExpressionIOModule: render_expr, get_variable_names

"""
    prompt_path(prompts_dir, name) -> String

Resolve one `.prompt` template: `prompts_dir` first, then the packaged templates.
"""
function prompt_path(prompts_dir::AbstractString, name::AbstractString)::String
    path = normpath(joinpath(prompts_dir, name))
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

Copy the packaged `.prompt` templates into `dest` as writable files, and return `dest`.
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

"""
    convertDict(d) -> NamedTuple

Convert a `Dict` of template variables into a `NamedTuple` for the prompt renderer.
"""
@unstable function convertDict(d)::NamedTuple
    return (; (Symbol(k) => v for (k, v) in d)...)
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
    construct_prompt(user_prompt, element_list, element_id_tag) -> String

Replace each tag line in `user_prompt` with one element of `element_list`.
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
