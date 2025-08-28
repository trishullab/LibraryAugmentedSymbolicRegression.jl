module ParseModule

using DynamicExpressions
using SymbolicRegression: AbstractOptions, DATA_TYPE

"""
    parse_expr(expr_str::String, options) -> AbstractExpressionNode

Given a string (e.g., from string_tree) and an options object (containing
operators, variable naming conventions, etc.), reconstruct an
AbstractExpressionNode.
"""
function parse_expr(
    ::Type{T}, expr_str::String, options::AbstractOptions
) where {T<:DATA_TYPE}
    try
        expr_str = replace(expr_str, r"\*\*" => "^")
        expr_str = replace(expr_str, r"\*\*" => "^")
        return parse_expression(
            Meta.parse(expr_str);
            operators=options.operators,
            node_type=options.node_type,
            expression_type=options.expression_type,
            variable_names=get_variable_names(options.variable_names),
        )

        # # If it's an assignment like `a = expr`, just take the RHS
        # if parsed.head == :(=)
        #     parsed = parsed.args[2]
        # end

        # expr = _parse_expr(parsed, options, T)
        # # ensure that the expression can be rendered
        # render_expr(expr, options)
        # return expr
    catch e
        @info "Failed to parse expression: $expr_str"
        @info "Error: $e"
        @info "Returning a constant node with value 1."
        return _make_constant_node(1, T)
    end
end

"""
    render_expr(ex::AbstractExpression{T}, options::AbstractOptions) -> String

Given an AbstractExpression and an options object, return a string representation
of the expression. Specifically, replace constants with "C" and variables with
"x", "y", "z", etc or the prespecified variable names.
"""
function render_expr(
    ex::AbstractExpression{T}, options::AbstractOptions
)::String where {T<:DATA_TYPE}
    return render_expr(get_contents(ex), options)
end

function render_expr(tree::AbstractExpressionNode{T}, options)::String where {T<:DATA_TYPE}
    variable_names = get_variable_names(options.variable_names)
    return string_tree(
        tree, options.operators; f_constant=_sketch_const, variable_names=variable_names
    )
end

function get_variable_names(variable_names::Dict)::Vector{String}
    return [variable_names[key] for key in sort(collect(keys(variable_names)))]
end

function get_variable_names(variable_names::Nothing)::Vector{String}
    return ["x", "y", "z", "k", "j", "l", "m", "n", "p", "a", "b"]
end

end
