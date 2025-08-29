module ParseModule

using DispatchDoctor: @unstable
using DynamicExpressions
using DynamicExpressions.NodeModule: Node
using SymbolicRegression: AbstractOptions, DATA_TYPE

"""
    parse_expr(expr_str::String, options)

Given a string (e.g., from string_tree) and an options object (containing
operators, variable naming conventions, etc.), reconstruct an
AbstractExpressionNode.
"""
@unstable function parse_expr(
    ::Type{T}, expr_str::String, options::AbstractOptions
)::AbstractExpression{T} where {T<:DATA_TYPE}
    node_type = options.node_type{T}::Type{<:AbstractExpressionNode{T}}
    expression_type = options.expression_type{T,node_type}::Type{<:AbstractExpression{T}}
    ops = options.operators
    varnames = get_variable_names(options.variable_names)
    expr_str_norm = _normalize_expr_string(expr_str)

    local ast
    try
        ast = Meta.parse(expr_str_norm)
    catch e
        if occursin(r"^\s*[^=\n]+=", expr_str_norm)
            stripped = replace(expr_str_norm, r"^\s*[^=\n]+=\s*" => "")
            try
                ast = Meta.parse(stripped)
            catch
                @warn "Failed to Meta.parse even after stripping LHS: $expr_str"
                @warn "Error: $e"
                @warn "Returning a constant node with value 1."
                return Expression(
                    node_type(; val=convert(T, 1.0));
                    options.operators,
                    options.variable_names,
                )
            end
        else
            @warn "Failed to Meta.parse: $expr_str"
            @warn "Error: $e"
            @warn "Returning a constant node with value 1."
            return Expression(
                node_type(; val=convert(T, 1.0)); options.operators, options.variable_names
            )
        end
    end

    ast = _rhs_of_assignment(ast)

    try
        return parse_expression(
            ast;
            operators=ops,
            node_type=node_type,
            expression_type=expression_type,
            variable_names=varnames,
        )::AbstractExpression{T}
    catch e
        @warn "Failed to parse expression: $expr_str"
        @warn "Normalized: $expr_str_norm"
        @warn "Error: $e"
        @warn "Returning a constant node with value 1."
        return Expression(
            node_type(; val=convert(T, 1.0)); options.operators, options.variable_names
        )
    end
end

@unstable function _rhs_of_assignment(ast::Expr)
    if ast isa Expr
        if ast.head === :(=) && length(ast.args) ≥ 2
            return ast.args[2]
        elseif ast.head === :block && !isempty(ast.args)
            return _rhs_of_assignment(last(ast.args))
        end
    end
    return ast
end

function _normalize_expr_string(s::AbstractString)
    # Normalize whitespace
    s = replace(s, r"\s+" => " ")
    # Replace standalone C or (C) with 1.0 or (1.0)
    s = replace(s, r"(?<!\w)C(?!\w)" => "1.0")
    s = replace(s, r"(?<!\w)\(C\)(?!\w)" => "(1.0)")
    # TODO: Right now, making all variables lowercase. This might not always be desired.
    s = lowercase(s)
    s = replace(s, r"\*\*" => "^")
    strip(s)
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

@unstable function _sketch_const(val)
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
