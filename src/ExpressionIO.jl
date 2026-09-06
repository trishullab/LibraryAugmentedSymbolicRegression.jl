module ExpressionIOModule

using DispatchDoctor: @unstable
using DynamicExpressions
using DynamicExpressions.NodeModule: Node
using SymbolicRegression: AbstractOptions, DATA_TYPE
using ..PluginModule: lasr_context, LaSRPluginState
using ..NormalizationRulesModule:
    apply_string_rules, apply_expr_rules, DEFAULT_RULES, resolve_rules
using ..ParseFailuresModule: ParseFailure, record_parse_failure!

@unstable function _parse_fallback(
    options, expr_str, expr_str_norm, stage::Symbol, e, node_type, ::Type{T}
) where {T}
    @warn "LaSR parse fallback ($stage): returning constant 1 for: $expr_str"
    @warn "Error: $e"
    # Record the fallback in the active plugin state, when there is one, so a scientist can
    # see which LLM strings stop the parser and add a `NormalizationRule` for them.
    state = getfield(options, :state)
    if state isa LaSRPluginState
        record_parse_failure!(
            state.parse_failures,
            ParseFailure(String(expr_str), String(expr_str_norm), stage, string(e)),
        )
    end
    return Expression(
        node_type(; val=convert(T, 1.0)); options.operators, options.variable_names
    )
end

"""
    parse_expr(::Type{T}, expr_str::String, options) -> AbstractExpression{T}

Read an expression string and build an expression tree.

`expr_str` is the text that an LLM sent, or the output of `render_expr`. `options` is a
`LaSRContext` or a plain `Options`. It supplies the operators, the variable names, and
the normalization rules.

The normalization rules run first, then Julia reads the result. If a step fails, this
function records a `ParseFailure` and returns a constant-1 tree, so the search continues.
"""
@unstable function parse_expr(
    ::Type{T}, expr_str::String, options::AbstractOptions
)::AbstractExpression{T} where {T<:DATA_TYPE}
    # `options` here may be a raw `SymbolicRegression.Options` or already a
    # `LaSRContext` (every real call site in the LLM modules converts before calling
    # `parse_expr`).
    options = lasr_context(options)
    rules = resolve_rules(DEFAULT_RULES, options.plugin.parse_rules)
    node_type = options.node_type{T}::Type{<:AbstractExpressionNode{T}}
    expression_type = options.expression_type{T,node_type}::Type{<:AbstractExpression{T}}
    ops = options.operators
    varnames = get_variable_names(options.variable_names)
    expr_str_norm = apply_string_rules(rules, expr_str)

    local ast
    try
        ast = Meta.parse(expr_str_norm)
    catch e
        return _parse_fallback(
            options, expr_str, expr_str_norm, :meta_parse, e, node_type, T
        )
    end

    try
        ast = apply_expr_rules(rules, ast)
    catch e
        return _parse_fallback(
            options, expr_str, expr_str_norm, :expr_stage, e, node_type, T
        )
    end

    try
        return parse_expression(
            ast;
            operators=ops,
            node_type=node_type,
            expression_type=expression_type,
            variable_names=varnames,
        )::AbstractExpression{T}
    catch e
        return _parse_fallback(
            options, expr_str, expr_str_norm, :tree_parse, e, node_type, T
        )
    end
end

"""
    render_expr(ex::AbstractExpression{T}, options::AbstractOptions) -> String

Write an expression as a string for a prompt.

Each constant becomes `C`. Each variable becomes its configured name, or `x`, `y`, `z`
and so on when `options` configures no names.
"""
function render_expr(
    ex::AbstractExpression{T}, options::AbstractOptions
)::String where {T<:DATA_TYPE}
    return render_expr(get_contents(ex), options)
end

function _sketch_const(val)
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
    options = lasr_context(options)
    variable_names = get_variable_names(options.variable_names)
    return string_tree(
        tree, options.operators; f_constant=_sketch_const, variable_names=variable_names
    )
end

function get_variable_names(variable_names::Dict)::Vector{String}
    isempty(variable_names) && return get_variable_names(nothing)
    return [variable_names[key] for key in sort(collect(keys(variable_names)))]
end

function get_variable_names(variable_names::Nothing)::Vector{String}
    return ["x", "y", "z", "k", "j", "l", "m", "n", "p", "a", "b"]
end

end
