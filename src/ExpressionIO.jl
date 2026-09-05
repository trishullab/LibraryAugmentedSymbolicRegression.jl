module ExpressionIOModule

using DispatchDoctor: @unstable
using DynamicExpressions
using DynamicExpressions.NodeModule: Node
using SymbolicRegression: AbstractOptions, DATA_TYPE
using ..PluginModule: lasr_context, LaSRContext, LaSRPluginState
using ..NormalizationRulesModule:
    apply_string_rules, apply_expr_rules, DEFAULT_RULES, resolve_rules
using ..ParseFailuresModule: ParseFailure, record_parse_failure!

# Record a constant-1 fallback into the active `LaSRPluginState`'s `ParseFailureStore`,
# if one is present. `ctx.state` is `nothing` for a `LaSRContext` built directly from a
# bare `Options` (e.g. a parser unit test with no plugin state) -- skip recording rather
# than error, since the fallback itself must always succeed regardless of observability.
function _record_parse_failure!(
    ctx::LaSRContext, expr_str::AbstractString, expr_str_norm::AbstractString,
    stage::Symbol, reason::AbstractString,
)
    state = getfield(ctx, :state)
    state isa LaSRPluginState || return nothing
    record_parse_failure!(
        state.parse_failures, ParseFailure(String(expr_str), String(expr_str_norm), stage, String(reason))
    )
    return nothing
end

@unstable function _parse_fallback(
    options, expr_str, expr_str_norm, stage::Symbol, e, node_type, ::Type{T}
) where {T}
    @warn "LaSR parse fallback ($stage): returning constant 1 for: $expr_str"
    @warn "Error: $e"
    _record_parse_failure!(options, expr_str, expr_str_norm, stage, string(e))
    return Expression(node_type(; val=convert(T, 1.0)); options.operators, options.variable_names)
end

"""
    parse_expr(expr_str::String, options)

Given a string (e.g., from string_tree) and an options object (containing
operators, variable naming conventions, etc.), reconstruct an
AbstractExpressionNode.
"""
@unstable function parse_expr(
    ::Type{T}, expr_str::String, options::AbstractOptions
)::AbstractExpression{T} where {T<:DATA_TYPE}
    # `options` here may be a raw `SymbolicRegression.Options` or already a
    # `LaSRContext` (every real call site in the LLM modules converts before calling
    # `parse_expr`). `lasr_context` is idempotent on an existing `LaSRContext` (returns
    # it unchanged) and is the one function that reaches the active `LaSRPlugin`
    # regardless of which form `options` arrives in, so resolve through it instead of
    # gating on `applicable(lasr_plugin, options)` against the pre-conversion argument.
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
        if occursin(r"^\s*[^=\n]+=", expr_str_norm)
            stripped = replace(expr_str_norm, r"^\s*[^=\n]+=\s*" => "")
            try
                ast = Meta.parse(stripped)
            catch
                return _parse_fallback(
                    options, expr_str, expr_str_norm, :meta_parse, e, node_type, T
                )
            end
        else
            return _parse_fallback(
                options, expr_str, expr_str_norm, :meta_parse, e, node_type, T
            )
        end
    end

    try
        # LLMs emit operator idioms the operator enum does not carry (unary `-`/`+`,
        # `pow(a, b)`), and the raw AST may still carry an LHS assignment (`y = ...`).
        # Left as-is these trees fail to parse and are discarded and a constant node
        # substituted; rewrite them into equivalent registered forms first.
        ast = apply_expr_rules(rules, ast)
    catch e
        return _parse_fallback(options, expr_str, expr_str_norm, :expr_stage, e, node_type, T)
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
        return _parse_fallback(options, expr_str, expr_str_norm, :tree_parse, e, node_type, T)
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
    # An empty Dict must fall back to the defaults. Returning an empty name list here
    # makes every parse fail with "Variable `x` not found in `variable_names`", which
    # silently discards *every* LLM suggestion rather than surfacing an error.
    isempty(variable_names) && return get_variable_names(nothing)
    return [variable_names[key] for key in sort(collect(keys(variable_names)))]
end

function get_variable_names(variable_names::Nothing)::Vector{String}
    return ["x", "y", "z", "k", "j", "l", "m", "n", "p", "a", "b"]
end

end
