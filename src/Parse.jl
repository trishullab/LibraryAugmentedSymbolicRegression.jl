module ParseModule

using DispatchDoctor: @unstable
using DynamicExpressions
using .DynamicExpressions.NodeModule: Node
using SymbolicRegression: AbstractOptions, DATA_TYPE

"""
    parse_expr(expr_str::String, options) -> AbstractExpressionNode

Given a string (e.g., from `string_tree`) and an options object (containing
operators, variable naming conventions, etc.), reconstruct an
AbstractExpressionNode.
"""
function parse_expr(
    expr_str::String, options::AbstractOptions, ::Type{T}
) where {T<:DATA_TYPE}
    parsed = Meta.parse(expr_str)
    return _parse_expr(parsed, options, T)
end

function _make_constant_node(val::Number, ::Type{T}) where {T<:DATA_TYPE}
    return Node{T}(; val=convert(T, val))
end

function _make_variable_node(sym::Symbol, ::Type{T}) where {T<:DATA_TYPE}
    if sym === :C
        return Node{T}(; val=convert(T, 1))
    end

    local idx = parse(Int, String(sym)[2:end])  # e.g. x5 => 5
    return Node{T}(; feature=idx)
end

function _make_call_node(ex::Expr, options::AbstractOptions, ::Type{T}) where {T<:DATA_TYPE}
    op_sym = ex.args[1]
    argexprs = ex.args[2:end]
    children = map(child_ex -> _parse_expr(child_ex, options, T), argexprs)
    op_idx = _find_operator_index(op_sym, options)
    if length(children) == 1
        return Node{T}(; op=op_idx, l=children[1])
    elseif length(children) == 2
        return Node{T}(; op=op_idx, l=children[1], r=children[2])
    else
        error("Operator with $(length(children)) children not supported.")
    end
end

function _render_function(
    fn::F, function_str_map::Dict{S,S}
)::String where {F<:Function,S<:AbstractString}
    fn_str = replace(string(fn), "safe_" => "")
    if haskey(function_str_map, fn_str)
        return function_str_map[fn_str]
    end
    return fn_str
end

@unstable function _find_operator_index(op_sym, options::AbstractOptions)
    function_str_map = Dict("pow" => "^")
    binops = map((x) -> _render_function(x, function_str_map), options.operators.binops)
    unaops = map((x) -> _render_function(x, function_str_map), options.operators.unaops)

    for (i, opfunc) in pairs(binops)
        if opfunc == string(op_sym)
            return UInt8(i)
        end
    end

    for (i, opfunc) in pairs(unaops)
        if opfunc == string(op_sym)
            return UInt8(i)
        end
    end

    return error("Unrecognized operator symbol: $op_sym")
end

@unstable function _parse_expr(ex, options::AbstractOptions, ::Type{T}) where {T<:DATA_TYPE}
    if ex isa Number
        return _make_constant_node(ex, T)
    elseif ex isa Symbol
        return _make_variable_node(ex, T)
    elseif ex isa Expr
        if ex.head === :call
            return _make_call_node(ex, options, T)
        elseif ex.head === :negative
            # If we see something like -(3.14),
            # parse it as (0 - 3.14).
            return _parse_expr(Expr(:call, :-, 0, ex.args[1]), options, T)
        else
            error("Unsupported expression head: $(ex.head)")
        end
    else
        error("Unsupported expression: $(ex)")
    end
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
