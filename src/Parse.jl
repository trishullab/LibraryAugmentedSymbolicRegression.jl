module ParseModule

using DispatchDoctor: @unstable
using DynamicExpressions
using .DynamicExpressions.NodeModule: Node
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
        parsed = Meta.parse(expr_str)

        # If it's an assignment like `a = expr`, just take the RHS
        if parsed.head == :(=)
            parsed = parsed.args[2]
        end

        expr = _parse_expr(parsed, options, T)
        # ensure that the expression can be rendered
        render_expr(expr, options)
        return expr
    catch e
        @info "Failed to parse expression: $expr_str"
        @info "Error: $e"
        @info "Returning a constant node with value 1."
        return _make_constant_node(1, T)
    end
end

function _make_constant_node(val::Number, ::Type{T}) where {T<:DATA_TYPE}
    return Node{T}(; val=convert(T, val))
end

function _make_variable_node(
    sym::Symbol, options::AbstractOptions, ::Type{T}
) where {T<:DATA_TYPE}
    str_sym = string(sym)
    # if C or C1 or c1, basically capital or lower c followed by number
    # use regex
    if occursin(r"^c\d*$", lowercase(str_sym))
        return Node{T}(; val=convert(T, 1))
    end
    # Get the list of variable names from options.
    var_names = get_variable_names(options.variable_names)
    idx = findfirst(x -> x == str_sym, var_names)
    if idx === nothing
        # Fallback: try to interpret as the previous "x5" format.
        try
            idx = parse(Int, str_sym[2:end])
        catch e
            error("Unrecognized variable symbol: $sym")
        end
    end
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
    elseif length(children) > 2
        # Reduce an n-ary operator to a binary tree (left-associative)
        node = Node{T}(; op=op_idx, l=children[1], r=children[2])
        for child in children[3:end]
            node = Node{T}(; op=op_idx, l=node, r=child)
        end
        return node
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
    binops = map(
        x -> lowercase(_render_function(x, function_str_map)), options.operators.binops
    )
    unaops = map(
        x -> lowercase(_render_function(x, function_str_map)), options.operators.unaops
    )

    op = lowercase(string(op_sym))

    if op == "^"
        # direct hit
        if (i = findfirst(==("^"), binops)) !== nothing
            return UInt8(i)
        end
        # fallback: anything that *looks like* pow
        if (i = findfirst(n -> occursin("pow", n) || occursin("power", n), binops)) !==
            nothing
            return UInt8(i)
        end
        error("Could not find a power operator in options.operators.binops")
    end

    if (i = findfirst(==(op), binops)) !== nothing
        return UInt8(i)
    end
    if (i = findfirst(==(op), unaops)) !== nothing
        return UInt8(i)
    end

    return error("Unrecognized operator symbol: $(op_sym)")
end
function _parse_expr(ex, options::AbstractOptions, ::Type{T}) where {T<:DATA_TYPE}
    if ex isa Number
        return _make_constant_node(ex, T)
    elseif ex isa Symbol
        return _make_variable_node(ex, options, T)
    elseif ex isa Expr
        if ex.head === :call
            op = ex.args[1]
            if op === :- && length(ex.args) == 2
                child = ex.args[2]
                if child isa Number
                    return _make_constant_node(-child, T)
                end
                # general case: 0 - child (use existing binary "-")
                minus_idx = _find_operator_index(:-, options)  # must return BINOP index
                return Node{T}(;
                    op=minus_idx,
                    l=_make_constant_node(zero(T), T),
                    r=_parse_expr(child, options, T),
                )
            end
            if op === :+ && length(ex.args) == 2
                return _parse_expr(ex.args[2], options, T)  # unary plus is a no-op
            end
            return _make_call_node(ex, options, T)
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
