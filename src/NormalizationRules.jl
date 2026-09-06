module NormalizationRulesModule

using DispatchDoctor: @unstable

"""
    NormalizationRule(name, stage, apply)

One rewrite that makes an LLM expression string readable to the parser.

- `name` identifies the rule.
- `stage` is `:string` for a rewrite of the raw text, or `:expr` for a rewrite of the
  parsed Julia AST.
- `apply` is the rewrite. A `:string` rule maps a string to a string. An `:expr` rule
  maps an AST to an AST.
"""
struct NormalizationRule
    name::String
    stage::Symbol
    apply::Function
end

"""
    NormalizationRule(pair; name="user")

Make a `:string` rule from a `pattern => replacement` pair.
"""
function NormalizationRule(p::Pair; name::AbstractString="user")
    return NormalizationRule(String(name), :string, s -> replace(s, first(p) => last(p)))
end

@unstable apply_string_rules(rules, s::AbstractString) = foldl(
    (acc, r) -> r.apply(acc),
    Iterators.filter(r -> r.stage === :string, rules);
    init=String(s),
)
@unstable apply_expr_rules(rules, ast) = foldl(
    (acc, r) -> r.apply(acc), Iterators.filter(r -> r.stage === :expr, rules); init=ast
)

"""
    resolve_rules(defaults, user)

Put the user rules after the default rules and keep the given order. The user rules come
from `LaSRPlugin(; parse_rules=[...])`.
"""
resolve_rules(defaults, user) = vcat(defaults, user)

rule_whitespace() = NormalizationRule(r"\s+" => " "; name="whitespace")
# Two pairs in one `replace` pass: `(C)` wins over the bare `C` at the same position.
function rule_c_placeholder()
    return NormalizationRule(
        "c_placeholder",
        :string,
        s -> replace(s, r"(?<!\w)\(C\)(?!\w)" => "(1.0)", r"(?<!\w)C(?!\w)" => "1.0"),
    )
end
"""
    rule_indexed_const()

Change each indexed constant placeholder, such as `C1` or `c2`, to the literal `1.0`.

An LLM writes a multi-coefficient expression as `C1*x0 + C2*x1`. Each `Ck` is a
different constant to fit. The `c_placeholder` rule matches the bare `C` only, because
its lookahead fails on the digit. Without this rule, `case_fold` changes `C1` to `c1`,
`parse_expression` reads `c1` as an unknown variable, and the search drops the full
proposal.

The literal `1.0` becomes an independent constant node, the same target as the bare `C`
placeholder. The rule accepts both letter cases, so its position relative to `case_fold`
does not matter. A guard on the left keeps the rule out of the middle of an identifier.
No SR operator has a name that is `c` and digits.
"""
function rule_indexed_const()
    return NormalizationRule(
        r"(?<![A-Za-z0-9_])[Cc]\d+(?![A-Za-z0-9_])" => "1.0"; name="indexed_const"
    )
end
rule_pipe_abs() = NormalizationRule(r"\|([^|]+)\|" => s"abs(\1)"; name="pipe_abs")
"""
    rule_ln_log()

Change `ln` to `log`.

`ln` is the usual LLM notation for the natural logarithm, but the SR operator is `log`.
Without this rule, `parse_expression` reads `ln(x)` as an unknown function and drops the
full proposal. This occurred in every LLM-SRBench domain. The rule ignores letter case
and matches a full word only, so it never changes a part of another identifier such as
`lnk`. It is an operator-dialect rule, like `pow_star`.
"""
rule_ln_log() = NormalizationRule(r"\b[Ll][Nn]\b" => "log"; name="ln_log")
function rule_subscript_var()
    return NormalizationRule(r"([A-Za-z])_(\d+)" => s"\1\2"; name="subscript_var")
end
rule_case_fold() = NormalizationRule("case_fold", :string, lowercase)
rule_pow_star() = NormalizationRule(r"\*\*" => "^"; name="pow_star")

# ---- LaSR-specific expr rules (ported from `_rewrite_llm_ops` + `_rhs_of_assignment`) ----
_strip_lhs(ast) = ast
@unstable function _strip_lhs(ast::Expr)
    ast.head === :(=) && length(ast.args) >= 2 && return _strip_lhs(ast.args[2])
    ast.head === :block && !isempty(ast.args) && return _strip_lhs(last(ast.args))
    return ast
end
rule_strip_lhs() = NormalizationRule("strip_lhs", :expr, _strip_lhs)

_unary_and_pow(ast) = ast
@unstable function _unary_and_pow(ast::Expr)
    if ast.head === :call && length(ast.args) == 2 && ast.args[1] === :-
        return Expr(:call, :-, 0.0, _unary_and_pow(ast.args[2]))
    elseif ast.head === :call && length(ast.args) == 2 && ast.args[1] === :+
        return _unary_and_pow(ast.args[2])
    elseif ast.head === :call && length(ast.args) == 3 && ast.args[1] === :pow
        return Expr(:call, :^, _unary_and_pow(ast.args[2]), _unary_and_pow(ast.args[3]))
    else
        return Expr(ast.head, map(_unary_and_pow, ast.args)...)
    end
end
"""
    rule_unary_sign()

Rewrite the unary `+` and `-` signs, and change `pow(a, b)` to `a ^ b`. One pass does
both.
"""
rule_unary_sign() = NormalizationRule("unary_sign", :expr, _unary_and_pow)

const _TOKEN_RE = r"([A-Za-z_][A-Za-z0-9_]*|\d+\.?\d*|\*\*|[-+*/^(),]|\s+)"
const _DEFAULT_UNARY_OPS = Set(["sin", "cos", "exp", "log", "sqrt", "tan", "abs", "cbrt"])

function _tokenize(s::AbstractString)
    toks = String[]
    i = firstindex(s)
    n = lastindex(s)
    while i <= n
        m = match(_TOKEN_RE, s, i)
        if m === nothing || m.offset != i
            # Unrecognized character (shouldn't normally happen): pass it through verbatim.
            push!(toks, string(s[i]))
            i = nextind(s, i)
        else
            push!(toks, m.match)
            i = m.offset + ncodeunits(m.match)
        end
    end
    return toks
end

_is_name_token(t::AbstractString) = occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", t)

function _implicit_multiplication(s::AbstractString)
    toks = _tokenize(s)
    out = String[]
    i = 1
    n = length(toks)
    while i <= n
        push!(out, toks[i])
        if toks[i] == " " && i > 1 && i < n
            left = toks[i - 1]
            right = toks[i + 1]
            left_val = _is_name_token(left) || occursin(r"^\d+\.?\d*$", left) || left == ")"
            right_val =
                _is_name_token(right) || occursin(r"^\d+\.?\d*$", right) || right == "("
            if left_val && right_val
                pop!(out)      # drop the space
                push!(out, "*")
            end
        end
        i += 1
    end
    return join(out)
end
function rule_implicit_multiplication()
    return NormalizationRule("implicit_multiplication", :string, _implicit_multiplication)
end

function _implicit_application(s::AbstractString, ops::Set{String}=_DEFAULT_UNARY_OPS)
    toks = _tokenize(s)
    out = String[]
    i = 1
    n = length(toks)
    while i <= n
        t = toks[i]
        if _is_name_token(t) && t in ops
            # Look ahead past a single space, or the `*` inserted by implicit_multiplication
            # (since implicit_multiplication runs first, `sin x` has already become `sin*x`).
            j = i + 1
            if j <= n && (toks[j] == " " || toks[j] == "*")
                j += 1
            end
            if j <= n && toks[j] == "("
                # Already parenthesized (possibly via `name * (arg)`): just drop the separator.
                push!(out, t)
                i = j
                continue
            elseif j <= n && (_is_name_token(toks[j]) || occursin(r"^\d+\.?\d*$", toks[j]))
                # `name arg` / `name*arg` (no parens) -> `name(arg)`.
                push!(out, t, "(", toks[j], ")")
                i = j + 1
                continue
            end
        end
        push!(out, t)
        i += 1
    end
    return join(out)
end
function rule_implicit_application(ops::Set{String}=_DEFAULT_UNARY_OPS)
    return NormalizationRule(
        "implicit_application", :string, s -> _implicit_application(s, ops)
    )
end

const _FUNC_EXP_RE =
    let ops = join(sort(collect(_DEFAULT_UNARY_OPS); by=length, rev=true), "|")
        Regex("\\b(" * ops * ")\\^(\\d+)\\s*\\(?([^)\\s]+)\\)?")
    end
"""
    rule_function_exponentiation()

Change `sin^2 x` or `sin^2(x)` to `sin(x)^2`.

The rule matches the known unary operator names only. Without that limit, the pattern
matches any identifier: a variable `t` in `t^2 + v` became `t(+)^2 v`, because the
pattern took `+` as the argument of `t`. That result is fatal when the variables are
single-character domain symbols. The rule tries the longest operator names first and
matches full words only, so it never matches a part of a name.

The rule handles the common `f^n(x)` and `f^n x` forms only. Other forms pass through
with no change: nested exponents, functions with more than one argument, and `f^n` with
no operand.
"""
function rule_function_exponentiation()
    return NormalizationRule(_FUNC_EXP_RE => s"\1(\3)^\2"; name="function_exponentiation")
end

const DEFAULT_RULES = NormalizationRule[
    rule_whitespace(),
    rule_subscript_var(),
    rule_c_placeholder(),
    rule_indexed_const(),
    rule_pipe_abs(),
    rule_case_fold(),
    rule_ln_log(),
    rule_implicit_multiplication(),
    rule_implicit_application(),
    rule_function_exponentiation(),
    rule_pow_star(),
    rule_strip_lhs(),
    rule_unary_sign(),
]

export NormalizationRule,
    apply_string_rules,
    apply_expr_rules,
    rule_implicit_multiplication,
    rule_implicit_application,
    rule_function_exponentiation,
    DEFAULT_RULES,
    resolve_rules

end # module
