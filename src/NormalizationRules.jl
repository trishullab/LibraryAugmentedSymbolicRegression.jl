module NormalizationRulesModule

using DispatchDoctor: @unstable

struct NormalizationRule
    name::String
    stage::Symbol            # :string or :expr
    apply::Function          # (::AbstractString)->AbstractString  |  (ast)->ast
end
# Convenience: a string-stage regex/string rewrite.
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

# Scientist-registerable extension point: append user-supplied rules (e.g. from
# `LaSRPlugin(; parse_rules=[...])`) after the built-in defaults, order preserved.
resolve_rules(defaults, user) = vcat(defaults, user)

# ---- LaSR-specific string rules (ported from the former Parse.jl `_normalize_expr_string`) ----
rule_whitespace() = NormalizationRule("whitespace", :string, s -> replace(s, r"\s+" => " "))
function rule_c_placeholder()
    return NormalizationRule(
        "c_placeholder",
        :string,
        s -> replace(
            replace(s, r"(?<!\w)\(C\)(?!\w)" => "(1.0)"), r"(?<!\w)C(?!\w)" => "1.0"
        ),
    )
end
# Indexed constant placeholders `C1`, `C2`, ... (and their lowercase forms): LLMs emit
# multi-coefficient expressions like `C1*x0 + C2*x1` where each `Ck` denotes a distinct
# constant to fit. `c_placeholder` only matches the *bare* `C` (its `(?!\w)` lookahead
# fails on the trailing digit), so without this rule `case_fold` would lower `C1`->`c1`,
# which `parse_expression` rejects as an unknown variable -- dropping the whole proposal.
# Map each `Ck` to the numeric literal `1.0`, which becomes an independent optimizable
# constant node (the same target as the bare-`C` placeholder). `[Cc]` covers both casings
# regardless of order relative to `case_fold`; the leading `(?<![A-Za-z0-9_])` guard keeps
# it from biting inside identifiers, and no SR operator name is `c`+digits.
function rule_indexed_const()
    return NormalizationRule(
        "indexed_const",
        :string,
        s -> replace(s, r"(?<![A-Za-z0-9_])[Cc]\d+(?![A-Za-z0-9_])" => "1.0"),
    )
end
function rule_pipe_abs()
    return NormalizationRule(
        "pipe_abs", :string, s -> replace(s, r"\|([^|]+)\|" => s"abs(\1)")
    )
end
# `ln` is the near-universal LLM/math notation for natural log; SR's operator is `log`. Without
# this, `ln(x)` reaches `parse_expression` as an unknown function and the whole proposal is
# dropped (observed across every LLM-SRBench domain). Case-insensitive, word-bounded so it never
# bites inside another identifier (e.g. a variable `lnk`). An operator-dialect rule (like pow_star).
function rule_ln_log()
    return NormalizationRule("ln_log", :string, s -> replace(s, r"\b[Ll][Nn]\b" => "log"))
end
function rule_subscript_var()
    return NormalizationRule(
        "subscript_var", :string, s -> replace(s, r"([A-Za-z])_(\d+)" => s"\1\2")
    )
end
rule_case_fold() = NormalizationRule("case_fold", :string, s -> lowercase(s))
rule_pow_star() = NormalizationRule("pow_star", :string, s -> replace(s, r"\*\*" => "^"))

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
# One pass handles unary `+`/`-` and `pow(a, b)` -> `a ^ b` together.
rule_unary_sign() = NormalizationRule("unary_sign", :expr, _unary_and_pow)

# ---- Inherited SymPy rules (implicit multiplication / application / function exponentiation) ----
# Julia's `Meta.parse` already handles *numeric* juxtaposition (`2x`, `2sin(x)`), so these
# rules only need to target what it rejects: identifier-identifier juxtaposition
# (`x y` -> `x*y`) and function-name-without-parens (`sin x` -> `sin(x)`).

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

# Handles the common `f^n(x)` / `f^n x` form only (`cos^2(t)` -> `cos(t)^2`).
# Exotic forms (nested exponents, multi-arg functions, `f^n` with no operand) are out of
# scope for this pass; they fall through unchanged.
# `sin^2 x` / `sin^2(x)` -> `sin(x)^2`. Restricted to KNOWN function names (the unary-operator
# set) so it never treats a VARIABLE as a function: without this restriction the old
# `([A-Za-z]+)` matched any identifier, so a variable like `t` in `t^2 + v` was rewritten to
# `t(+)^2 v` (grabbing `+` as `t`'s "argument") -- catastrophic once variables are single-char
# domain symbols. Longest-first alternation + word boundary avoid partial-name matches.
const _FUNC_EXP_RE =
    let ops = join(sort(collect(_DEFAULT_UNARY_OPS); by=length, rev=true), "|")
        Regex("\\b(" * ops * ")\\^(\\d+)\\s*\\(?([^)\\s]+)\\)?")
    end
_function_exponentiation(s::AbstractString) = replace(s, _FUNC_EXP_RE => s"\1(\3)^\2")
function rule_function_exponentiation()
    return NormalizationRule("function_exponentiation", :string, _function_exponentiation)
end

# ---- Combined default pipeline used by `parse_expr` ----
# String rules run in this exact order (order matters):
#   whitespace, subscript_var, c_placeholder, indexed_const, pipe_abs,
#   case_fold, ln_log, implicit_multiplication, implicit_application,
#   function_exponentiation, pow_star
# then expr rules: strip_lhs, unary_sign.
# `subscript_var` runs before `case_fold` so subscript stripping sees the original casing;
# `c_placeholder` also runs before `case_fold` since it matches the *uppercase* `C`
# placeholder specifically (a lowercased `c` is an ordinary variable name, not a constant
# placeholder).
# `case_fold`/lowercasing runs *before* the operator-name rules (`implicit_multiplication`,
# `implicit_application`, `function_exponentiation`) so a mixed-case function written
# without parens (e.g. an LLM's `Sin x`) is already lowercase (`sin x`) by the time
# `implicit_application` looks it up in its (lowercase) operator-name set -- otherwise
# `implicit_application` never recognizes `Sin` and the call never gets wrapped in
# parens. Folding case up front still guarantees the operator-name symbols Julia parses
# (e.g. `:pow`, `:sin`) are lowercase by the time `Meta.parse` runs, regardless of how an
# LLM (or the test fuzzer) capitalized them.
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
