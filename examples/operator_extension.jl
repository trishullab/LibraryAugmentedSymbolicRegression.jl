# Worked example: adding a new operator to LaSR's search vocabulary (factorial).
#
# Extending a symbolic-regression search with a new operator that matches a scientist's
# domain requires two operations in LaSR:
#
#   1. THE OPERATOR ITSELF -- a `unary_operators`/`binary_operators` entry must be a
#      plain Julia function SR can *evaluate* on the search's numeric type
#      (Float32/Float64) and, ideally, *differentiate*: SR fits constants with
#      gradient-based optimization, so an operator built out of smooth elementary
#      functions (`+`, `*`, `exp`, `log`, `sqrt`, ...) composes into the search's autodiff
#      machinery for free, the same way `sin`/`cos`/`abs` do. See `safe_factorial` below.
#
#   2. THE PARSE-SIDE VOCABULARY -- a scientist (or an LLM) writes equations in their own
#      domain notation, here `x!` for factorial, which is not valid Julia syntax and is
#      not, by itself, anything `Options`/`Meta.parse` knows how to read. `LaSRPlugin`'s
#      `parse_rules` extension point (see `src/NormalizationRules.jl`'s `resolve_rules`) lets a
#      scientist register a string-level `NormalizationRule` that rewrites their notation
#      into the registered operator's call syntax *before* `parse_expr` hands the string
#      to `Meta.parse`. Without this half, `x0!` fails `Meta.parse` and `parse_expr`
#      silently falls back to a constant-1 node (recorded in the plugin's
#      `ParseFailureStore` -- see `src/ParseFailures.jl`) even though `safe_factorial` is a
#      perfectly good operator once its call syntax is spelled out.
#
# The wiring below is exercised (without a model server) by
# `test/test_lasr_operator_extension.jl`.
#
# Run: `julia --project=. examples/operator_extension.jl`

using SymbolicRegression: Options, string_tree, eval_tree_array
using LibraryAugmentedSymbolicRegression: LaSRPlugin, NormalizationRule, parse_expr

# ---- 1. A differentiable operator ---------------------------------------------------
#
# `safe_factorial(x)` stands in for `x!` = `gamma(x + 1)`. A real deployment that already
# depends on `SpecialFunctions` would just write `safe_factorial(x) = gamma(x + 1)`
# (exact, well-tested, and differentiable via `SpecialFunctions`'s own `ChainRules`
# integration). `SpecialFunctions` is only a *transitive* dependency of this package here
# (pulled in by `SymbolicRegression`/`Optim` -- see `Manifest.toml`), not a direct
# dependency listed in `Project.toml`, so per this project's no-new-dependency policy this
# example does not `using SpecialFunctions`. Instead it builds a closed-form,
# differentiable approximation of `gamma(x + 1)` out of elementary functions already
# available (`log`, `exp`, arithmetic): Ramanujan's approximation of `ln(n!)`,
#
#     ln(n!) ~= n*ln(n) - n + (1/6)*ln(n*(1 + 4n*(1 + 2n))) + (1/2)*ln(pi)
#
# TRADEOFF: this is an *approximation*, not an exact factorial -- accurate to within
# ~1e-4 relative error for n >= 1 (the error term is O(1/n^3)), but it drifts near n = 0
# (`safe_factorial(0) ~= 0.056`, not the exact `0! = 1`) since the asymptotic expansion
# breaks down there. A production operator on a domain where the pole at 0 matters would
# import `SpecialFunctions.gamma`/`loggamma` directly instead of approximating it.
# `abs(x)` keeps the domain defined for the negative inputs SR's mutation/constant
# optimization can generate mid-search; the `1e-9` nudge keeps `log` away from its pole at
# exactly `n = 0`.
function safe_factorial(x::T) where {T<:Real}
    n = abs(x) + T(1e-9)
    ln_n_factorial = n * log(n) - n + log(n * (1 + 4n * (1 + 2n))) / 6 + log(T(pi)) / 2
    return exp(ln_n_factorial)
end

println("safe_factorial(4) = ", safe_factorial(4.0), "  (exact 4! = 24)")

# ---- 2 & 3. Register the operator AND the parse rule on the search ------------------
#
# `unary_operators` makes `safe_factorial` something the search can place in a tree and
# evaluate. `LaSRPlugin(; parse_rules=[...])` makes `x!` -- the notation a scientist or an
# LLM actually writes -- resolve to a call on that operator when `parse_expr` normalizes
# an equation string: `src/NormalizationRules.jl`'s `resolve_rules(DEFAULT_RULES,
# plugin.parse_rules)` appends scientist-registered rules after the built-ins, so `x!`
# gets rewritten to `safe_factorial(x)` before `Meta.parse` ever sees the string.
# `use_llm=false` keeps this example fully deterministic and server-free.
options = Options(;
    binary_operators=[+, -, *, /],
    unary_operators=[safe_factorial, sin, cos],
    plugins=(
        LaSRPlugin(;
            use_llm=false,
            variable_names=Dict("x0" => "x0"),
            parse_rules=[NormalizationRule(r"(\w+)!" => s"safe_factorial(\1)")],
        ),
    ),
)

# `x0!` is not valid Julia -- without the parse rule above, `Meta.parse` would throw and
# `parse_expr` would fall back to a constant-1 node instead of the factorial operator.
ex = parse_expr(Float64, "x0!", options)
println("parse_expr(\"x0!\", options) -> ", string_tree(ex, options))

X = reshape([0.0, 1.0, 2.0, 3.0, 4.0], 1, :)
out, ok = eval_tree_array(ex, X, options)
println("evaluated on x0 = ", vec(X), " -> ", out, "  (ok = $ok)")
@assert ok
@assert isapprox(out, safe_factorial.(vec(X)); atol=1e-9)

println(
    "Operator-extension wiring OK: `x0!` parses to safe_factorial(x0) and evaluates correctly.",
)
