# Example 2: adding a new operator to LaSR's search vocabulary.
# Run (after `julia --project=examples -e 'using Pkg; Pkg.instantiate()'`):
#   julia --project=examples examples/example_2_operator_extension.jl
#
# Brief:  Many scientific domains have their own specialized operations / notations. LaSR lets you extend the search operators vocabulary to add these domain-specific operations. Adding an operator is always a two-step process.
# Step 1: Define the operator with a julia function.
# Step 2: Define a rule on how to parse the operator from a string.
# This example adds a factorial operator to the search vocabulary. The factorial operator is not a built-in Julia operator, so we need to define it ourselves. We also need to define a rule on how to parse the operator from a string. Along the way, we'll touch on best practices for defining new operators in LaSR, including how to make them differentiable and how to handle parse failures gracefully.

using SymbolicRegression: Options, string_tree, eval_tree_array
using LibraryAugmentedSymbolicRegression: LaSRPlugin, NormalizationRule, parse_expr

# Step 1. Differentiable operator definition
#
# NOTE: SymbolicRegression.jl's search is not limited to differentiable operators, but differentiability is a nice property to have for the search's internal optimization routines (e.g., gradient descent on constants) and for downstream analysis (e.g., sensitivity analysis, uncertainty quantification, etc.). We'll pursue this "harder" path here, but if you just want to add a non-differentiable operator, you can skip the approximation and just define a function that computes the factorial exactly (e.g., `factorial(x::Int)`).
#
# `safe_factorial(x)` builds a closed-form, differentiable approximation of factorial out of elementary functions already available (`log`, `exp`, arithmetic). This follows Ramanujan's approximation of `ln(n!)`,
#
#     ln(n!) ~= n*ln(n) - n + (1/6)*ln(n*(1 + 4n*(1 + 2n))) + (1/2)*ln(pi)
#
# TRADEOFF: this is an *approximation*, not an exact factorial -- accurate to within
# ~1e-4 relative error for n >= 1 (the error term is O(1/n^3)), but it drifts near n = 0
# (`safe_factorial(0) ~= 0.056`, not the exact `0! = 1`) since the asymptotic expansion
# breaks down there.
function safe_factorial(x::T) where {T<:Real}
    n = abs(x) + T(1e-9)
    ln_n_factorial = n * log(n) - n + log(n * (1 + 4n * (1 + 2n))) / 6 + log(T(pi)) / 2
    return exp(ln_n_factorial)
end

println("safe_factorial(4) = ", safe_factorial(4.0), "  (exact 4! = 24)")

# Step 2. Parse rule definition
#
# We want the search to recognize `x!` as a call to `safe_factorial(x)`. To do this, we define a parse rule that rewrites the string `x!` to `safe_factorial(x)` before it is parsed by Julia's `Meta.parse`. This is done using a regular expression that captures the variable name and rewrites it accordingly.
#
# You can read up on regular expressions in Julia here: https://docs.julialang.org/en/v1/manual/strings/#man-regex-literals
#
# Stated plainly, the parse rule says: "Any word followed by a `!` should be rewritten to `safe_factorial(word)`". The `(\w+)` captures the word (variable name) and the `\1` in the replacement string refers to that captured group.
# This isn't robust -- it may not handle `x! + y!` or `(x!)!` correctly, but it's fine for exposition.
factorial_parse_rule = NormalizationRule(r"(\w+)!" => s"safe_factorial(\1)")

# Step 3. Register the operator and parse rule with LaSR
#
options = Options(;
    binary_operators=[+, -, *, /],
    unary_operators=[safe_factorial, sin, cos], # <-- Add the new operator to the search vocabulary
    plugins=(
        LaSRPlugin(;
            use_llm=false,
            variable_names=Dict("x0" => "x0"),
            parse_rules=[factorial_parse_rule], # <-- Add the parse rule to the plugin so that `x!` is recognized as `safe_factorial(x)`.
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
