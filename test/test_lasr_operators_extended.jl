using Test
using LibraryAugmentedSymbolicRegression
using LibraryAugmentedSymbolicRegression: LaSRPlugin
using LibraryAugmentedSymbolicRegression.ExpressionIOModule: parse_expr
using SymbolicRegression: Options, eval_tree_array

# NOTE: LaSR's `parse_expr` reads variable names / operators through the LaSRPlugin,
# so options must carry a `LaSRPlugin` (a bare `SymbolicRegression.Options` with no
# plugins is rejected by `parse_expr`).
opts = Options(;
    binary_operators=[+, *, -, /, ^],
    unary_operators=[abs, cbrt, tan, exp, log, sqrt, sin, cos],
    plugins=(LaSRPlugin(; variable_names=Dict("x0" => "x0", "x1" => "x1")),),
)
# Rows are features in sorted variable-name order: row 1 = x0, row 2 = x1.
X = [
    -2.0 -1.0 3.0;   # x0
    0.5 4.0 -2.0
]    # x1
x0 = X[1, :]
x1 = X[2, :]

@testset "abs/cbrt/tan round-trip when provided as operators" begin
    # These three operators are outside LaSR's historical vocabulary but must
    # parse and evaluate (not fall back to the constant-1 node) once supplied.
    for (s, f) in (("abs(x0)", abs), ("cbrt(x0)", cbrt), ("tan(x0)", tan))
        ex = parse_expr(Float64, s, opts)
        out, ok = eval_tree_array(ex, X, opts)
        @test ok
        @test isapprox(out, f.(x0); atol=1e-9)   # NOT the constant-1 fallback
    end
end

@testset "abs pipe notation |x| maps to the abs operator" begin
    # LLMs write absolute value as |x|, which is not valid Julia. Without
    # normalization `Meta.parse` fails and `parse_expr` returns the constant-1
    # fallback [1, 1, 1] instead of |x0| = [2, 1, 3].
    ex = parse_expr(Float64, "|x0|", opts)
    out, ok = eval_tree_array(ex, X, opts)
    @test ok
    @test isapprox(out, abs.(x0); atol=1e-9)   # NOT the constant-1 fallback

    # It composes with a power: |x|^(1/3) stays real because abs(x) ≥ 0,
    # which is the `|v|^⅓` idiom the design calls out.
    ex2 = parse_expr(Float64, "|x0|^(1/3)", opts)
    out2, ok2 = eval_tree_array(ex2, X, opts)
    @test ok2
    @test isapprox(out2, abs.(x0) .^ (1 / 3); atol=1e-9)

    # And it composes inside a larger expression: the design names `|x0| + x1`.
    ex3 = parse_expr(Float64, "|x0| + x1", opts)
    out3, ok3 = eval_tree_array(ex3, X, opts)
    @test ok3
    @test isapprox(out3, abs.(x0) .+ x1; atol=1e-9)   # NOT the constant-1 fallback
end

@testset "ordinary (non-pipe) expression round-trips through normalization" begin
    # Regression guard: `_normalize_expr_string`'s pipe-to-`abs` rewrite (and the other
    # normalization steps) must leave ordinary, pipe-free input semantically untouched.
    ex = parse_expr(Float64, "x0 + sin(x1)", opts)
    out, ok = eval_tree_array(ex, X, opts)
    @test ok
    @test isapprox(out, x0 .+ sin.(x1); atol=1e-9)   # NOT the constant-1 fallback
end
