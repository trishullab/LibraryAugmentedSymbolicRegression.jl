# Regression coverage for `examples/operator_extension.jl` -- see that file's header for
# the full two-sided story (a differentiable operator, plus a `NormalizationRule` that
# maps a scientist's/LLM's notation onto it). This test asserts the wiring works
# end-to-end WITHOUT a model server: `use_llm=false`, no LLM client ever invoked, fully
# deterministic.
using Test
using LibraryAugmentedSymbolicRegression
using LibraryAugmentedSymbolicRegression: LaSRPlugin, NormalizationRule
using LibraryAugmentedSymbolicRegression.ExpressionIOModule: parse_expr
using SymbolicRegression: Options, eval_tree_array

# Same definition as `examples/operator_extension.jl`, duplicated here so this test file
# has no runtime dependency on `examples/` and stays a self-contained regression test.
# `SpecialFunctions` (which would give an exact `gamma(x + 1)`) is only a *transitive*
# dependency of this package (pulled in via `SymbolicRegression`/`Optim`), not a direct
# one in `Project.toml`; per the project's no-new-dependency policy this uses Ramanujan's
# closed-form approximation of `ln(n!)` instead -- see the example file for the full
# derivation and its accuracy tradeoff.
function safe_factorial(x::T) where {T<:Real}
    n = abs(x) + T(1e-9)
    ln_n_factorial = n * log(n) - n + log(n * (1 + 4n * (1 + 2n))) / 6 + log(T(pi)) / 2
    return exp(ln_n_factorial)
end

@testset "safe_factorial evaluates correctly on a sample" begin
    # Ramanujan's ln(n!) approximation is accurate to ~1e-4 relative error for n >= 1.
    @test isapprox(safe_factorial(4.0), 24.0; rtol=1e-3)
    @test isapprox(safe_factorial(5.0), 120.0; rtol=1e-3)
    @test isapprox(safe_factorial(1.0), 1.0; rtol=1e-3)
    @test isapprox(safe_factorial(3.0f0), 6.0f0; rtol=1.0f-3)   # works on Float32 too
end

@testset "Options(plugins=(LaSRPlugin(parse_rules=...),)) + parse_expr compose" begin
    # Deterministic sanity: constructing `Options` with a `LaSRPlugin` carrying a
    # scientist-registered `parse_rules` entry, and reaching it back out through
    # `parse_expr`, requires no LLM call at all.
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
    @test options.plugins[1] isa LaSRPlugin
    @test options.plugins[1].use_llm == false
    @test length(options.plugins[1].parse_rules) == 1

    # `x0!` is not valid Julia syntax; without the plugin's parse rule this would fail
    # `Meta.parse` and `parse_expr` would fall back to a constant-1 node (see
    # `src/ParseFailures.jl`'s `ParseFailureStore`) instead of the factorial operator.
    ex = parse_expr(Float64, "x0!", options)

    X = reshape([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], 1, :)
    x0 = vec(X)
    out, ok = eval_tree_array(ex, X, options)
    @test ok
    @test isapprox(out, safe_factorial.(x0); atol=1e-9)   # NOT the constant-1 fallback
    @test !isapprox(out, ones(length(x0)); atol=1e-9)      # explicitly rule out the fallback
end
