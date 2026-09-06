# Test: the parser fallback for malformed LLM output (ExpressionIO.jl).
# What is it supposed to do? parse_expr must never throw on bad input. It returns a
# constant-1 node. It first tries to strip an `x =` left side, then falls back.
# What do we hope to learn from the tests implemented here? The two :meta_parse branches
# (with a left side and without one) run, and empty variable_names uses the default names.
# This was the second-darkest module (78.9%): the round-trip tests feed only clean strings.
#
# Note: Meta.parse returns an `incomplete` value (it does not throw) for `((((`, which
# reaches the later :tree_parse branch. The input must truly fail (`x0 ++ )(`) to reach
# the :meta_parse branch.
using Test
using LibraryAugmentedSymbolicRegression: LaSRPlugin, parse_expr
using SymbolicRegression: Options, string_tree

opts = Options(;
    binary_operators=[+, -, *, /],
    unary_operators=[cos],
    plugins=(LaSRPlugin(; use_llm=false, variable_names=Dict("x0" => "x0", "x1" => "x1")),),
)
is_const1(ex) = string_tree(ex, opts) == "1.0"

@testset "bad input WITH a left side: strip it, then return constant 1" begin
    @test is_const1(parse_expr(Float64, "y = x0 ++ )(", opts))
end

@testset "bad input with NO left side: return constant 1" begin
    @test is_const1(parse_expr(Float64, "x0 ++ )(", opts))
end

@testset "empty variable_names uses the default names (x, y, ...)" begin
    # An empty Dict must not make every variable unknown, which would drop every
    # suggestion. It uses the default names, so `x + y` parses for real.
    optsd = Options(;
        binary_operators=[+],
        plugins=(LaSRPlugin(; use_llm=false, variable_names=Dict{Any,Any}()),),
    )
    @test string_tree(parse_expr(Float64, "x + y", optsd), optsd) == "x + y"
end
