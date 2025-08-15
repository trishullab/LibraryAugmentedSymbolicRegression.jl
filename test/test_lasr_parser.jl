# LaSR needs a parser to convert LLM-generated expression strings into DynamicExpressions compatible trees.
# These are round trip tests to ensure that the parser is working correctly.
println("Testing LaSR expression parser")

using Random: MersenneTwister
using LibraryAugmentedSymbolicRegression:
    LaSROptions, string_tree, parse_expr, render_expr, gen_random_tree
include("test_params.jl")

@inline safepow(x, y) = sign(x) * abs(x)^y
options = LaSROptions(;
    default_params..., binary_operators=[-, +, *, safepow], unary_operators=[sin, cos, exp]
)

rng = MersenneTwister(314159)

for depth in [5, 9]
    for nvar in [5, 9]
        random_trees = [gen_random_tree(depth, options, nvar, T, rng) for _ in 1:1e3]
        data = rand(T, nvar, 1000)

        for (i, tree) in enumerate(random_trees)
            output = tree(data, options.operators)
            if any(isnan.(output))
                continue
            end
            str_tree = string_tree(tree, options)
            @test str_tree == String(strip(str_tree, [' ', '\n', '"', ',', '.', '[', ']']))
            expr_tree = parse_expr(T, str_tree, options)
            expr_output = expr_tree(data, options.operators)
            @test isapprox(expr_output, output)
        end
    end
end
println("Passed.")
