# LaSR needs a parser to convert LLM-generated expression strings into DynamicExpressions compatible trees.
# These are round trip tests to ensure that the parser is working correctly.
println("Testing LaSR expression parser")

using Revise
using Random: MersenneTwister
using DynamicExpressions: parse_expression
using LibraryAugmentedSymbolicRegression:
    LaSROptions, string_tree, parse_expr, render_expr, gen_random_tree
include("test/test_params.jl")

@inline safepow(x, y) = sign(x) * abs(x)^y
options = Options(;
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
            @assert str_tree ==
                String(strip(str_tree, [' ', '\n', '"', ',', '.', '[', ']']))
            expr_tree = parse_expression(
                Meta.parse(str_tree);
                operators=options.operators,
                node_type=options.node_type,
                expression_type=options.expression_type,
                variable_names=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"][1:nvar],
            )
            expr_output = expr_tree(data, options.operators)
            @assert string_tree(expr_tree) == str_tree "[$i] String representation mismatch: $(string_tree(expr_tree)) vs $str_tree for tree: $str_tree"
            @assert isapprox(expr_output, output) "[$i] Output mismatch: $(expr_output) vs $(output) for tree: $str_tree"
        end
    end
end
println("Passed.")
