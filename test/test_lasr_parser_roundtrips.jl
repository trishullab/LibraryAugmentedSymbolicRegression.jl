# LaSR needs a parser to convert LLM-generated expression strings into DynamicExpressions compatible trees.
# These are round trip tests but with the render_expr function.
println("Testing LaSR expression parser with render_expr")

using Random: MersenneTwister
using LibraryAugmentedSymbolicRegression:
    LaSROptions, string_tree, parse_expr, render_expr, gen_random_tree
include("test_params.jl")
options = LaSROptions(;
    default_params..., binary_operators=[+, *, ^, -], unary_operators=[sin, cos, exp]
)

rng = MersenneTwister(314159)

# nvar = 9 is the maximum number of variables that this test can support since
# x10 gets lexicographically sorted after x1 but before x2, messing up the variable names.
for depth in [5, 9]
    for nvar in [5, 9]
        options.variable_names = Dict('x' * string(i) => ('x' * string(i)) for i in 1:nvar)
        random_trees = [gen_random_tree(depth, options, nvar, Float32, rng) for _ in 1:1e3]

        for (i, tree) in enumerate(random_trees)
            str_tree = string_tree(tree, options)
            # Replace all floats or integers with 1.0
            str_tree_wo_constants = replace(
                str_tree, r"-?\b\d+(\.\d+)?([eE]-?\d+)?\b" => "1.0"
            )

            rendered_tree = render_expr(tree, options)
            expr_tree = parse_expr(Float32, rendered_tree, options)
            expr_tree_str = string_tree(expr_tree, options)
            @test str_tree_wo_constants == expr_tree_str
        end
    end
end
println("Passed.")
