# LaSR needs a parser to convert LLM-generated expression strings into DynamicExpressions compatible trees.
# These are round trip tests to ensure that the parser is working correctly.
println("Testing LaSR expression parser [hard]")

using Random: MersenneTwister, shuffle
using LibraryAugmentedSymbolicRegression:
    LaSROptions, string_tree, parse_expr, render_expr, gen_random_tree
include("test_params.jl")
options = LaSROptions(;
    default_params..., binary_operators=[+, *, ^], unary_operators=[sin, cos, exp]
)

rng = MersenneTwister(314159)

"""
Introduce some noise in the string to simulate real-world scenarios
where the string might not be perfectly formatted.
This function takes as argument a well formatted string such as
`x1 + sin(x2)**2` and adds noise (expected of an LLM output) such as:
 - Introduce a left hand side: `y = x1 + sin(x2)**2`
 - Add some random spaces: `y = x1 +  sin(x2) ** 2`
 - random capitalization `Y = x1 + Sin(x2) ** 2`
 - Replace "**" with "^": `Y = x1 + sin(x2) ^ 2`
"""
function fuzz_string(str::String)
    fuzzers = [
        s -> "y = " * s,  # Introduce a left hand side
        s -> replace(s, r"\s" => " "),  # Add random spaces
        s -> replace(s, r"sin" => "Sin"),  # capitalization
        s -> replace(s, r"cos" => "Cos"),  # capitalization
        s -> replace(s, r"exp" => "Exp"),  # capitalization
        s -> replace(s, r"\^" => "**"),  # Replace "^" with "**"
    ]
    # Apply the selected fuzzers in a random order
    for fuzzer in shuffle(rng, fuzzers)
        str = fuzzer(str)
    end
    return str
end

for depth in [5, 9]
    for nvar in [5, 9]
        random_trees = [gen_random_tree(depth, options, nvar, Float32, rng) for _ in 1:1e4]
        data = rand(Float32, nvar, 1000)

        for tree in random_trees
            output = tree(data, options.operators)
            if any(isnan.(output))
                continue
            end
            str_tree = string_tree(tree, options)
            # The string might not always be perfectly formatted. Introducing noise.
            str_tree = fuzz_string(str_tree)
            expr_tree = parse_expr(Float32, str_tree, options)
            expr_output = expr_tree(data, options.operators)
            @test isapprox(expr_output, output)
        end
    end
end
println("Passed.")
