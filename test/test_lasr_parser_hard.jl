# LaSR needs a parser to convert LLM-generated expression strings into DynamicExpressions compatible trees.
# These are round trip tests to ensure that the parser is working correctly.
println("Testing LaSR expression parser [hard]")

using Random: MersenneTwister, shuffle
using LibraryAugmentedSymbolicRegression:
    LaSROptions, string_tree, parse_expr, gen_random_tree
include("test_params.jl")

@inline sinf(x) = sin(T(x))::T
@inline cosf(x) = cos(T(x))::T
@inline expf(x) = exp(T(clamp(x, -40.0, 40.0)))::T
@inline safelog(x) = log(abs(T(x)) + eps(T))::T  # no 1e-12!
@inline sqr(x) = (t=T(x); (t*t)::T)
@inline cube(x) = (t=T(x); (t*t*t)::T)
@inline addf(a, b) = (T(a) + T(b))::T
@inline mulf(a, b) = (T(a) * T(b))::T
@inline divf(a, b) = (ta=T(a); tb=T(b); (ta / (abs(tb) + eps(T)))::T)

options = LaSROptions(;
    default_params...,
    binary_operators=[+, addf, mulf, divf],
    unary_operators=[sinf, cosf, expf, safelog, sqr, cube],
    variable_names=Dict('x' * string(i) => ('x' * string(i)) for i in 1:9),
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
        s -> replace(s, r"\s" => " "),  # Add spaces
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
        random_trees = [gen_random_tree(depth, options, nvar, T, rng) for _ in 1:1e4]
        data = rand(T, nvar, 1000)

        for (i, tree) in enumerate(random_trees)
            output = tree(data, options.operators)
            if any(isnan.(output))
                continue
            end
            str_tree = string_tree(tree, options)
            # The string might not always be perfectly formatted. Introducing noise.
            str_tree = fuzz_string(str_tree)
            expr_tree = parse_expr(T, str_tree, options)
            expr_output = expr_tree(data, options.operators)
            @test isapprox(expr_output, output)
        end
    end
end
println("Passed.")
