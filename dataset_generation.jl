# julia dataset_generation.jl > pysr_dataset.log
# import Pkg;
# Pkg.add("SymbolicRegression")
# Pkg.add("Zygote")
# Pkg.add("Random")
# Pkg.add("Statistics")

using SymbolicRegression: SymbolicRegression as SR
using Zygote: Zygote
using Random: MersenneTwister, seed!
using Statistics: mean, std

"""
This treats the binary tree like an iterator, and checks if
there are any nodes with degree 1 that have a child with degree 1.
"""
function has_nested_unary(eq)
    return any(eq) do node
        node.degree == 1 && any(node) do child
            child !== node && child.degree == 1
        end
    end
end

"""
This checks if an equation has a good range of values,
or whether there are anomalous points.
"""
function has_good_range(eq, sample_X, options)
    sample_y = eq(sample_X, options)
    return any(isnan, sample_y) || maximum(abs, sample_y) > 1e4
end

"""
This checks if an equation has a good range of derivative values,
or whether there are anomalous points.
"""
function has_good_derivative_range(eq, sample_X, options)
    # Also check derivative with respect to input:
    sample_dy = eq'(sample_X, options)
    return any(isnan, sample_dy) || maximum(abs, sample_dy) > 1e4
end

"""
This combines the above three tests into one, returning false
for any equation that breaks any of the three tests.
"""
function equation_filter(eq, sample_X, options)
    if (
        has_nested_unary(eq)
        || has_good_range(eq, sample_X, options)
        || has_good_derivative_range(eq, sample_X, options)
    )
        return false
    else
        return true
    end
end

function normalize(eq, sample_y)
    return (eq - mean(sample_y)) / std(sample_y)
end

"""
Generate a vector of equations that satisfy `equation_filter`
and use the input properties.
"""
function generate_equations(;
    num_equations=10_000,
    max_attempts=100_000,
    T=Float64,
    num_features=5,
    num_samples=100,
    binary_operators=(+, -, *, /),
    unary_operators=(cos, sqrt),
)
    options = SR.Options(;
        enable_autodiff=true,
        binary_operators,
        unary_operators
    )
    SR.@extend_operators options
    sample_X = rand(MersenneTwister(0), T, num_features, num_samples) .* 10 .- 5

    equations = SR.Node{T}[]
    i = 0
    while length(equations) < num_equations && i < max_attempts
        i += 1
        seed!(i)
        isinteger(log2(i)) && println("Tried $i equations. Currently have $(length(equations)) equations saved.")

        number_additions = rand(MersenneTwister(length(equations)), 5:20)
        eq = SR.gen_random_tree(number_additions, options, num_features, T)
        sample_y = eq(sample_X, options)
        
        # Normalize equation:
        eq = Base.invokelatest(normalize, eq, sample_y)

        if equation_filter(eq, sample_X, options)
            push!(equations, eq)
        end
    end

    return equations, options
end

equations, options = generate_equations(max_attempts=100_000)

for equation in equations
    println(SR.string_tree(equation, options))
end