# This isn't in the automated testing suite since it requires an LLM server running in the background.

using LibraryAugmentedSymbolicRegression

X = randn(Float32, 5, 100)
y = 2 * cos.(X[4, :]) + X[1, :] .^ 2 .- 2

llm_options = LibraryAugmentedSymbolicRegression.LLMOptions(;
    active=true,
    weights=LibraryAugmentedSymbolicRegression.LLMWeights(;
        llm_mutate=0.01, llm_crossover=0.01, llm_gen_random=0.01
    ),
    num_pareto_context=5,
    prompt_evol=true,
    prompt_concepts=true,
    api_key="token-abc123",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    api_kwargs=Dict("url" => "http://localhost:11440/v1"),
)

options = LibraryAugmentedSymbolicRegression.Options(;
    binary_operators=[+, *, /, -],
    unary_operators=[cos, exp],
    populations=20,
    llm_options=llm_options,
)

## The rest of the code is the same as the example.jl file.
hall_of_fame = equation_search(
    X, y; niterations=40, options=options, parallelism=:multithreading
)

dominating = calculate_pareto_frontier(hall_of_fame)

trees = [member.tree for member in dominating]

tree = trees[end]
output, did_succeed = eval_tree_array(tree, X, options)

for member in dominating
    complexity = compute_complexity(member, options)
    loss = member.loss
    string = string_tree(member.tree, options)

    println("$(complexity)\t$(loss)\t$(string)")
end
