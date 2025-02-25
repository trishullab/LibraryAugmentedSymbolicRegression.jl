# This isn't in the automated testing suite since it requires an LLM server running in the background.
using Pkg
Pkg.activate(".")
Pkg.develop(; path="../")
Pkg.instantiate()
using Revise
using LibraryAugmentedSymbolicRegression:
    LaSROptions,
    LaSRMutationWeights,
    LLMOperationWeights,
    equation_search,
    calculate_pareto_frontier,
    compute_complexity,
    string_tree,
    eval_tree_array
import LibraryAugmentedSymbolicRegression: mutate!

X = randn(Float32, 5, 100)
y = 2 * cos.(X[4, :]) + X[1, :] .^ 2 .- 2

p = 0.001
options = LaSROptions(;
    binary_operators=[+, -, *, /, ^],
    unary_operators=[cos],
    populations=20,
    use_llm=true,
    use_concepts=true,
    use_concept_evolution=true,
    llm_operation_weights=LLMOperationWeights(;
        llm_crossover=p, llm_mutate=p, llm_randomize=p
    ),
    llm_context="We believe the function to be a trigonometric function of the angle and a quadratic function of the bias.",
    llm_recorder_dir="lasr_runs/",
    variable_names=Dict("a" => "angle", "b" => "bias"),
    prompts_dir="prompts/",
    api_key="token-abc123",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    api_kwargs=Dict("url" => "http://localhost:11440/v1"),
    verbose=true,
)

## The rest of the code is the same as the example.jl file.
hall_of_fame = equation_search(X, y; niterations=40, options=options)

# dominating = calculate_pareto_frontier(hall_of_fame)

# trees = [member.tree for member in dominating]

# tree = trees[end]
# output, did_succeed = eval_tree_array(tree, X, options)

# for member in dominating
#     complexity = compute_complexity(member, options)
#     loss = member.loss
#     string = string_tree(member.tree, options)

#     println("$(complexity)\t$(loss)\t$(string)")
# end
