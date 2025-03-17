using Pkg
Pkg.activate(".")
Pkg.instantiate()
using TensorBoardLogger
using LibraryAugmentedSymbolicRegression

# Dataset with 5 features:
X = randn(Float64, 5, 100)

# and one target:
y = 2 * cos.(X[4, :]) + X[1, :] .^ 2 .- 2

# with some noise:
y = y .+ randn(100) .* 1e-3

logger = SRLogger(TBLogger("logs/lasr_runs"); log_interval=1)
p = 0.0001
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
    llm_context="We believe the relationship between the theta and offset parameter is a function of the cosine of the theta variable and the square of the offset.",
    variable_names=Dict("x1" => "theta", "x2" => "offset"),
    prompts_dir="prompts/",
    api_key="token-abc123",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    api_kwargs=Dict("url" => "http://localhost:11440/v1"),
    verbose=true, # Set to true to see LLM generation logs.
)

hall_of_fame = equation_search(
    X, y; niterations=40, options=options, parallelism=:multithreading, logger=logger
)

dominating = calculate_pareto_frontier(hall_of_fame)

trees = [member.tree for member in dominating]

tree = trees[end]
output, did_succeed = eval_tree_array(tree, X, options)

println("Complexity\tMSE\tEquation")

for member in dominating
    complexity = compute_complexity(member, options)
    loss = member.loss
    string = string_tree(member.tree, options)

    println("$(complexity)\t$(loss)\t$(string)")
end
