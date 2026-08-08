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
llm_options = LLMOptions(;
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    api_kwargs=Dict("url" => "http://localhost:11440/v1"),
    verbose=true, # Set to true to see LLM generation logs.
)
plugin = LaSRPlugin(;
    llm_options,
    use_llm=true,
    use_concepts=true,
    use_concept_evolution=true,
    context="We believe the relationship between the theta and offset parameter is a function of the cosine of the theta variable and the square of the offset.",
    variable_names=Dict("x1" => "theta", "x2" => "offset"),
)
options = Options(;
    binary_operators=[+, -, *, /, ^],
    unary_operators=[cos],
    populations=20,
    plugins=(plugin,),
    mutations=(LLMMutateMutation() => p, LLMRandomizeMutation() => p),
    crossovers=(LLMCrossover() => p,),
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
