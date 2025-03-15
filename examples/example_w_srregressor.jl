# This isn't in the automated testing suite since it requires an LLM server running in the background.
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using Revise
using TensorBoardLogger
using LibraryAugmentedSymbolicRegression:
    LaSROptions,
    LaSRMutationWeights,
    LLMOperationWeights,
    equation_search,
    calculate_pareto_frontier,
    compute_complexity,
    string_tree,
    SRLogger,
    eval_tree_array,
    LaSRRegressor
import MLJ: machine, fit!, predict, report

logger = SRLogger(TBLogger("logs/lasr_runs"); log_interval=1)

X = randn(Float32, 2, 100)
y = 2 * cos.(X[1, :]) + X[2, :] .^ 2 .- 2

p = 0.001
model = LaSRRegressor(;
    niterations=40,
    logger=logger,
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

mach = machine(model, transpose(X), y)
fit!(mach)
rep = report(mach)
pred = predict(mach, transpose(X))
# The error should be less than 1e-5
maxerr = maximum(abs.(pred - y))
println("Maximum error: $maxerr for model: $(rep.equations[rep.best_idx])")
@assert maxerr < 1e-5
