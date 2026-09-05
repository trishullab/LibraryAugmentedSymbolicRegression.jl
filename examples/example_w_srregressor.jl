# This isn't in the automated testing suite since it requires an LLM server running in the background.
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using Revise
using TensorBoardLogger
using SymbolicRegression:
    equation_search,
    calculate_pareto_frontier,
    compute_complexity,
    string_tree,
    SRLogger,
    eval_tree_array,
    SRRegressor
using LibraryAugmentedSymbolicRegression: LaSRPlugin
import MLJ: machine, fit!, predict, report

logger = SRLogger(TBLogger("logs/lasr_runs"); log_interval=1)

X = randn(Float32, 2, 100)
y = 2 * cos.(X[1, :]) + X[2, :] .^ 2 .- 2

p = 0.001
plugin = LaSRPlugin(;
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    api_kwargs=Dict("url" => "http://localhost:11440/v1"),
    verbose=true,
    use_concepts=true,
    use_concept_evolution=true,
    context="We believe the relationship between the theta and offset parameter is a function of the cosine of the theta variable and the square of the offset.",
    variable_names=Dict("x1" => "theta", "x2" => "offset"),
    mutate_weight=p,
    randomize_weight=p,
    crossover_probability=p,
)
model = SRRegressor(;
    plugins=(plugin,),
    niterations=40,
    logger=logger,
    binary_operators=[+, -, *, /, ^],
    unary_operators=[cos],
    populations=20,
)

mach = machine(model, transpose(X), y)
fit!(mach)
rep = report(mach)
pred = predict(mach, transpose(X))
# The error should be less than 1e-5
maxerr = maximum(abs.(pred - y))
println("Maximum error: $maxerr for model: $(rep.equations[rep.best_idx])")
@assert maxerr < 1e-5
