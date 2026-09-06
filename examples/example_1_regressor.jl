# Example 1: Using LaSRPlugin with SRRegressor
# Run (after `julia --project=examples -e 'using Pkg; Pkg.instantiate()'`):
#   julia --project=examples examples/example_1_regressor.jl
# 
# Brief: LaSR is a plugin within SymbolicRegression.jl that allows you to incorporate domain knowledge into the symbolic regression search process, using a language model to guide the search. This example demonstrates how to use LaSRPlugin with SRRegressor to find a symbolic expression that fits a given dataset.

# Imports:
#  - SymbolicRegression: The main package for performing symbolic regression.
#  - LibraryAugmentedSymbolicRegression: Contains the LaSRPlugin for integrating language model guidance into the symbolic regression process.
#  - MLJ: A machine learning framework in Julia that provides a consistent interface for training and evaluating models.
using SymbolicRegression
using LibraryAugmentedSymbolicRegression: LaSRPlugin
import MLJ: machine, fit!, predict, report

# LLM connection settings are read from a local `.env` (see `.env.example`) so the model
# and endpoint are never hardcoded. Discover what's deployed locally with `model-blame`.
function load_dotenv!(path=joinpath(@__DIR__, "..", ".env"))
    isfile(path) || return nothing
    for raw in eachline(path)
        line = strip(raw)
        (isempty(line) || startswith(line, "#")) && continue
        line = replace(line, r"^export\s+" => "")
        i = findfirst(==('='), line)
        i === nothing && continue
        get!(ENV, strip(line[1:(i - 1)]), strip(line[(i + 1):end], ['"', '\'', ' ', '\t']))
    end
end
load_dotenv!()
const LLM_URL = get(ENV, "LASR_LLM_URL", "http://127.0.0.1:8001/v1")
const LLM_MODEL = get(ENV, "LASR_LLM_MODEL", "gemma-4-12b")
const LLM_API_KEY = get(ENV, "VLLM_API_KEY", "local")

# We will try to recover the function `y = 2 * cos(theta) + offset^2 - 2` from noisy data. The input data is generated randomly, and the output is computed using the known function.
X = randn(Float32, 2, 100)
X_test = randn(Float32, 2, 100)
y = 2 * cos.(X[1, :]) + X[2, :] .^ 2 .- 2

# LaSR's goal is to _augment_ the search with domain knowledge, not to replace it. The probability parameter (p) controls how often the LLM is consulted during the search. A small value of p means that the search will mostly rely on traditional symbolic regression methods, while occasionally consulting the LLM for guidance. A larger value of p would make the search more dependent on the LLM's suggestions.
# Generally, it's best to start with a very small (or zero) probability of querying the LLM to guage how well _base_ symbolic regression performs on the problem. If the base search struggles, you can increase p to allow the LLM to provide more guidance. 
p = 0.001
model = SRRegressor(;
    plugins=(
        LaSRPlugin(;
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            api_kwargs=Dict(
                "url" => LLM_URL,
                "max_tokens" => 4096,
                "chat_template_kwargs" => Dict("enable_thinking" => false),
            ),
            verbose=true,
            use_concepts=true,
            use_concept_evolution=true,
            # The context provided to the LLM should describe the relationship between the input variables and the output variable. Generally, the more specific and informative the context, the better the LLM can guide the search.
            context="We believe the relationship between the theta and offset parameter is a function of the cosine of the theta variable and the square of the offset.",
            variable_names=Dict("x1" => "theta", "x2" => "offset"),
            mutate_weight=p,
            randomize_weight=p,
            crossover_probability=p,
        ),
    ),
    niterations=40,
    binary_operators=[+, -, *, /, ^],
    unary_operators=[cos],
    populations=20,
)
# Fit the model using MLJ's interface.
mach = machine(model, transpose(X), y)
fit!(mach)
# A sample report output might look like this:
# Complexity  Loss       Score      Equation
# 1           1.757e+00  0.000e+00  y = 0.084188
# 2           1.546e+00  1.280e-01  y = cos(x₁)
# 3           1.318e+00  1.595e-01  y = x₂ * x₂
# 4           1.084e+00  1.950e-01  y = cos(x₁ / x₂)
# 5           6.379e-01  5.304e-01  y = cos(x₁) - cos(x₂)
# 7           9.506e-02  9.518e-01  y = (cos(x₁) - cos(x₂)) / 0.42965
# 9           5.230e-02  2.988e-01  y = (cos(x₁ * 0.87706) - cos(x₂)) / 0.39831
# 10          0.000e+00  1.033e+02  y = ((x₂ * x₂) + (cos(x₁) * 2)) - 2

# Notice that at Complexity 10, we have recovered the exact equation that generated the data: `y = ((x₂ * x₂) + (cos(x₁) * 2)) - 2`

# SRRegressor's `report` method provides a summary of the search process, including the best-found equations and their performance metrics. This information can be useful for understanding how well the model has learned the underlying relationship in the data.
rep = report(mach)

# Let's evaluate the best equation found by the search on a test dataset. We will compute the predictions and compare them to the true values to assess the model's accuracy.
pred = predict(mach, transpose(X_test))
# The error should be less than 1e-5
maxerr = maximum(abs.(pred - (2 * cos.(X_test[1, :]) + X_test[2, :] .^ 2 .- 2)))
println("Maximum error: $maxerr for model: $(rep.equations[rep.best_idx])")
