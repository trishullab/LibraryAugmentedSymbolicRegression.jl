# Test that LaSR runs with active=true and can solve simple problems.
import LibraryAugmentedSymbolicRegression:
    LaSROptions, LaSRRegressor, LaSRMutationWeights, LLMOperationWeights
import MLJ: machine, fit!, predict, report

# Dataset with two named features:
X = (a=rand(500), b=rand(500))

# and one target:
y = @. 2 * cos(X.a * 23.5) - X.b^2

# with some noise:
y = y .+ randn(500) .* 1e-3

model = LaSRRegressor(;
    niterations=50,
    binary_operators=[+, -, *],
    unary_operators=[cos],
    use_llm=true,
    use_concepts=true,
    use_concept_evolution=true,
    lasr_mutation_weights=LaSRMutationWeights(; llm_mutate=0.1, llm_randomize=0.1),
    llm_operation_weights=LLMOperationWeights(; llm_crossover=0.1),
    llm_context="We believe the function to be a trigonometric function of the angle and a quadratic function of the bias.",
    llm_recorder_dir="lasr_runs/",
    variable_names=Dict("a" => "angle", "b" => "bias"),
    prompts_dir="prompts/",
    api_key="token-abc123",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    api_kwargs=Dict("url" => "http://localhost:11440/v1"),
    verbose=true,
)
mach = machine(model, X, y)

# ensure ./prompts/ exists. If not, download and extract the prompts.zip file from the repository.
fit!(mach)
# open ./lasr_runs/debug_0/llm_calls.txt to see the LLM interactions.
rep = report(mach)
ypred = predict(mach, X)
