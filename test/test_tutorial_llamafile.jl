# Test that LaSR runs with active=true and can solve simple problems.
import LibraryAugmentedSymbolicRegression:
    LaSROptions,
    LaSRRegressor,
    LaSRMutationWeights,
    LLMOperationWeights,
    LLAMAFILE_MODEL,
    LLM_PORT
import MLJ: machine, fit!, predict, report

# Ensure that TEST_LLM env variable is set to true.
is_llm_started = (get(ENV, "START_LLAMASERVER", "false") == "true")
is_llm_started ||
    @warn "Please set the START_LLAMASERVER env variable to true to run this test."
@test is_llm_started
X = randn(Float32, 2, 100)
y = 2 * cos.(X[1, :]) + X[2, :] .^ 2 .- 2

p = 1e-6 # really small because the llamafile is VERY slow...
model = LaSRRegressor(;
    niterations=40,
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
    api_key="OpenAI complains if this isn't set. but LLamafile doesn't need one.",
    model=LLAMAFILE_MODEL,
    api_kwargs=Dict("url" => "http://localhost:$(LLM_PORT)/v1"),
    verbose=true, # Set to true to see LLM generation logs.
)

mach = machine(model, transpose(X), y)
fit!(mach)
rep = report(mach)
pred = predict(mach, transpose(X))
# The error should be less than 1e-5
@test maximum(abs.(pred - y)) < 1e-5
