# Test: the MLJ SRRegressor path against the mock server.
# What is it supposed to do? It fits SRRegressor with the LaSR plugin through the MLJ
# interface. It uses the mock server, not a live model.
# What do we hope to learn from the tests implemented here? The MLJ path fits and gives a
# Pareto frontier. This is the only test of that path. It used to need a live server on
# :11440 and a specific model, so CI never ran it, and it asserted a lucky error bound.
using Test
import LibraryAugmentedSymbolicRegression: LaSRPlugin
import SymbolicRegression: SRRegressor
import MLJ: machine, fit!, predict, report

include("mock_llm_server.jl")
using .MockLLMServer: with_server

const TUTORIAL_PORT = 11_453
const PROMPTS_DIR = joinpath(@__DIR__, "prompts") * "/"

X = randn(Float32, 2, 100)
y = 2 * cos.(X[1, :]) + X[2, :] .^ 2 .- 2

with_server(TUTORIAL_PORT) do url
    model = SRRegressor(;
        niterations=3,
        binary_operators=[+, -, *, /, ^],
        unary_operators=[cos],
        populations=3,
        plugins=(
            LaSRPlugin(;
                use_llm=true,
                mutate_weight=1.0,
                randomize_weight=1.0,
                prompts_dir=PROMPTS_DIR,
                api_key="mock",
                model="mock-model",
                # The mock answers in x/y. Register those names so suggestions parse.
                api_kwargs=Dict("url" => url, "max_tokens" => 512),
                variable_names=Dict(1 => "x", 2 => "y"),
                verbose=false,
            ),
        ),
    )
    mach = machine(model, transpose(X), y)
    fit!(mach)
    rep = report(mach)
    @test !isempty(rep.equations)
    pred = predict(mach, transpose(X))
    @test length(pred) == length(y)
end
