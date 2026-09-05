# End-to-end coverage of the LLM code path against a local mock server.
#
# These tests exist because the suite otherwise has *no* coverage of an actual LLM round
# trip: the `online` tag only exercises prompt/parser units, and the tagged tutorial tests
# never run in CI. That let two regressions ship undetected -- `llm_mutate` crashed with a
# `FieldError`, and the LLM crossover was never dispatched at all.

using Test
using LibraryAugmentedSymbolicRegression
using LibraryAugmentedSymbolicRegression: Options, LaSRPlugin, LLMCrossover
using SymbolicRegression: equation_search, calculate_pareto_frontier

include("mock_llm_server.jl")
using .MockLLMServer: MockLLMServer, with_server, count_for, total_calls

const PROMPTS_DIR = joinpath(@__DIR__, "prompts") * "/"
const MOCK_PORT = 11_451

function lasr_options(url; kws...)
    return Options(;
        binary_operators=[+, -, *, /, ^],
        unary_operators=[cos],
        populations=2,
        population_size=33,
        # Every mutation here costs an HTTP round trip, so keep the cycle count small
        # enough that the whole file stays cheap in CI.
        ncycles_per_iteration=3,
        plugins=(
            LaSRPlugin(;
                use_llm=true,
                use_concepts=true,
                use_concept_evolution=true,
                mutate_weight=1.0,
                randomize_weight=1.0,
                crossover_probability=1.0,
                prompts_dir=PROMPTS_DIR,
                api_key="mock-key",
                model="mock-model",
                api_kwargs=Dict("url" => url, "max_tokens" => 1000),
                # The mock server answers in terms of `x`/`y`. Register those names so
                # suggestions actually parse -- otherwise every one falls back to the
                # constant-1 node and the test passes without exercising insertion.
                variable_names=Dict(1 => "x", 2 => "y"),
                verbose=false,
            ),
        ),
        progress=false,
        kws...,
    )
end

@testset "LLM operators are actually invoked during a search" begin
    X = randn(Float32, 2, 60)
    y = @. 2 * cos(X[1, :]) + X[2, :]^2 - 2

    with_server(MOCK_PORT) do url
        options = lasr_options(url)
        # The search must complete rather than throwing. `llm_mutate` used to raise
        # `FieldError: Expression has no field val` on the first LLM suggestion.
        hof = equation_search(X, y; options, niterations=2, parallelism=:serial)
        @test hof !== nothing

        # And the LLM must genuinely have been consulted, not silently skipped. A
        # swallowed exception in the LLM path shows up here as a zero count.
        @test total_calls() > 0
        @test count_for("mutate") > 0
    end
end

@testset "LLM suggestions reach the population" begin
    X = randn(Float32, 2, 60)
    y = @. 2 * cos(X[1, :]) + X[2, :]^2 - 2

    with_server(MOCK_PORT) do url
        options = lasr_options(url)
        hof = equation_search(X, y; options, niterations=2, parallelism=:serial)
        frontier = calculate_pareto_frontier(hof)
        @test !isempty(frontier)
        @test all(m -> m.tree isa SymbolicRegression.AbstractExpression, frontier)
    end
end

@testset "LLMCrossover is dispatched and produces valid children" begin
    # The plugin injects `LLMCrossover() => crossover_probability` into `options.crossovers`,
    # and the engine dispatches `crossover(m1, m2, ::LLMCrossover, options; kws...)`. This
    # pins that wiring: the crossover operator was ported to the `AbstractCrossover` API
    # without ever being exercised against a live endpoint.
    using SymbolicRegression:
        Dataset,
        PopMember,
        create_expression,
        gen_random_tree_fixed_size,
        crossover,
        init_plugin_state

    with_server(MOCK_PORT) do url
        options = lasr_options(url)
        X = randn(Float32, 2, 40)
        y = @. 2 * cos(X[1, :]) + X[2, :]^2 - 2
        dataset = Dataset(X, y)

        mk(m) = PopMember(
            dataset,
            create_expression(
                gen_random_tree_fixed_size(5, options, 2, Float32), options, dataset
            ),
            options;
            deterministic=false,
        )
        p1, p2 = mk(1), mk(2)
        states = map(p -> init_plugin_state(p, options, dataset), options.plugins)

        # LaSR's method must win dispatch over SR's built-in subtree crossover.
        @test which(
            crossover, (typeof(p1), typeof(p2), LLMCrossover, typeof(options))
        ).module === LibraryAugmentedSymbolicRegression.MutateModule

        before = count_for("crossover")
        result = crossover(
            p1,
            p2,
            LLMCrossover(),
            options;
            dataset=dataset,
            curmaxsize=20,
            plugin_states=states,
            attempt=1,
        )
        @test result.child1 isa SymbolicRegression.AbstractExpression
        @test result.child2 isa SymbolicRegression.AbstractExpression
        @test count_for("crossover") > before
    end
end

@testset "a constraint retry does not burn a second LLM call" begin
    # The expensive-crossover contract: on `attempt > 1` the operator hands the parents
    # back instead of re-querying the model.
    using SymbolicRegression:
        Dataset,
        PopMember,
        create_expression,
        gen_random_tree_fixed_size,
        crossover,
        init_plugin_state

    with_server(MOCK_PORT) do url
        options = lasr_options(url)
        X = randn(Float32, 2, 40)
        y = @. 2 * cos(X[1, :]) + X[2, :]^2 - 2
        dataset = Dataset(X, y)
        mk(m) = PopMember(
            dataset,
            create_expression(
                gen_random_tree_fixed_size(5, options, 2, Float32), options, dataset
            ),
            options;
            deterministic=false,
        )
        p1, p2 = mk(1), mk(2)
        states = map(p -> init_plugin_state(p, options, dataset), options.plugins)

        before = count_for("crossover")
        crossover(
            p1,
            p2,
            LLMCrossover(),
            options;
            dataset=dataset,
            curmaxsize=20,
            plugin_states=states,
            attempt=2,
        )
        @test count_for("crossover") == before
    end
end

@testset "search still succeeds when the LLM server is unreachable" begin
    # Pointing at a dead port must degrade to plain symbolic regression, not crash.
    options = lasr_options("http://127.0.0.1:1/v1")
    X = randn(Float32, 2, 60)
    y = @. 2 * cos(X[1, :]) + X[2, :]^2 - 2
    hof = equation_search(X, y; options, niterations=1, parallelism=:serial)
    @test hof !== nothing
end
