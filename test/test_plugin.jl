using Test
using LibraryAugmentedSymbolicRegression
using SymbolicRegression

function mock_llm(calls, content)
    return function (args...; kwargs...)
        calls[] += 1
        return (; content)
    end
end

@testset "LaSRPlugin mutations and mocked search" begin
    calls = Ref(0)
    llm_options = LLMOptions(; verbose=false, llm_generate=mock_llm(calls, "[\"x1 + 1\"]"))
    plugin = LaSRPlugin(;
        llm_options,
        variable_names=Dict(1 => "x1"),
        prompts_dir=joinpath(pkgdir(LibraryAugmentedSymbolicRegression), "prompts") * "/",
    )
    options = Options(;
        binary_operators=[+],
        plugins=(plugin,),
        default_plugins=(),
        default_mutations=(),
        mutations=(LLMMutateMutation() => 1.0, LLMRandomizeMutation() => 2.0),
        populations=1,
        population_size=8,
        tournament_selection_n=3,
        ncycles_per_iteration=2,
        maxsize=10,
        seed=11,
        deterministic=true,
        save_to_file=false,
    )

    @test options.plugins === (plugin,)
    @test any(
        pair -> pair.first isa LLMMutateMutation && pair.second == 1.0, options.mutations
    )
    @test any(
        pair -> pair.first isa LLMRandomizeMutation && pair.second == 2.0, options.mutations
    )

    overridden = Options(;
        binary_operators=[+],
        plugins=(plugin,),
        default_plugins=(),
        default_mutations=(),
        mutations=(LLMMutateMutation() => 3.0,),
    )
    @test count(pair -> pair.first isa LLMMutateMutation, overridden.mutations) == 1
    @test only(
        filter(pair -> pair.first isa LLMMutateMutation, overridden.mutations)
    ).second == 3.0
    @test LaSROptions(; binary_operators=[+], default_plugins=()) isa Options

    X = reshape(collect(range(-1.0, 1.0; length=20)), 1, :)
    y = vec(X) .+ 1
    hof = equation_search(X, y; options, niterations=1, parallelism=:serial)
    @test hof isa HallOfFame
    @test calls[] > 0
end

@testset "LaSRPlugin mocked concept lifecycle" begin
    calls = Ref(0)
    llm_options = LLMOptions(;
        verbose=false, llm_generate=mock_llm(calls, "[\"additive relationship\"]")
    )
    plugin = LaSRPlugin(;
        llm_options,
        use_concept_evolution=true,
        num_concept_crossover=1,
        prompts_dir=joinpath(pkgdir(LibraryAugmentedSymbolicRegression), "prompts") * "/",
    )
    options = Options(;
        binary_operators=[+],
        plugins=(plugin,),
        default_plugins=(),
        populations=1,
        population_size=8,
        tournament_selection_n=3,
        ncycles_per_iteration=2,
        maxsize=10,
        seed=13,
        deterministic=true,
        save_to_file=false,
    )
    X = reshape(collect(range(-1.0, 1.0; length=20)), 1, :)
    hof = equation_search(X, vec(X) .+ 1; options, niterations=1, parallelism=:serial)
    @test hof isa HallOfFame
    @test calls[] > 0
end

@testset "LaSRPlugin mocked crossover" begin
    calls = Ref(0)
    llm_options = LLMOptions(;
        verbose=false, llm_generate=mock_llm(calls, "[\"x1 + 1\", \"x1 * x1\"]")
    )
    plugin = LaSRPlugin(;
        llm_options,
        variable_names=Dict(1 => "x1"),
        prompts_dir=joinpath(pkgdir(LibraryAugmentedSymbolicRegression), "prompts") * "/",
    )
    options = Options(;
        binary_operators=[+, *],
        plugins=(plugin,),
        default_plugins=(),
        crossovers=(LLMCrossover() => 1.0,),
        default_crossovers=(),
        populations=1,
        population_size=8,
        tournament_selection_n=3,
        ncycles_per_iteration=2,
        crossover_probability=1.0,
        maxsize=10,
        seed=12,
        deterministic=true,
        save_to_file=false,
    )
    X = reshape(collect(range(-1.0, 1.0; length=20)), 1, :)
    y = vec(X) .^ 2
    hof = equation_search(X, y; options, niterations=1, parallelism=:serial)
    @test hof isa HallOfFame
    @test calls[] > 0
end

@testset "LaSROptions maps legacy crossover probability" begin
    options = LaSROptions(;
        use_llm=true,
        llm_operation_weights=(; llm_crossover=0.25),
        binary_operators=[+],
        default_plugins=(),
    )
    @test length(options.crossovers) == 2
    @test only(filter(pair -> pair.first isa LLMCrossover, options.crossovers)).second ==
        0.25
    @test only(
        filter(pair -> pair.first isa SubtreeCrossover, options.crossovers)
    ).second == 0.75
end
