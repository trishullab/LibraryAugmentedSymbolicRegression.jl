using Test
using LibraryAugmentedSymbolicRegression
using SymbolicRegression

function mock_llm(calls, content)
    return function (args...; kwargs...)
        calls[] += 1
        return (; content)
    end
end

@testset "LaSRPlugin mutation defaults and mocked search" begin
    calls = Ref(0)
    # Weights set far above SR's structural-mutation total (~9, see MutationWeights
    # defaults) so the mocked LLM mutations are overwhelmingly likely to be sampled at
    # least once across the tiny deterministic search below, now that they compete
    # against the full structural default set (see NOTE below).
    plugin = LaSRPlugin(;
        verbose=false,
        llm_generate=mock_llm(calls, "[\"x1 + 1\"]"),
        variable_names=Dict(1 => "x1"),
        prompts_dir=joinpath(pkgdir(LibraryAugmentedSymbolicRegression), "prompts") * "/",
        mutate_weight=100.0,
        randomize_weight=200.0,
    )
    # NOTE: no `default_mutations` here -- passing it (even as `()`) suppresses the
    # `plugin_mutations` merge in SR v2.0.0-beta.2's `Options` (see `Options.jl`:
    # `_plugin_mutations` is only populated `if default_mutations === nothing`), which
    # would keep the plugin's LLMMutateMutation/LLMRandomizeMutation out of
    # `options.mutations` entirely and make the assertions below vacuously fail.
    options = Options(;
        binary_operators=[+],
        plugins=(plugin,),
        default_plugins=(),
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
    @test any(pair -> pair.first isa LLMMutateMutation && pair.second == 100.0, options.mutations)
    @test any(
        pair -> pair.first isa LLMRandomizeMutation && pair.second == 200.0,
        options.mutations,
    )

    overridden = Options(;
        binary_operators=[+],
        plugins=(plugin,),
        default_plugins=(),
        default_mutations=(),
        mutations=(LLMMutateMutation() => 3.0,),
    )
    @test count(pair -> pair.first isa LLMMutateMutation, overridden.mutations) == 1
    @test only(filter(pair -> pair.first isa LLMMutateMutation, overridden.mutations)).second == 3.0
    @test Options(; binary_operators=[+], default_plugins=(), plugins=(LaSRPlugin(),)) isa
        Options

    X = reshape(collect(range(-1.0, 1.0; length=20)), 1, :)
    y = vec(X) .+ 1
    hof = equation_search(X, y; options, niterations=1, parallelism=:serial)
    @test hof isa HallOfFame
    @test calls[] > 0
end

@testset "LaSRPlugin mocked concept lifecycle actually generates concepts" begin
    calls = Ref(0)
    saw_concept = Ref(false)
    # A recording mock. The concept templates ask for "hypotheses"; equation templates ask
    # for "expressions". So this word shows the concept path ran, not just some LLM call.
    concept_aware = function (_schema, conversation; kwargs...)
        calls[] += 1
        text = lowercase(join((string(getproperty(m, :content)) for m in conversation), "\n"))
        occursin("hypothes", text) && (saw_concept[] = true)
        return (; content="[\"additive relationship\"]")
    end
    plugin = LaSRPlugin(;
        verbose=false,
        llm_generate=concept_aware,
        use_concepts=true,
        use_concept_evolution=true,
        num_concept_crossover=1,
        # Seed past the window so the evolution pool is non-empty and the refined-concept path runs.
        idea_store=WindowedIdeaStore(; window=2, seed=["seed a", "seed b", "seed c", "seed d"]),
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
    @test saw_concept[]   # the concept path ran, not just some LLM call
end

@testset "crossover_probability normalizes to a true conditional probability" begin
    # crossover_probability is documented and bounded [0, 1] as P(LLM crossover | crossover).
    # It must equal the LLM crossover weight over the total crossover weight, so pinning
    # SubtreeCrossover to (1 - p) is required (without it p=1.0 gave only 0.5).
    for p in (0.25, 0.5, 1.0)
        opts = Options(;
            binary_operators=[+, *],
            default_plugins=(),
            plugins=(LaSRPlugin(; use_llm=true, crossover_probability=p),),
        )
        total = sum(last, opts.crossovers)
        llm_w = only([last(x) for x in opts.crossovers if first(x) isa LLMCrossover])
        @test llm_w / total ≈ p
    end
end

@testset "LaSRPlugin mocked crossover" begin
    calls = Ref(0)
    plugin = LaSRPlugin(;
        verbose=false,
        llm_generate=mock_llm(calls, "[\"x1 + 1\", \"x1 * x1\"]"),
        variable_names=Dict(1 => "x1"),
        prompts_dir=joinpath(pkgdir(LibraryAugmentedSymbolicRegression), "prompts") * "/",
        crossover_probability=1.0,
    )
    # crossover_probability=1.0 pins SubtreeCrossover to weight 0, so LLM crossover fires on
    # every crossover event. A larger population/cycle count than the mutation testset is
    # used so at least one crossover event occurs in this tiny deterministic search.
    options = Options(;
        binary_operators=[+, *],
        plugins=(plugin,),
        default_plugins=(),
        populations=1,
        population_size=20,
        tournament_selection_n=3,
        ncycles_per_iteration=10,
        crossover_probability=1.0,
        maxsize=10,
        seed=12,
        deterministic=true,
        save_to_file=false,
    )
    X = reshape(collect(range(-1.0, 1.0; length=20)), 1, :)
    y = vec(X) .^ 2
    hof = equation_search(X, y; options, niterations=2, parallelism=:serial)
    @test hof isa HallOfFame
    @test calls[] > 0
end
