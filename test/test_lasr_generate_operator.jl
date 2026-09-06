using Test
using LibraryAugmentedSymbolicRegression
using LibraryAugmentedSymbolicRegression.LLMOperatorsModule: llm_generate_candidates
using LibraryAugmentedSymbolicRegression.PluginModule: LaSRPlugin
using SymbolicRegression: SymbolicRegression, Options

include("test_helpers.jl")  # provides mock_llm(calls, content)

@testset "llm_generate_candidates returns all usable proposals" begin
    calls = Ref(0)
    # three proposals; "1.0" is not usable (no feature), the other two are
    content = "[\"x0 + x1\", \"1.0\", \"x0 * cos(x1)\"]"
    # Register x0/x1 as features 1/2 so the proposals parse to feature-referencing trees;
    # with the default variable names DynamicExpressions rejects `x0`/`x1` and the parser
    # substitutes a constant, which would (correctly) be filtered as unusable.
    plugin = LaSRPlugin(;
        verbose=false,
        llm_generate=mock_llm(calls, content),
        use_llm=true,
        num_generated_equations=3,
        variable_names=Dict(1 => "x0", 2 => "x1"),
    )
    options = Options(; binary_operators=[+, *], unary_operators=[cos], plugins=(plugin,))
    # A state-free context is sufficient here: with `state === nothing` the LaSRContext
    # reads `idea_store`/`variable_names` straight from the plugin. (Building a real
    # `LaSRPluginState` is exercised in the mutate! test below.)
    ctx = LibraryAugmentedSymbolicRegression.PluginModule.lasr_context(options)
    cands = llm_generate_candidates(ctx, 20, 2, Float64)
    @test length(cands) == 2   # the two usable ones
    @test calls[] == 1         # a single batched call
end

using SymbolicRegression: plugin_mutations
@testset "LLMGenerateMutation injected at generate_weight" begin
    plugin = LaSRPlugin(;
        use_llm=true, mutate_weight=0.0, randomize_weight=0.0, generate_weight=1.5
    )
    muts = plugin_mutations(plugin)
    kinds = [typeof(first(p)) for p in muts]
    @test LLMGenerateMutation in kinds
    @test any(p -> first(p) isa LLMGenerateMutation && last(p) == 1.5, muts)
    # backward compat: off by default
    @test !any(
        p -> first(p) isa LLMGenerateMutation, plugin_mutations(LaSRPlugin(; use_llm=true))
    )
end

using SymbolicRegression: mutate!, Dataset
using SymbolicRegression.PopMemberModule: PopMember
using Random: Xoshiro
@testset "LLMGenerateMutation keep-best-of-K beats a single draw" begin
    # Target y = 2*x0 + cos(x1). This proves the one thing LLMGenerateMutation adds over
    # single-shot LLMRandomizeMutation: best-of-K STRUCTURE selection. Both operators call
    # `optimize_constants`, which already does `optimizer_nrestarts` multi-restart
    # internally, so multi-restart is NOT the differentiator -- structure selection is.
    rng = Xoshiro(0)
    X = rand(rng, 2, 200)
    y = 2 .* X[1, :] .+ cos.(X[2, :])
    dataset = Dataset(X, y)
    correct = "2.0 * x0 + cos(x1)"

    # `x0`/`x1` registered as features 1/2 so proposals parse to feature-referencing trees;
    # `default_plugins=()` keeps the LaSRPlugin at index 1 so `plugin_states=(state,)` lines
    # up. (Bare-symbol proposals like "x0" are avoided: the parser's `_rhs_of_assignment`
    # has no Symbol method, so a lone variable throws.)
    make_opts(mock) = Options(;
        binary_operators=[+, *],
        unary_operators=[cos],
        optimizer_nrestarts=3,
        default_plugins=(),
        plugins=(
            LaSRPlugin(;
                verbose=false,
                llm_generate=mock,
                use_llm=true,
                num_generated_equations=3,
                generate_weight=1.0,
                randomize_weight=1.0,
                variable_names=Dict(1 => "x0", 2 => "x1"),
            ),
        ),
    )
    # A real LaSRPluginState is required: `lasr_state(opts, (nothing,))` would throw a
    # TypeError, so we construct the state the search itself would build.
    make_states(opts) = (
        SymbolicRegression.init_plugin_state(
            only(filter(p -> p isa LaSRPlugin, opts.plugins)), opts, dataset
        ),
    )

    # GENERATE mock: a K=3 batch with the correct skeleton in a NON-TERMINAL (middle) slot,
    # flanked by strictly-worse skeletons. best-of-K must compare all K and keep the lowest
    # loss -- this kills both a keep-first and a keep-last confound, either of which would
    # return a wrong skeleton here and fail the assertion below.
    gen_calls = Ref(0)
    gen_opts = make_opts(mock_llm(gen_calls, "[\"x0 * x1\", \"$correct\", \"x0 * x0\"]"))
    gen_states = make_states(gen_opts)

    # RANDOMIZE mock: a SEPARATE, single WRONG skeleton, so the single draw cannot stumble
    # onto the correct one; constant fitting cannot rescue wrong structure.
    rand_calls = Ref(0)
    rand_opts = make_opts(mock_llm(rand_calls, "[\"x0 * x1\"]"))
    rand_states = make_states(rand_opts)

    parent_tree = parse_expr(Float64, "x0 * x1", gen_opts)
    parent = PopMember(dataset, parent_tree, gen_opts; deterministic=false)

    res_gen = mutate!(
        copy(parent_tree),
        parent,
        LLMGenerateMutation(),
        gen_opts;
        dataset=dataset,
        plugin_states=gen_states,
        curmaxsize=30,
        nfeatures=2,
        recorder=Dict{String,Any}(),
    )
    # best-of-K picked the exact (non-terminal) skeleton and fit its constants.
    @test res_gen.member.loss < 1e-6
    @test gen_calls[] == 1   # a single batched K-call

    res_rand = mutate!(
        copy(parent_tree),
        parent,
        LLMRandomizeMutation(),
        rand_opts;
        dataset=dataset,
        plugin_states=rand_states,
        curmaxsize=30,
        nfeatures=2,
        recorder=Dict{String,Any}(),
    )
    # The single draw got a wrong skeleton; its fitted loss stays high. Real, deterministic,
    # non-tautological bound (independent of `res_gen`'s value).
    @test res_rand.member.loss > 1e-3
end

@testset "LLMGenerateMutation reaches options.mutations via the plugin" begin
    # The intended entry point: a LaSRPlugin passed to Options contributes its weighted
    # mutations through `plugin_mutations`, which SR merges as defaults.
    opts = Options(;
        binary_operators=[+, *],
        unary_operators=[cos],
        default_plugins=(),
        plugins=(LaSRPlugin(; use_llm=true, generate_weight=1.25),),
    )
    @test any(p -> first(p) isa LLMGenerateMutation && last(p) == 1.25, opts.mutations)
    # Opt-in: default generate_weight=0.0 means the operator is not contributed at all.
    off = Options(;
        binary_operators=[+, *], default_plugins=(), plugins=(LaSRPlugin(; use_llm=true),)
    )
    @test all(p -> !(first(p) isa LLMGenerateMutation), off.mutations)
end
