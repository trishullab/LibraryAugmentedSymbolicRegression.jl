using Test
using LibraryAugmentedSymbolicRegression
using LibraryAugmentedSymbolicRegression.LLMOptionsStructModule: LaSRPlugin
using SymbolicRegression: Options, Dataset, compute_complexity
using SymbolicRegression.PopMemberModule: PopMember
using SymbolicRegression.PopulationModule: Population
using Random: Xoshiro

@testset "amnesty optimizes constants of complex members" begin
    rng = Xoshiro(0)
    X = rand(rng, 2, 200)
    y = 2 .* X[1, :] .+ cos.(X[2, :])
    dataset = Dataset(X, y)

    # `default_plugins=()` keeps the LaSRPlugin at index 1 so `only(...)` + `init_plugin_state`
    # line up (mirrors test_lasr_generate_operator.jl); `variable_names` maps x0/x1 to features
    # 1/2 so the parsed expression references data columns rather than falling back to a constant.
    # `amnesty_complexity=3` is below the member's complexity (6) so amnesty triggers.
    opts = Options(;
        binary_operators=[+, *],
        unary_operators=[cos],
        optimizer_nrestarts=3,
        default_plugins=(),
        plugins=(
            LaSRPlugin(;
                use_llm=false,
                amnesty_complexity=3,
                variable_names=Dict(1 => "x0", 2 => "x1"),
            ),
        ),
    )

    # A structurally-CORRECT member carrying a DELIBERATELY WRONG constant (9.0 vs the true
    # 2.0) → high loss. Only constant optimization can rescue it; the structure is already right.
    bad = parse_expr(Float64, "9.0 * x0 + cos(x1)", opts)
    m_bad = PopMember(dataset, bad, opts; deterministic=false)
    loss_before = m_bad.loss
    @test loss_before > 1e-3
    # The member genuinely clears the amnesty threshold (guards against a trivially-passing test).
    @test compute_complexity(m_bad.tree, opts) >= 3

    pop = Population([m_bad])

    plugin = only(filter(p -> p isa LaSRPlugin, opts.plugins))
    state = SymbolicRegression.init_plugin_state(plugin, opts, dataset)

    # Real SR.jl beta.2 signature: (state, plugin, search_state, dataset, options, ropt,
    # returned_pop). `use_llm=false`, yet amnesty must still run because amnesty_complexity > 0.
    # search_state/ropt are untouched on this path, so `nothing` is safe.
    SymbolicRegression.on_generation_end!(state, plugin, nothing, dataset, opts, nothing, pop)

    # Constants were optimized in place (9.0 → ~2.0): loss genuinely drops toward zero.
    @test pop.members[1].loss < loss_before
    @test pop.members[1].loss < 1e-6
end

@testset "amnesty_complexity is honored on LaSRPlugin" begin
    plugin = LaSRPlugin(; use_llm=false, amnesty_complexity=7)
    @test plugin.amnesty_complexity == 7
    @test LaSRPlugin(; use_llm=false).amnesty_complexity == 0  # opt-in default
end
