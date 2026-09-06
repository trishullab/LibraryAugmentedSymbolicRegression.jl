# Test: concept evolution and generation (Concepts.jl).
# What is it supposed to do? concept_evolution distills one overflow idea through the LLM.
# generate_concepts turns the Pareto frontier into new ideas.
# What do we hope to learn from the tests implemented here? The concept path runs and
# returns the distilled idea. It was the darkest module (66.7%): a fresh store has no
# overflow, so concept_evolution stopped at its first guard and its body never ran.
using Test
using LibraryAugmentedSymbolicRegression.ConceptsModule: concept_evolution, generate_concepts
using LibraryAugmentedSymbolicRegression.IdeaStoreModule: WindowedIdeaStore
using LibraryAugmentedSymbolicRegression.PluginModule: LaSRPlugin
using SymbolicRegression: Options

include("test_helpers.jl")  # provides mock_llm(calls, content)

@testset "concept_evolution distills an overflow idea through the LLM" begin
    calls = Ref(0)
    # window=1 with 3 seeded ideas gives a non-empty evolution pool, so the function runs
    # past its guard and queries the model.
    plugin = LaSRPlugin(;
        llm_generate=mock_llm(calls, "[\"a merged, higher-level concept\"]"),
        use_concepts=true,
        num_generated_concepts=3,
        variable_names=Dict(1 => "x0"),
        idea_store=WindowedIdeaStore(; window=1, seed=["raw a", "raw b", "raw c"]),
    )
    options = Options(; binary_operators=[+], plugins=(plugin,))
    @test concept_evolution(options) == "a merged, higher-level concept"
    @test calls[] == 1   # one batched LLM call
end

@testset "concept_evolution stops when nothing has overflowed the window" begin
    calls = Ref(0)
    plugin = LaSRPlugin(;
        llm_generate=mock_llm(calls, "[\"unused\"]"),
        use_concepts=true,
        variable_names=Dict(1 => "x0"),
        idea_store=WindowedIdeaStore(; window=10, seed=["only one idea"]),
    )
    options = Options(; binary_operators=[+], plugins=(plugin,))
    @test concept_evolution(options) === nothing
    @test calls[] == 0   # it returns before any LLM call
end

@testset "generate_concepts returns nothing without a dominating frontier" begin
    calls = Ref(0)
    plugin = LaSRPlugin(;
        llm_generate=mock_llm(calls, "[\"unused\"]"),
        use_concepts=true,
        variable_names=Dict(1 => "x0"),
    )
    options = Options(; binary_operators=[+], plugins=(plugin,))
    @test generate_concepts(nothing, nothing, options) === nothing
    @test calls[] == 0
end
