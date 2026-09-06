# Test: WindowedIdeaStore, the default concept store.
# What is it supposed to do? It keeps ideas in one list. It adds a refined idea to the
# front and a raw idea to the back. It samples retrieval from the front window. It gives
# the ideas past the window back as evolution candidates.
# What do we hope to learn from the tests implemented here? The store obeys this contract.
# Only the non-default ScoredIdeaStore had a test before. The default store had none.
using Test
using Random
using LibraryAugmentedSymbolicRegression.IdeaStoreModule:
    WindowedIdeaStore, AbstractIdeaStore, add_idea!, retrieve_ideas, evolution_candidates

@testset "WindowedIdeaStore (the default store)" begin
    @testset "empty store and seed constructor" begin
        s = WindowedIdeaStore()
        @test s isa AbstractIdeaStore
        @test isempty(s) && length(s) == 0
        @test retrieve_ideas(s, 3) == String[]
        @test evolution_candidates(s) == String[]

        seeded = WindowedIdeaStore(; window=2, seed=["a", "b", "c"])
        @test length(seeded) == 3
    end

    @testset "add_idea! sends a raw idea to the back and a refined idea to the front" begin
        s = WindowedIdeaStore(; window=10)
        add_idea!(s, "raw1")
        add_idea!(s, "raw2")
        add_idea!(s, "merged"; refined=true)
        @test s.ideas == ["merged", "raw1", "raw2"]
        @test length(s) == 3
    end

    @testset "retrieve_ideas samples the front window only, distinct and capped" begin
        s = WindowedIdeaStore(; window=3, seed=["a", "b", "c", "d", "e"])
        r = retrieve_ideas(s, 2)
        @test length(r) == 2 && length(unique(r)) == 2
        # d and e are past the window. They must not appear.
        @test all(x -> x in ["a", "b", "c"], retrieve_ideas(s, 3))
        @test length(retrieve_ideas(s, 99)) == 3   # capped at the window
        short = WindowedIdeaStore(; window=10, seed=["only1", "only2"])
        @test length(retrieve_ideas(short, 99)) == 2   # capped at the store length
    end

    @testset "evolution_candidates are the ideas past the window" begin
        s = WindowedIdeaStore(; window=2, seed=["a", "b", "c", "d"])
        @test evolution_candidates(s) == ["c", "d"]
        @test evolution_candidates(WindowedIdeaStore(; window=5, seed=["a", "b"])) ==
            String[]
    end
end
