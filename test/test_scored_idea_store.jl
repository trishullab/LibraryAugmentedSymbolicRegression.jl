using Test
using Random
using LibraryAugmentedSymbolicRegression.IdeaStoreModule:
    ScoredIdeaStore,
    AbstractIdeaStore,
    add_idea!,
    retrieve_ideas,
    update_idea_value!,
    evolution_candidates

Random.seed!(0)

@testset "ScoredIdeaStore" begin
    @testset "add / length / empty-safe" begin
        s = ScoredIdeaStore()
        @test s isa AbstractIdeaStore
        @test retrieve_ideas(s, 3) == String[]
        for w in ["oscillatory cos", "exponential decay", "linear x0", "power law"]
            add_idea!(s, w)
        end
        @test length(s) == 4
    end

    @testset "update_idea_value! reweights retrieval" begin
        s = ScoredIdeaStore()
        for w in ["oscillatory cos", "exponential decay", "linear x0", "power law"]
            add_idea!(s, w)
        end
        update_idea_value!(s, "power law", 30.0)
        cnt = Dict{String,Int}()
        for _ in 1:3000, idea in retrieve_ideas(s, 1)
            cnt[idea] = get(cnt, idea, 0) + 1
        end
        @test get(cnt, "power law", 0) > 2400
    end

    @testset "query relevance surfaces the relevant idea" begin
        s = ScoredIdeaStore()
        for w in ["oscillatory motion with cos", "exponential decay term", "linear scaling"]
            add_idea!(s, w)
        end
        cnt = Dict{String,Int}()
        for _ in 1:3000, idea in retrieve_ideas(s, 1; query="x0 * cos(x1)")
            cnt[idea] = get(cnt, idea, 0) + 1
        end
        @test get(cnt, "oscillatory motion with cos", 0) >
            get(cnt, "exponential decay term", 0)
    end

    @testset "refined ideas enter with a higher prior" begin
        s = ScoredIdeaStore()
        add_idea!(s, "raw one")
        add_idea!(s, "refined one"; refined=true)
        cnt = Dict{String,Int}()
        for _ in 1:3000, idea in retrieve_ideas(s, 1)
            cnt[idea] = get(cnt, idea, 0) + 1
        end
        @test get(cnt, "refined one", 0) > get(cnt, "raw one", 0)
    end

    @testset "evolution_candidates are the low-value ideas" begin
        s = ScoredIdeaStore()
        for w in ["a", "b", "c", "d"]
            add_idea!(s, w)
        end
        update_idea_value!(s, "a", 10.0)
        update_idea_value!(s, "b", 10.0)
        ec = evolution_candidates(s)
        @test ("c" in ec) && ("d" in ec) && !("a" in ec)
    end

    @testset "retrieval returns n distinct, capped at length" begin
        s = ScoredIdeaStore()
        for w in ["a", "b", "c", "d"]
            add_idea!(s, w)
        end
        r = retrieve_ideas(s, 3)
        @test length(r) == 3 && length(unique(r)) == 3
        @test length(retrieve_ideas(s, 99)) == 4
    end
end
