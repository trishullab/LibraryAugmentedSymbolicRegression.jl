# The suggestion pool and the call budget both change how many LLM requests a search
# issues, so they need to be pinned: a regression here shows up as a slow search or a
# runaway bill, neither of which the other tests would catch.

using Test
using LibraryAugmentedSymbolicRegression:
    LaSRPlugin, SuggestionCache, CallBudget, cache_stats, budget_used
using LibraryAugmentedSymbolicRegression.SuggestionCacheModule:
    cache_key, take_suggestion!, store_suggestions!, reset_cache!
using LibraryAugmentedSymbolicRegression.CallBudgetModule: claim_call!
using SymbolicRegression: Options, equation_search

include("mock_llm_server.jl")
using .MockLLMServer: with_server, total_calls

const PROMPTS_DIR = joinpath(@__DIR__, "prompts") * "/"
const PORT = 11_452

@testset "pool hands each suggestion out exactly once" begin
    cache = SuggestionCache(; capacity=16)
    key = cache_key("mutate", "x + y")

    @test take_suggestion!(cache, key) === nothing  # empty pool

    store_suggestions!(cache, key, ["a", "b"])
    got = String[]
    for _ in 1:2
        s = take_suggestion!(cache, key)
        @test s !== nothing
        push!(got, s)
    end
    @test sort(got) == ["a", "b"]
    # Exhausted: the next request must fall through to a real call.
    @test take_suggestion!(cache, key) === nothing

    stats = cache_stats(cache)
    @test stats[:hits] == 2
    @test stats[:misses] == 2
end

@testset "distinct prompts do not share a pool" begin
    cache = SuggestionCache(; capacity=16)
    store_suggestions!(cache, cache_key("mutate", "x + y"), ["from-xy"])
    @test take_suggestion!(cache, cache_key("mutate", "x * y")) === nothing
    @test take_suggestion!(cache, cache_key("crossover", "x + y")) === nothing
    @test take_suggestion!(cache, cache_key("mutate", "x + y")) == "from-xy"
end

@testset "pool respects its capacity" begin
    cache = SuggestionCache(; capacity=4)
    for i in 1:20
        store_suggestions!(cache, cache_key("mutate", "expr$(i)"), ["s$(i)"])
    end
    @test cache_stats(cache)[:live_keys] <= 4
end

@testset "budget admits exactly its limit, then refuses" begin
    budget = CallBudget(3)
    @test count(_ -> claim_call!(budget), 1:10) == 3
    used = budget_used(budget)
    @test used.used >= 3
    @test used.denied == 7
end

@testset "a nothing limit is unbounded" begin
    budget = CallBudget(nothing)
    @test all(_ -> claim_call!(budget), 1:1000)
end

@testset "max_llm_calls is validated" begin
    @test_throws ArgumentError LaSRPlugin(; mutate_weight=1.0, max_llm_calls=0)
    @test_throws ArgumentError LaSRPlugin(; mutate_weight=1.0, max_llm_calls=-3)
end

@testset "SuggestionCache capacity is validated" begin
    # Like max_llm_calls, a bad capacity must fail at construction, not deep in a search
    # after an LLM call was already billed.
    @test_throws ArgumentError SuggestionCache(; capacity=0)
    @test_throws ArgumentError SuggestionCache(; capacity=-4)
end

@testset "budget caps the LLM calls a search actually issues" begin
    X = randn(Float64, 1, 60)
    y = @. X[1, :]^3 + X[1, :]^2 + X[1, :]

    function search(cap)
        return with_server(PORT) do url
            options = Options(;
                binary_operators=[+, -, *],
                unary_operators=Function[],
                populations=2,
                population_size=33,
                ncycles_per_iteration=20,
                progress=false,
                verbosity=0,
                plugins=(
                    LaSRPlugin(;
                        use_llm=true,
                        use_concepts=true,
                        # Force the LLM path so the budget is what limits calls, not chance.
                        mutate_weight=1.0,
                        randomize_weight=1.0,
                        crossover_probability=1.0,
                        prompts_dir=PROMPTS_DIR,
                        api_key="mock",
                        model="mock-model",
                        api_kwargs=Dict("url" => url, "max_tokens" => 512),
                        http_kwargs=Dict("retries" => 1, "readtimeout" => 60),
                        variable_names=Dict(1 => "x"),
                        verbose=false,
                        # Isolate the budget's effect from the pool's.
                        suggestion_cache=nothing,
                        max_llm_calls=cap,
                    ),
                ),
            )
            equation_search(X, y; options, niterations=2, parallelism=:serial)
            return total_calls()
        end
    end

    capped = search(5)
    uncapped = search(nothing)

    @test capped < uncapped
    @test capped <= 5
end

@testset "the pool is exercised by a real search" begin
    # Call *count* is not a sound assertion here: the search is stochastic, so pooling
    # changes which suggestions arrive and therefore how the population evolves. What is
    # meaningful is that the pool genuinely served suggestions that would otherwise have
    # been round trips.
    X = randn(Float64, 1, 60)
    y = @. X[1, :]^3 + X[1, :]^2 + X[1, :]

    cache = SuggestionCache(; capacity=512)
    with_server(PORT) do url
        options = Options(;
            binary_operators=[+, -, *],
            unary_operators=Function[],
            populations=2,
            population_size=33,
            ncycles_per_iteration=20,
            progress=false,
            verbosity=0,
            plugins=(
                LaSRPlugin(;
                    use_llm=true,
                    mutate_weight=1.0,
                    prompts_dir=PROMPTS_DIR,
                    api_key="mock",
                    model="mock-model",
                    api_kwargs=Dict("url" => url, "max_tokens" => 512),
                    http_kwargs=Dict("retries" => 1, "readtimeout" => 60),
                    variable_names=Dict(1 => "x"),
                    verbose=false,
                    suggestion_cache=cache,
                ),
            ),
        )
        equation_search(X, y; options, niterations=2, parallelism=:serial)
    end

    stats = cache_stats(cache)
    # Each call banks the proposals it did not use, and later identical prompts consume
    # them. Both must actually happen for the pool to be doing anything.
    @test stats[:stored] > 0
    @test stats[:hits] > 0
end
