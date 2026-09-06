module SuggestionCacheModule

using Base.Threads: ReentrantLock
using DispatchDoctor: @unstable

export SuggestionCache, take_suggestion!, store_suggestions!, cache_stats, reset_cache!

"""
    SuggestionCache(; capacity=8192)

A pool of not-yet-used LLM suggestions, keyed by the prompt that produced them.

Two properties of LaSR's search make this worth far more than an ordinary cache:

1. **Every call already asks for `num_generated_equations` suggestions and uses one.**
   The other N-1 were parsed and thrown away, so serving them to later requests costs
   nothing and removes N-1 round trips outright.

2. **Structurally identical parents collapse to one key.** `render_expr` abstracts
   constants to `C`, so expressions differing only in their fitted constants share a
   prompt. Measured benefit is modest -- populations stay fairly diverse, so observed
   hit rates are around 20% -- but it is free: the suggestions were already paid for.

Entries are consumed rather than merely read: each suggestion is handed out once, so
the population still sees varied material instead of the same expression repeatedly.
When a key is exhausted the next request falls through to a real LLM call, which also
refreshes the pool.
"""
struct SuggestionCache
    pools::Dict{UInt64,Vector{String}}
    lock::ReentrantLock
    capacity::Int
    stats::Dict{Symbol,Int}
end

function SuggestionCache(; capacity::Int=8192)
    capacity > 0 || throw(ArgumentError("`capacity` must be positive."))
    return SuggestionCache(
        Dict{UInt64,Vector{String}}(),
        ReentrantLock(),
        capacity,
        Dict(:hits => 0, :misses => 0, :stored => 0, :evicted => 0),
    )
end

"""
    cache_key(mode, parts...)

Hash the prompt-determining inputs of a request.

`parts` must include everything that changes the answer -- the rendered expression(s)
and the sampled concepts -- so two requests share a key only when the same prompt
would have been sent.
"""
cache_key(mode::AbstractString, parts...) = hash((mode, parts...))

"""
    take_suggestion!(cache, key) -> Union{String,Nothing}

Consume one pooled suggestion for `key`, or `nothing` if the pool is empty.
"""
# Returns `Union{Nothing,String}` by design -- an empty pool must be distinguishable
# from a suggestion. Annotated so DispatchDoctor does not raise under
# `dispatch_doctor_mode = "error"`, where the error would be caught by the LLM
# operators' own `try`/`catch` and silently disable caching.
@unstable function take_suggestion!(cache::SuggestionCache, key::UInt64)
    return lock(cache.lock) do
        pool = get(cache.pools, key, nothing)
        if pool === nothing || isempty(pool)
            cache.stats[:misses] += 1
            return nothing
        end
        cache.stats[:hits] += 1
        suggestion = pop!(pool)
        isempty(pool) && delete!(cache.pools, key)
        return suggestion
    end
end

"""
    store_suggestions!(cache, key, suggestions)

Pool the suggestions a call produced but did not use.
"""
function store_suggestions!(cache::SuggestionCache, key::UInt64, suggestions)
    isempty(suggestions) && return nothing
    lock(cache.lock) do
        # Evict when full. Random eviction rather than LRU: pools are small and
        # short-lived, and tracking recency would cost more than it saves here.
        while length(cache.pools) >= cache.capacity
            delete!(cache.pools, first(keys(cache.pools)))
            cache.stats[:evicted] += 1
        end
        cache.pools[key] = collect(String, suggestions)
        cache.stats[:stored] += length(suggestions)
    end
    return nothing
end

"""Snapshot of hit/miss counters, for benchmarking and debugging."""
function cache_stats(cache::SuggestionCache)
    return lock(cache.lock) do
        stats = copy(cache.stats)
        total = stats[:hits] + stats[:misses]
        stats[:hit_rate_pct] = total == 0 ? 0 : round(Int, 100 * stats[:hits] / total)
        stats[:live_keys] = length(cache.pools)
        return stats
    end
end

function reset_cache!(cache::SuggestionCache)
    lock(cache.lock) do
        empty!(cache.pools)
        for k in keys(cache.stats)
            cache.stats[k] = 0
        end
    end
    return nothing
end

end # module
