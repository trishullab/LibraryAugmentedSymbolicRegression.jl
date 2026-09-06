module SuggestionCacheModule

using Base.Threads: ReentrantLock
using DispatchDoctor: @unstable

export SuggestionCache, take_suggestion!, store_suggestions!, cache_stats

"""
    SuggestionCache(; capacity=8192)

A pool of LLM suggestions that no operator used yet, organized by the parent prompt. Useful when the same prompt is requested multiple times.

Two properties of the LaSR search make this pool worth much more than an ordinary cache:

1. **Each call already asks for `num_generated_equations` suggestions and uses one.** LaSR parsed the other N-1 suggestions and then discarded them. To give them to a later request costs nothing and removes N-1 round trips.

2. **Structurally equal parents give one key.** `render_expr` writes each constant as `C`, so two expressions that differ only in their fitted constants share a prompt. The measured gain is small, because the populations stay diverse: the observed hit rate is about 20%. The gain is still free, because the calls already paid for the suggestions.

Suggestions are deleted after use, so the population does not see the same expression repeatedly. When a key holds no more suggestions, the next request makes a real LLM call, which replenishes the pool.
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

Hashes the prompt.

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
        return cache.stats[:stored] += length(suggestions)
    end
    return nothing
end

"""
    cache_stats(cache) -> Dict{Symbol,Int}

A snapshot of the counters, for a benchmark or for debugging: `:hits`, `:misses`,
`:stored`, `:evicted`, the derived `:hit_rate_pct`, and `:live_keys`.
"""
function cache_stats(cache::SuggestionCache)
    return lock(cache.lock) do
        stats = copy(cache.stats)
        total = stats[:hits] + stats[:misses]
        stats[:hit_rate_pct] = total == 0 ? 0 : round(Int, 100 * stats[:hits] / total)
        stats[:live_keys] = length(cache.pools)
        return stats
    end
end

end # module
