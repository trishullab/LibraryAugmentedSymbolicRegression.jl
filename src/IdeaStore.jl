module IdeaStoreModule

using Random: randperm, shuffle

export AbstractIdeaStore,
    WindowedIdeaStore,
    ScoredIdeaStore,
    add_idea!,
    retrieve_ideas,
    update_idea_value!,
    evolution_candidates

"""
    AbstractIdeaStore

The concept library LaSR accumulates during a search. A subtype implements:

- `add_idea!(store, idea; refined=false)` — add one idea. `refined=true` marks a merged
  concept (the output of concept evolution) that the store may choose to prefer.
- `retrieve_ideas(store, n; query=nothing)` — return up to `n` ideas, relevant to `query`
  when the store supports it. `query` is typically the expression currently being mutated,
  so a query-aware store surfaces concepts about the terms in play.
- `evolution_candidates(store)` — the ideas eligible to be merged/distilled by concept
  evolution (for the windowed store, everything past the sampling window).
- `Base.length(store)` — number of ideas held.
"""
abstract type AbstractIdeaStore end

Base.isempty(store::AbstractIdeaStore) = length(store) == 0

# By default, fallbacks to an empty list of evolution candidates. Subtypes can override.
evolution_candidates(store::AbstractIdeaStore) = String[]

"""
    WindowedIdeaStore(; window=30, seed=String[])

Ideas live in one list; refined (merged) ideas are pushed to the front and raw ideas to the back. Retrieval draws a distinct uniform sample from the front `window` ideas, so recently-refined concepts dominate. `query` is ignored.

Ideas beyond the window are the `evolution_candidates`: overflow that concept evolution distills back into new front-of-list ideas.
"""
struct WindowedIdeaStore <: AbstractIdeaStore
    ideas::Vector{String}
    window::Int
end

function WindowedIdeaStore(; window::Int=30, seed=String[])
    return WindowedIdeaStore(collect(String, seed), window)
end

Base.length(store::WindowedIdeaStore) = length(store.ideas)

function add_idea!(store::WindowedIdeaStore, idea::AbstractString; refined::Bool=false)
    if refined
        pushfirst!(store.ideas, String(idea))
    else
        push!(store.ideas, String(idea))
    end
    return nothing
end

function retrieve_ideas(
    store::WindowedIdeaStore, n::Integer; query::Union{AbstractString,Nothing}=nothing
)
    isempty(store.ideas) && return String[]
    hi = min(store.window, length(store.ideas))
    k = min(Int(n), hi)
    # Distinct uniform sample from the front window, in randomized order -- exactly what
    # the previous sample_context/sample_one_context pair produced.
    return store.ideas[randperm(hi)[1:k]]
end

function evolution_candidates(store::WindowedIdeaStore)
    length(store.ideas) <= store.window && return String[]
    return store.ideas[(store.window + 1):end]
end

function _tokenize(s::AbstractString)::Vector{String}
    return String.(filter(!isempty, split(lowercase(s), r"[^a-z0-9_]+")))
end

"""Core BM25 scoring over a token corpus."""
function _bm25_scores(
    corpus::Vector{Vector{String}}, query_terms::Vector{String}, k1::Float64, b::Float64
)::Vector{Float64}
    N = length(corpus)
    N == 0 && return Float64[]
    doc_len = Float64[length(t) for t in corpus]
    avgdl = max(sum(doc_len) / N, eps())

    # Document frequency for each query term.
    df = Dict{String,Int}()
    for term in unique(query_terms)
        df[term] = count(toks -> term in toks, corpus)
    end

    scores = zeros(Float64, N)
    for (i, toks) in enumerate(corpus)
        for term in query_terms
            n_t = get(df, term, 0)
            n_t == 0 && continue
            f = count(==(term), toks)
            f == 0 && continue
            idf = log(1 + (N - n_t + 0.5) / (n_t + 0.5))
            denom = f + k1 * (1 - b + b * doc_len[i] / avgdl)
            scores[i] += idf * (f * (k1 + 1)) / denom
        end
    end
    return scores
end

# ---------------------------------------------------------------------------------------
# ScoredIdeaStore: quality-weighted retrieval that can be UPDATED from search outcomes.
# ---------------------------------------------------------------------------------------

"""
    ScoredIdeaStore(; k1=1.5, b=0.75, decay=0.99, refined_prior=2.0, seed=String[])

An idea store that carries a mutable **value** per idea and retrieves by
`value * (1 + BM25 relevance to the query)`. It addresses both weaknesses of the earlier
stores: the windowed store ignores the query (uniform random) and a purely lexical store weights
only lexical overlap with no notion of which ideas have actually been *useful*.

The value is the "updating" half the search can drive: call `update_idea_value!(store,
idea, delta)` to reinforce ideas that preceded an improved member and penalize ones that
did not, turning the concept library into a bandit over concepts rather than a passive log.
New ideas enter with value `1.0` (or `refined_prior` for distilled/merged concepts), and
every `add_idea!` multiplies existing values by `decay` so stale concepts fade unless
reinforced. Retrieval is stochastic (Efraimidis–Spirakis weighted sampling without
replacement), so high-value/relevant ideas dominate while exploration continues.

Drop-in for the other stores: only `retrieve_ideas`/`add_idea!` change from the search's
side; `update_idea_value!` is an additional, optional lever.
"""
struct ScoredIdeaStore <: AbstractIdeaStore
    ideas::Vector{String}
    tokens::Vector{Vector{String}}   # cached tokenization, parallel to `ideas`
    values::Vector{Float64}          # mutable quality score, parallel to `ideas`
    k1::Float64
    b::Float64
    decay::Float64
    refined_prior::Float64
end

function ScoredIdeaStore(;
    k1::Float64=1.5,
    b::Float64=0.75,
    decay::Float64=0.99,
    refined_prior::Float64=2.0,
    seed=String[],
)
    store = ScoredIdeaStore(
        String[], Vector{Vector{String}}(), Float64[], k1, b, decay, refined_prior
    )
    for idea in seed
        add_idea!(store, idea)
    end
    return store
end

Base.length(store::ScoredIdeaStore) = length(store.ideas)

function add_idea!(store::ScoredIdeaStore, idea::AbstractString; refined::Bool=false)
    # Decay existing values so the library forgets concepts that stop being reinforced.
    if store.decay != 1.0 && !isempty(store.values)
        store.values .*= store.decay
    end
    push!(store.ideas, String(idea))
    push!(store.tokens, _tokenize(idea))
    push!(store.values, refined ? store.refined_prior : 1.0)
    return nothing
end

"""
    update_idea_value!(store, idea, delta)

Reinforce (`delta > 0`) or penalize (`delta < 0`) an idea by content. No-op if the idea is
absent. Values are floored at a small positive constant so a penalized idea can still be
resampled (and later redeemed) rather than being permanently zeroed out.
"""
function update_idea_value!(store::ScoredIdeaStore, idea::AbstractString, delta::Real)
    idx = findfirst(==(String(idea)), store.ideas)
    idx === nothing && return nothing
    store.values[idx] = max(store.values[idx] + Float64(delta), 1e-3)
    return nothing
end

function retrieve_ideas(
    store::ScoredIdeaStore, n::Integer; query::Union{AbstractString,Nothing}=nothing
)
    isempty(store.ideas) && return String[]
    k = min(Int(n), length(store.ideas))

    rel = zeros(Float64, length(store.ideas))
    if query !== nothing
        q = _tokenize(query)
        if !isempty(q)
            rel = _bm25_scores(store.tokens, q, store.k1, store.b)
        end
    end
    weights = store.values .* (1.0 .+ rel)

    # Efraimidis–Spirakis: sampling key rand()^(1/w) gives weighted sampling without
    # replacement; taking the top-k keys draws k distinct ideas ∝ their weights.
    keys = [w <= 0 ? -Inf : rand()^(1.0 / w) for w in weights]
    order = partialsortperm(keys, 1:k; rev=true)
    return store.ideas[order]
end

# Low-value ideas are the ones concept evolution should try to distill or replace.
function evolution_candidates(store::ScoredIdeaStore)
    isempty(store.ideas) && return String[]
    med = _median(store.values)
    return [store.ideas[i] for i in eachindex(store.ideas) if store.values[i] <= med]
end

function _median(v::AbstractVector{<:Real})
    isempty(v) && return 0.0
    s = sort(v)
    m = length(s)
    return isodd(m) ? Float64(s[(m + 1) ÷ 2]) : (s[m ÷ 2] + s[m ÷ 2 + 1]) / 2
end

end # module
