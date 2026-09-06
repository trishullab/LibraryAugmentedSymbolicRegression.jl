module IdeaStoreModule

using Random: randperm, shuffle

export AbstractIdeaStore,
    WindowedIdeaStore, ScoredIdeaStore, add_idea!, retrieve_ideas, evolution_candidates

"""
    AbstractIdeaStore

The concept library that LaSR fills during a search. A subtype must supply these
methods:

- `add_idea!(store, idea; refined=false)`: add one idea. `refined=true` marks a concept
  that concept evolution merged. A store can prefer such a concept.
- `retrieve_ideas(store, n; query=nothing)`: return `n` ideas at most. A store that
  supports `query` returns the ideas that are relevant to it. `query` is usually the
  expression that the search mutates now, so such a store gives concepts about the terms
  in use.
- `evolution_candidates(store)`: return the ideas that concept evolution can merge or
  distill. For the windowed store these are the ideas past the sampling window.
- `Base.length(store)`: return the number of ideas that the store holds.
"""
abstract type AbstractIdeaStore end

Base.isempty(store::AbstractIdeaStore) = length(store) == 0

"""
    evolution_candidates(store::AbstractIdeaStore) -> Vector{String}

Return no candidates. A subtype can replace this method.
"""
evolution_candidates(store::AbstractIdeaStore) = String[]

"""
    WindowedIdeaStore(; window=30, seed=String[])

Ideas are stored in a vector; refined (merged) ideas are pushed to the front and raw ideas to the back. Retrieval draws a uniform sample from the front `window` ideas, so recently-refined concepts dominate. `query` is ignored.

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

"""
    _bm25_scores(corpus, query_terms, k1, b) -> Vector{Float64}

The BM25 score of each document in `corpus` against `query_terms`. `k1` sets the term
frequency saturation, and `b` sets the length normalization.
"""
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

"""
    ScoredIdeaStore(; k1=1.5, b=0.75, decay=0.99, refined_prior=2.0, seed=String[])

An idea store that carries a mutable **value** per idea and retrieves by `value * (1 + BM25 relevance to the query)`.
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

"""
    evolution_candidates(store::ScoredIdeaStore) -> Vector{String}

Return the ideas with a value at or below the median. These are the low-value ideas, so
concept evolution must try to distill or replace them.
"""
function evolution_candidates(store::ScoredIdeaStore)
    isempty(store.ideas) && return String[]
    med = _median(store.values)
    return [store.ideas[i] for i in eachindex(store.ideas) if store.values[i] <= med]
end

# ponytail: hand-rolled to avoid a `Statistics` dep entry for one call; swap in
# `Statistics.median` if anything else in the package ever needs Statistics.
function _median(v::AbstractVector{<:Real})
    s = sort(v)
    m = length(s)
    return isodd(m) ? Float64(s[(m + 1) ÷ 2]) : (s[m ÷ 2] + s[m ÷ 2 + 1]) / 2
end

end # module
