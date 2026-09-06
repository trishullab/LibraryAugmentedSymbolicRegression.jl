module ParseFailuresModule

"""
    ParseFailure(raw, normalized, stage, reason)

One record of an LLM expression string that `parse_expr` could not read.

- `raw` is the string as the LLM sent it.
- `normalized` is the same string after the normalization rules ran.
- `stage` is the step that failed: `:meta_parse`, `:expr_stage`, or `:tree_parse`.
- `reason` is the error that the step reported.
"""
struct ParseFailure
    raw::String
    normalized::String
    stage::Symbol
    reason::String
end

"""
    ParseFailureStore(; cap=500)

A record of the LLM strings that `parse_expr` could not read.

`parse_expr` returns a constant-1 tree for each unreadable string. That fallback is
cheap and keeps the search alive, but it hides the reason the LLM output was unusable.
This store keeps each fallback, so a scientist can see the shape of string that stops
the parser. The scientist can then add a `NormalizationRule` (see `resolve_rules` in
`NormalizationRules.jl`) to correct the cause instead of accepting the fallback forever.
The store keeps `cap` records at most, so a pathological run cannot leak memory.

The store is a thread-safe ring buffer. `fork_plugin_state` and
`refresh_worker_plugin_state` (see `src/SRInterface.jl`) pass it by reference, so every
worker of a `:serial` or `:multithreading` run writes into one store. The workers of a
`:multiprocessing` run use separate address spaces, so each worker gets its own store.
Aggregation across processes is a documented follow-up; the `lasr_logger` route already
gives observability across processes.
"""
mutable struct ParseFailureStore
    records::Vector{ParseFailure}
    cap::Int
    lock::ReentrantLock
end
ParseFailureStore(; cap::Int=500) = ParseFailureStore(ParseFailure[], cap, ReentrantLock())

"""
    record_parse_failure!(store, failure)

Add one `ParseFailure` to `store`. Drop the oldest record when the store is full.
"""
function record_parse_failure!(s::ParseFailureStore, f::ParseFailure)
    lock(s.lock) do
        push!(s.records, f)
        return length(s.records) > s.cap && popfirst!(s.records)
    end
    return nothing
end

"""
    parse_failures(store) -> Vector{ParseFailure}

Return a copy of the records that `store` holds, oldest first.
"""
parse_failures(s::ParseFailureStore) = lock(() -> copy(s.records), s.lock)

"""
    parse_failure_summary(store; n=10) -> Vector{Pair{String,Int}}

Return the `n` raw strings that failed most often, with a count for each. The most
frequent string comes first.
"""
function parse_failure_summary(s::ParseFailureStore; n::Int=10)
    counts = Dict{String,Int}()
    for f in parse_failures(s)
        counts[f.raw] = get(counts, f.raw, 0) + 1
    end
    return first(sort!(collect(counts); by=last, rev=true), min(n, length(counts)))
end

export ParseFailure,
    ParseFailureStore, record_parse_failure!, parse_failures, parse_failure_summary

end # module
