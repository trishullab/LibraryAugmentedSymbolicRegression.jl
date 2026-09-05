module ParseFailuresModule

# ---- Parse failure store (observability instead of silent fallback) ----
# `parse_expr` falls back to a constant-1 tree on any unparseable LLM string. That
# fallback is cheap and keeps the search alive, but a silent fallback hides *why* the
# LLM's output was unusable. This store records each fallback occurrence (bounded, so a
# pathological run cannot leak memory) so a scientist can inspect what shape of string is
# tripping the parser and, e.g., register a `NormalizationRule` (see `resolve_rules`
# in `NormalizationRules.jl`) to fix the root cause instead of just eating the
# fallback forever.
struct ParseFailure
    raw::String
    normalized::String
    stage::Symbol             # :meta_parse | :expr_stage | :tree_parse
    reason::String
end

# Thread-safe bounded ring buffer. Held BY REFERENCE across `fork_plugin_state`/
# `refresh_worker_plugin_state` (see `src/SRInterface.jl`) so `:serial`/`:multithreading` runs
# aggregate records from every worker into one store; `:multiprocessing` runs live in
# separate address spaces so each worker gets its own store (a documented follow-up --
# the `lasr_logger` route already covers cross-process observability).
mutable struct ParseFailureStore
    records::Vector{ParseFailure}
    cap::Int
    lock::ReentrantLock
end
ParseFailureStore(; cap::Int=500) = ParseFailureStore(ParseFailure[], cap, ReentrantLock())

function record_parse_failure!(s::ParseFailureStore, f::ParseFailure)
    lock(s.lock) do
        push!(s.records, f)
        length(s.records) > s.cap && popfirst!(s.records)
    end
    return nothing
end

parse_failures(s::ParseFailureStore) = lock(() -> copy(s.records), s.lock)

function parse_failure_summary(s::ParseFailureStore; n::Int=10)
    counts = Dict{String,Int}()
    for f in parse_failures(s)
        counts[f.raw] = get(counts, f.raw, 0) + 1
    end
    return first(sort!(collect(counts); by=last, rev=true), min(n, length(counts)))
end

export ParseFailure, ParseFailureStore, record_parse_failure!,
    parse_failures, parse_failure_summary

end # module
