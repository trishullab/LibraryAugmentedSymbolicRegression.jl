# Shared helpers for the mock-only LaSR operator tests.
#
# `mock_llm` returns a drop-in replacement for `llm_generate` that ignores its arguments,
# counts how many times it was invoked, and always returns a fixed `content` string. It
# mirrors the mock used in `test_plugin.jl` (positional/keyword-agnostic signature) so no
# model server is ever contacted.
function mock_llm(calls, content)
    return function (args...; kwargs...)
        calls[] += 1
        return (; content=content)
    end
end
