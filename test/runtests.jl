using TestItems: @testitem
using TestItemRunner: @run_package_tests

ENV["SYMBOLIC_REGRESSION_TEST"] = "true"
tags_to_run = let t = get(ENV, "SYMBOLIC_REGRESSION_TEST_SUITE", "llm")
    t = split(t, ",")
    t = map(Symbol, t)
    t
end

@testitem "LLM Integration tests" tags = [:llm] begin
    include("test_lasr_integration.jl")
end
