using TestItems: @testitem
using TestItemRunner: @run_package_tests

ENV["SYMBOLIC_REGRESSION_TEST"] = "true"
# online - test is run on github actions
# offline - test is run locally
tags_to_run = let t = get(ENV, "SYMBOLIC_REGRESSION_TEST_SUITE", "online,offline")
    t = split(t, ",")
    t = map(Symbol, t)
    t
end
@eval @run_package_tests filter = ti -> !isdisjoint(ti.tags, $tags_to_run) verbose = true

@testitem "Test tutorial (MLJ SRRegressor path vs mock server)" tags = [:online] begin
    include("test_tutorial.jl")
end

@testitem "Test expression parser" tags = [:online] begin
    include("test_lasr_parser.jl")
end

@testitem "Test expression parser [hard]" tags = [:online] begin
    include("test_lasr_parser_hard.jl")
end

@testitem "Test expression parser round trips" tags = [:online] begin
    include("test_lasr_parser_roundtrips.jl")
end

@testitem "Test llm output parser" tags = [:online] begin
    include("test_lasr_gen_parsing.jl")
end

@testitem "Test llm prompt construction" tags = [:online] begin
    include("test_lasr_prompt_construction.jl")
end

@testitem "Test prompt path resolution" tags = [:online] begin
    include("test_prompt_paths.jl")
end

@testitem "Test scored idea store" tags = [:online] begin
    include("test_scored_idea_store.jl")
end

@testitem "Test windowed idea store (the default store)" tags = [:online] begin
    include("test_windowed_idea_store.jl")
end

@testitem "Test concept evolution and generation" tags = [:online] begin
    include("test_lasr_concepts.jl")
end

@testitem "Test SR v2 plugin integration" tags = [:online] begin
    include("test_plugin.jl")
end

@testitem "Test extended operators (abs/cbrt/tan, pipe idiom)" tags = [:online] begin
    include("test_lasr_operators_extended.jl")
end

@testitem "Test operator extension example (factorial)" tags = [:online] begin
    include("test_lasr_operator_extension.jl")
end

@testitem "Test LLMGenerateMutation operator" tags = [:online] begin
    include("test_lasr_generate_operator.jl")
end

@testitem "Test complexity amnesty" tags = [:online] begin
    include("test_lasr_amnesty.jl")
end

@testitem "Test llm output is never executed" tags = [:online] begin
    include("test_lasr_parse_msg_safety.jl")
end

@testitem "Test llm round trip against a mock server" tags = [:online] begin
    include("test_lasr_llm_integration.jl")
end

@testitem "Test suggestion cache and call budget" tags = [:online] begin
    include("test_lasr_cache_budget.jl")
end

@testitem "Test normalization pipeline" tags = [:online] begin
    include("test_lasr_normalize.jl")
end

@testitem "Test parse-failure sink" tags = [:online] begin
    include("test_lasr_failure_sink.jl")
end

@testitem "Test parser fallback contract (malformed LLM output)" tags = [:online] begin
    include("test_lasr_parse_fallback.jl")
end

@testitem "Test LLM-output reader (JSON array and object paths)" tags = [:online] begin
    include("test_lasr_client_parse.jl")
end

@testitem "Test LLM operator stress (adversarial output)" tags = [:online] begin
    include("test_lasr_llm_stress.jl")
end

@testitem "Aqua tests" tags = [:online, :aqua] begin
    include("test_aqua.jl")
end

@testitem "JET tests" tags = [:online, :jet] begin
    test_jet_file = joinpath((@__DIR__), "test_jet.jl")
    run(`$(Base.julia_cmd()) --startup-file=no $test_jet_file`)
end
