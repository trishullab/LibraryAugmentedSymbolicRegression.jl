using TestItems: @testitem
using TestItemRunner: @run_package_tests

ENV["SYMBOLIC_REGRESSION_TEST"] = "true"
# online - test is run on github actions
# online_llamafile - test is run on github actions and downloads the llamafile
# offline - test is run locally
tags_to_run =
    let t = get(ENV, "SYMBOLIC_REGRESSION_TEST_SUITE", "online,online_llamafile,offline")
        t = split(t, ",")
        t = map(Symbol, t)
        t
    end
@eval @run_package_tests filter = ti -> !isdisjoint(ti.tags, $tags_to_run) verbose = true

# @testitem "Test handshake" tags = [:online_llamafile] begin
#     include("test_handshake.jl")
# end

# # This test takes too long. Best to perform it offline.
# @testitem "Test tutorial" tags = [:offline] begin
#     include("test_tutorial_llamafile.jl")
# end

# @testitem "Test tutorial" tags = [:offline] begin
#     include("test_tutorial.jl")
# end

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

# Test SymbolicRegression.jl backwards compatibility (~15 min)
include("test_backwards_compat.jl")

@testitem "Test whether the precompilation script works." tags = [:online] begin
    include("test_precompilation.jl")
end

# @testitem "Aqua tests" tags = [:online, :aqua] begin
#     include("test_aqua.jl")
# end

# @testitem "JET tests" tags = [:online, :jet] begin
#     test_jet_file = joinpath((@__DIR__), "test_jet.jl")
#     run(`$(Base.julia_cmd()) --startup-file=no $test_jet_file`)
# end
