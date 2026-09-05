# `parse_msg_content` reads text produced by an LLM, which is untrusted input: it can
# be steered by a prompt-injected dataset, a hostile `llm_context`, or a compromised
# inference endpoint. It previously ran `eval(Meta.parse(msg_content))` on that text,
# which executed arbitrary Julia code. These tests pin the safe behaviour.

using Test
using LibraryAugmentedSymbolicRegression: LaSROptions, parse_msg_content

options = LaSROptions(; binary_operators=[+, -, *, /], unary_operators=[cos])

@testset "well-formed responses still parse" begin
    @test parse_msg_content("```json\n[\"x + y\", \"cos(x)\"]\n```", options) ==
        ["x + y", "cos(x)"]
    @test parse_msg_content("here you go:\n[\"x * y\"]\n", options) == ["x * y"]
end

@testset "model output is never executed" begin
    marker = tempname()
    @test !isfile(marker)

    # A response that would write a file if it were evaluated.
    payload = "begin; write(\"$marker\", \"pwned\"); [\"x+y\"]; end"
    parse_msg_content(payload, options)
    @test !isfile(marker)

    # A bare call expression must also not run.
    marker2 = tempname()
    parse_msg_content("[write(\"$marker2\", \"pwned\")]", options)
    @test !isfile(marker2)
end

@testset "unparseable responses yield no expressions" begin
    @test parse_msg_content("I'm sorry, I can't help with that.", options) == String[]
    @test parse_msg_content("", options) == String[]
end
