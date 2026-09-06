# Test: the LLM-output reader (Client.jl).
# What is it supposed to do? parse_msg_content reads model text into a list of strings.
# It accepts a JSON object of strings. safe_literal_parse reads Julia literals without
# evaluation, and it refuses a bare symbol.
# What do we hope to learn from the tests implemented here? The Dict return and the
# Dict-literal / symbol-refusal paths run (they were dark at 89.6%). test_lasr_parse_msg_safety
# shows the reader never RUNS model output; this shows what it accepts.
using Test
using SymbolicRegression: Options
using LibraryAugmentedSymbolicRegression: LaSRPlugin, parse_msg_content
using LibraryAugmentedSymbolicRegression.ClientModule: safe_literal_parse

options = Options(;
    binary_operators=[+, -, *, /], unary_operators=[cos],
    plugins=(LaSRPlugin(; use_llm=false),),
)

@testset "parse_msg_content accepts a JSON object of strings" begin
    out = parse_msg_content("{\"first\": \"x + y\", \"second\": \"cos(x)\"}", options)
    @test sort(out) == ["cos(x)", "x + y"]
end

@testset "safe_literal_parse reads Julia literals but refuses a symbol" begin
    @test safe_literal_parse("[\"a\", \"b\"]") == ["a", "b"]
    @test safe_literal_parse("Dict(\"k\" => \"x * y\")") == Dict("k" => "x * y")
    @test safe_literal_parse("42") == 42
    @test_throws ArgumentError safe_literal_parse("some_identifier")
end

@testset "a Julia-dialect Dict reaches the Dict return in parse_msg_content" begin
    # It is not valid JSON, so it goes through safe_literal_parse to the Dict return.
    @test parse_msg_content("Dict(\"k\" => \"x * y\")", options) == ["x * y"]
end

@testset "a FENCED Julia-dialect literal is not dropped" begin
    # The literal fallback reads the fence-extracted content, not the raw message. A fenced
    # Dict or tuple that is not valid JSON must still yield its expressions.
    @test parse_msg_content("```\nDict(\"k\" => \"x * y\")\n```", options) == ["x * y"]
    @test parse_msg_content("```\n(\"x + y\", \"x - y\")\n```", options) == ["x + y", "x - y"]
end
