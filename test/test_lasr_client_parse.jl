# Test: the LLM-output reader (Client.jl).
# What is it supposed to do? parse_msg_content reads model text into a list of strings.
# It accepts a JSON array or a JSON object of strings, fenced or bare.
# What do we hope to learn from the tests implemented here? The Dict return path runs (it
# was dark at 89.6%). test_lasr_parse_msg_safety shows the reader never RUNS model output;
# this shows what it accepts.
using Test
using SymbolicRegression: Options
using LibraryAugmentedSymbolicRegression: LaSRPlugin, parse_msg_content

options = Options(;
    binary_operators=[+, -, *, /],
    unary_operators=[cos],
    plugins=(LaSRPlugin(; use_llm=false),),
)

@testset "parse_msg_content accepts a JSON object of strings" begin
    out = parse_msg_content("{\"first\": \"x + y\", \"second\": \"cos(x)\"}", options)
    @test sort(out) == ["cos(x)", "x + y"]
end

@testset "a fenced JSON payload is read, not the surrounding prose" begin
    @test parse_msg_content("thinking...\n```json\n[\"x * y\"]\n```\ndone", options) ==
        ["x * y"]
    @test parse_msg_content("```\n{\"a\": \"x - y\"}\n```", options) == ["x - y"]
end

@testset "a trailing comma is recovered" begin
    # Models emit `["x + y",]` often enough to be worth a retry; strict JSON rejects it.
    @test parse_msg_content("[\"x + y\", \"cos(x)\",]", options) == ["x + y", "cos(x)"]
    @test parse_msg_content("```json\n[\"x * y\",]\n```", options) == ["x * y"]
    # The retry only runs on content that already failed, so a valid payload is untouched.
    @test parse_msg_content("[\"f(a, b)\", \"g(x,y)\"]", options) == ["f(a, b)", "g(x,y)"]
end

@testset "a non-JSON dialect yields no expressions" begin
    # The reader is JSON-only by design: it never evaluates model output, so a Julia-style
    # `Dict(...)` or tuple is simply not read.
    @test parse_msg_content("Dict(\"k\" => \"x * y\")", options) == String[]
end
