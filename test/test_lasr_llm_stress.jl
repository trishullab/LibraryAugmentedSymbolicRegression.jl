# Test: stress the LLM operators and reader with adversarial model output.
# What is it supposed to do? The operators read untrusted model output. They must not
# throw. They must not insert a constant-1 tree in place of a usable proposal or a parent.
# What do we hope to learn from the tests implemented here? Each test targets a real
# defect, not a style point: the selection loop can miss the one usable proposal; crossover
# can insert a constant child; the reader can drop valid strings next to a non-string.
using Test
using LibraryAugmentedSymbolicRegression
using LibraryAugmentedSymbolicRegression: llm_mutate_tree, llm_crossover_trees, parse_expr, parse_msg_content
using LibraryAugmentedSymbolicRegression.PluginModule: LaSRPlugin, lasr_context
using LibraryAugmentedSymbolicRegression.ClientModule: request_suggestions
using SymbolicRegression: Options, string_tree
using DynamicExpressions: get_contents, filter_map
using PromptingTools: AIMessage, SystemMessage, UserMessage

include("test_helpers.jl")  # provides mock_llm(calls, content)

_refs_feature(ex) =
    !isempty(filter_map(n -> n.degree == 0 && !n.constant, n -> 1, get_contents(ex), Int))

@testset "candidate selection never misses the one usable proposal" begin
    # The batch has four constant proposals and one usable one, last. The old loop drew an
    # index with replacement, so it missed the usable one about a third of the time and
    # returned a constant. Over 100 trials a miss is near-certain.
    calls = Ref(0)
    plugin = LaSRPlugin(;
        llm_generate=mock_llm(calls, "[\"1.0\", \"1.0\", \"1.0\", \"1.0\", \"x0 + x1\"]"),
        num_generated_equations=5,
        variable_names=Dict(1 => "x0", 2 => "x1"),
        use_concepts=false,
    )
    options = Options(; binary_operators=[+, *], plugins=(plugin,))
    parent = parse_expr(Float64, "x0 * x1", options)
    n_usable = count(_ -> _refs_feature(llm_mutate_tree(copy(parent), options)), 1:100)
    @test n_usable == 100
end

@testset "crossover with only a constant proposal returns the parents" begin
    # A lone constant proposal must not become a child. Both children stay the parents.
    calls = Ref(0)
    plugin = LaSRPlugin(;
        llm_generate=mock_llm(calls, "[\"1.0\"]"),
        num_generated_equations=3,
        variable_names=Dict(1 => "x0", 2 => "x1"),
        use_concepts=false,
    )
    options = Options(; binary_operators=[+, *], plugins=(plugin,))
    p1 = parse_expr(Float64, "x0 + x1", options)
    p2 = parse_expr(Float64, "x0 * x1", options)
    c1, c2 = llm_crossover_trees(copy(p1), copy(p2), options)
    @test string_tree(c1, options) == string_tree(p1, options)
    @test string_tree(c2, options) == string_tree(p2, options)
end

@testset "parse_msg_content keeps valid strings next to a non-string" begin
    # `["x + y", 3.0]` must yield the valid string, not an empty list.
    options = Options(; binary_operators=[+], plugins=(LaSRPlugin(; use_llm=false),))
    @test parse_msg_content("[\"x + y\", 3.0]", options) == ["x + y"]
end

@testset "parse_msg_content handles a very large list" begin
    options = Options(; binary_operators=[+], plugins=(LaSRPlugin(; use_llm=false),))
    huge = "[" * join(("\"x$i\"" for i in 1:10_000), ", ") * "]"
    @test length(parse_msg_content(huge, options)) == 10_000
end

@testset "a null-content response returns empty, not a crash" begin
    # A content-filtered or empty completion has message content === nothing. The seam must
    # return an empty list, not throw out of the operator.
    null_gen(_schema, _conversation; kwargs...) = AIMessage(; content=nothing)
    options = Options(;
        binary_operators=[+, -, *],
        plugins=(
            LaSRPlugin(;
                use_llm=true, mutate_weight=1.0, llm_generate=null_gen,
                api_key="mock", model="mock-model", variable_names=Dict(1 => "x"),
            ),
        ),
    )
    ctx = lasr_context(options)
    conv = [SystemMessage("sys"), UserMessage("user")]
    cands, _ = request_suggestions(ctx, "mutate", conv, 4)
    @test isempty(cands)
end
