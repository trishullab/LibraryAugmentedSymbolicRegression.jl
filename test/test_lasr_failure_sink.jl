using Test
using LibraryAugmentedSymbolicRegression: LaSRPlugin
using LibraryAugmentedSymbolicRegression.ParseFailuresModule:
    ParseFailureStore, ParseFailure, record_parse_failure!, parse_failures
using LibraryAugmentedSymbolicRegression.PluginModule: LaSRPluginState
using SymbolicRegression: init_plugin_state, fork_plugin_state

# init_plugin_state only reads `dataset.variable_names` when `plugin.variable_names`
# is nothing, and never reads `options`; with variable_names set we can pass nothing
# for both and keep this a pure, server-free unit test.
@testset "parse_failure_sink: use-if-present" begin
    sink = ParseFailureStore()
    plugin = LaSRPlugin(; use_llm=false, variable_names=Dict(1 => "x0"),
                        parse_failure_sink=sink)
    state = init_plugin_state(plugin, nothing, nothing)
    @test state isa LaSRPluginState
    @test state.parse_failures === sink                      # head state uses the sink

    forked = fork_plugin_state(state, plugin, nothing)
    @test forked.parse_failures === sink                     # shared by reference across fork

    record_parse_failure!(sink, ParseFailure("x_0", "x0", :tree_parse, "boom"))
    @test length(parse_failures(state.parse_failures)) == 1  # caller holds a live handle
end

@testset "parse_failure_sink: default is a fresh, unshared store" begin
    plugin = LaSRPlugin(; use_llm=false, variable_names=Dict(1 => "x0"))
    state = init_plugin_state(plugin, nothing, nothing)
    @test isempty(parse_failures(state.parse_failures))
    other = init_plugin_state(plugin, nothing, nothing)
    @test state.parse_failures !== other.parse_failures       # each search gets its own
end
