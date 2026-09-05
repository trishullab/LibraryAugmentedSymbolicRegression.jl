module LLMOptionsModule

using SymbolicRegression
import SymbolicRegression: plugin_mutations, plugin_crossovers
using ..LLMOptionsStructModule:
    LLMMutateMutation,
    LLMRandomizeMutation,
    LLMGenerateMutation,
    LLMCrossover,
    LaSRPlugin,
    LaSRPluginState,
    LaSRContext
using ..IdeaStoreModule: AbstractIdeaStore

function plugin_mutations(plugin::LaSRPlugin)
    plugin.use_llm || return Pair{SymbolicRegression.AbstractMutation,Float64}[]
    pairs = Pair{SymbolicRegression.AbstractMutation,Float64}[]
    plugin.mutate_weight > 0 && push!(pairs, LLMMutateMutation() => plugin.mutate_weight)
    plugin.randomize_weight > 0 &&
        push!(pairs, LLMRandomizeMutation() => plugin.randomize_weight)
    plugin.generate_weight > 0 &&
        push!(pairs, LLMGenerateMutation() => plugin.generate_weight)
    return pairs
end

# Crossover is now an `AbstractCrossover` sampled by weight from `options.crossovers`
# (parallel to `plugin_mutations`), replacing the old `propose_crossover` hook. The plugin
# injects an `LLMCrossover` weighted by `crossover_probability`; the built-in
# `SubtreeCrossover` carries the remaining weight.
function plugin_crossovers(plugin::LaSRPlugin)
    plugin.use_llm || return Pair{SymbolicRegression.AbstractCrossover,Float64}[]
    plugin.crossover_probability > 0 || return Pair{SymbolicRegression.AbstractCrossover,Float64}[]
    return Pair{SymbolicRegression.AbstractCrossover,Float64}[
        LLMCrossover() => plugin.crossover_probability
    ]
end

lasr_context(context::LaSRContext, state=nothing) = context

function lasr_plugin(options::SymbolicRegression.Options)
    matches = filter(plugin -> plugin isa LaSRPlugin, options.plugins)
    length(matches) == 1 ||
        throw(ArgumentError("Expected exactly one LaSRPlugin in `options.plugins`."))
    return only(matches)::LaSRPlugin
end

function lasr_context(options::SymbolicRegression.Options, state=nothing)
    plugin = lasr_plugin(options)
    return LaSRContext(options, plugin, state)
end

function lasr_state(options::SymbolicRegression.Options, plugin_states::Tuple)
    index = findfirst(plugin -> plugin isa LaSRPlugin, options.plugins)
    isnothing(index) && throw(ArgumentError("LaSRPlugin is not active."))
    return plugin_states[index]::LaSRPluginState
end

end
