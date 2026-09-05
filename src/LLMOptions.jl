module LLMOptionsModule

using SymbolicRegression
using PromptingTools
using DispatchDoctor: @unstable
import SymbolicRegression: plugin_mutations, plugin_crossovers
using ..LLMOptionsStructModule:
    LLMOptions,
    LLMOperationWeights,
    LLMMutateMutation,
    LLMRandomizeMutation,
    LLMGenerateMutation,
    LLMCrossover,
    LaSRPlugin,
    LaSRPluginState,
    LaSRContext
using ..LaSRMutationWeightsModule: LaSRMutationWeights
using ..LLMServeModule: LLAMAFILE_MODEL
using ..IdeaStoreModule: AbstractIdeaStore

function set_llm_mutation_weights(
    weights::LaSRMutationWeights, probabilities::LLMOperationWeights
)
    weights = copy(weights)
    randomize_weight = weights.randomize
    ordinary = filter(
        name -> !startswith(string(name), "llm_") && name !== :randomize,
        fieldnames(typeof(weights)),
    )
    ordinary_total = sum(name -> getproperty(weights, name), ordinary)
    weights.randomize *= 1 - probabilities.llm_randomize
    for name in ordinary
        setproperty!(weights, name, (1 - probabilities.llm_mutate) * getproperty(weights, name))
    end
    weights.llm_randomize == 0.0 &&
        (weights.llm_randomize = probabilities.llm_randomize * randomize_weight)
    weights.llm_mutate == 0.0 &&
        (weights.llm_mutate = probabilities.llm_mutate * ordinary_total / length(ordinary))
    return weights
end

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

const _MUTATION_TYPES = (
    mutate_constant=ConstantMutation,
    mutate_operator=OperatorMutation,
    mutate_feature=FeatureMutation,
    swap_operands=SwapOperandsMutation,
    rotate_tree=RotateTreeMutation,
    add_node=AddNodeMutation,
    insert_node=InsertNodeMutation,
    delete_node=DeleteNodeMutation,
    simplify=SimplifyMutation,
    randomize=RandomizeMutation,
    do_nothing=DoNothingMutation,
    optimize=OptimizeMutation,
    form_connection=FormConnectionMutation,
    break_connection=BreakConnectionMutation,
)

function _mutation_pairs(weights::LaSRMutationWeights)
    mutation_pairs = Pair{SymbolicRegression.AbstractMutation,Float64}[
        mutation_type() => getproperty(weights, name) for
        (name, mutation_type) in pairs(_MUTATION_TYPES)
    ]
    push!(mutation_pairs, LLMMutateMutation() => weights.llm_mutate)
    push!(mutation_pairs, LLMRandomizeMutation() => weights.llm_randomize)
    push!(mutation_pairs, LLMGenerateMutation() => weights.llm_generate)
    return mutation_pairs
end

"""
    LaSROptions(; kws...)

Compatibility constructor returning a native `SymbolicRegression.Options`
with a [`LaSRPlugin`](@ref). New code should construct `LLMOptions`,
`LaSRPlugin`, and `Options` explicitly.
"""
@unstable function LaSROptions(;
    use_llm::Bool=false,
    use_concepts::Bool=false,
    use_concept_evolution::Bool=false,
    mutation_weights::Union{LaSRMutationWeights,NamedTuple,Nothing}=nothing,
    llm_operation_weights::Union{LLMOperationWeights,NamedTuple,Nothing}=nothing,
    generate_weight::Real=0.0,
    amnesty_complexity::Integer=0,
    num_pareto_context::Integer=5,
    num_generated_equations::Integer=5,
    num_generated_concepts::Integer=5,
    num_concept_crossover::Integer=2,
    max_concepts::Integer=30,
    is_parametric::Bool=false,
    llm_context::Union{String,Nothing}=nothing,
    variable_names::Union{Dict,Nothing}=nothing,
    prompts_dir::Union{String,Nothing}=nothing,
    idea_database::Union{Vector{<:AbstractString},Nothing}=nothing,
    idea_store::Union{AbstractIdeaStore,Nothing}=nothing,
    api_key::Union{String,Nothing}=nothing,
    model::Union{String,Nothing}=nothing,
    api_kwargs::Union{Dict,Nothing}=nothing,
    http_kwargs::Union{Dict,Nothing}=nothing,
    verbose::Bool=true,
    llm_generate::Function=PromptingTools.aigenerate,
    plugins::Union{Tuple,AbstractVector}=(),
    mutations::Union{Tuple,AbstractVector,Nothing}=nothing,
    default_mutations::Union{Tuple,AbstractVector,Nothing}=nothing,
    kws...,
)
    weights = if isnothing(mutation_weights)
        LaSRMutationWeights()
    elseif mutation_weights isa NamedTuple
        LaSRMutationWeights(; mutation_weights...)
    else
        copy(mutation_weights)
    end
    probabilities = if isnothing(llm_operation_weights)
        LLMOperationWeights()
    elseif llm_operation_weights isa NamedTuple
        LLMOperationWeights(; llm_operation_weights...)
    else
        llm_operation_weights
    end
    weights = use_llm ? set_llm_mutation_weights(weights, probabilities) : weights
    if !use_llm
        weights.llm_mutate = 0.0
        weights.llm_randomize = 0.0
    end
    # `generate_weight` is a direct (non-probability-redistributed) weight, mirroring how the
    # plugin threads `mutate_weight`/`randomize_weight`; keep it opt-in and off when disabled.
    weights.llm_generate = use_llm ? Float64(generate_weight) : 0.0

    prompt_path =
        something(prompts_dir, joinpath(pkgdir(parentmodule(@__MODULE__)), "prompts")) * "/"
    llm_options = LLMOptions(;
        api_key,
        model=something(model, LLAMAFILE_MODEL),
        api_kwargs=something(api_kwargs, Dict("max_tokens" => 1000)),
        http_kwargs=something(http_kwargs, Dict("retries" => 3, "readtimeout" => 3600)),
        llm_generate,
        verbose,
    )
    plugin = LaSRPlugin(;
        llm_options,
        use_llm,
        use_concepts,
        use_concept_evolution,
        num_pareto_context,
        num_generated_equations,
        num_generated_concepts,
        num_concept_crossover,
        max_concepts,
        is_parametric,
        context=something(llm_context, ""),
        variable_names,
        prompts_dir=prompt_path,
        idea_database=AbstractString[something(idea_database, AbstractString[])...],
        idea_store,
        mutate_weight=weights.llm_mutate,
        randomize_weight=weights.llm_randomize,
        crossover_probability=use_llm ? probabilities.llm_crossover : 0.0,
        generate_weight=weights.llm_generate,
        amnesty_complexity,
    )
    resolved_mutations = isnothing(mutations) ? _mutation_pairs(weights) : mutations
    resolved_defaults = isnothing(default_mutations) ? () : default_mutations
    return SymbolicRegression.Options(;
        kws...,
        plugins=(Tuple(plugins)..., plugin),
        mutations=resolved_mutations,
        default_mutations=resolved_defaults,
    )
end

end
