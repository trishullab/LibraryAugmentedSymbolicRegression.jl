module LLMOptionsModule

using DispatchDoctor: @unstable
using SymbolicRegression
using ..LLMOptionsStructModule:
    LaSROptions, LLM_OPTIONS_KEYS, LaSRMutationWeights, LLMOperationWeights, LLMOptions
using ..UtilsModule: @save_kwargs, @ignore

create_lasr_mutation_weights(w::LaSRMutationWeights) = w
create_lasr_mutation_weights(w::NamedTuple) = LaSRMutationWeights(; w...)

create_llm_operation_weights(w::LLMOperationWeights) = w
create_llm_operation_weights(w::NamedTuple) = LLMOperationWeights(; w...)

@ignore const LASR_DEFAULT_OPTIONS = ()

const LASR_OPTIONS_DESCRIPTION = """
- `use_llm::Bool`: Whether to use LLM inference. (default: false)
- `use_concepts::Bool`: Whether to summarize candidate programs into natural-language concepts and use those concepts to guide the search (i.e., a specialization of FunSearch). (default: false)
    NOTE: If `use_llm` is false, then `use_concepts` is ignored.
- `use_concept_evolution::Bool`: Whether to evolve the concepts after every iteration. (default: false)
    NOTE: If `use_concepts` is false, then `use_concept_evolution` is ignored.
- `lasr_mutation_weights::LaSRMutationWeights`: Unnormalized mutation weights for the mutation operators (e.g., `llm_mutate`, `llm_randomize`).
- `llm_operation_weights::LLMOperationWeights`: Normalized probabilities of using LLM-based crossover vs. symbolic crossover.
- `num_pareto_context::Int64`: Number of equations to sample from the Pareto frontier for summarization.
- `num_generated_equations::Int64`: Number of new equations to generate from the LLM per iteration.
- `num_generated_concepts::Int64`: Number of new concepts to generate from the LLM per iteration.
- `max_concepts::UInt32`: Maximum number of concepts to retain in the concept library. (default: 30)
- `is_parametric::Bool`: A special flag to allow sampling parametric equations from LaSR. (default: false)
- `llm_context::AbstractString`: A natural-language hint or context string passed to the LLM.
- `llm_recorder_dir::AbstractString`: Directory to log LLM interactions (creates `llm_calls.txt`). (default: "lasr_runs/")
- `variable_names::Union{Dict,Nothing}`: A mapping of symbolic variable names to domain-meaningful names. (default: nothing)
- `prompts_dir::AbstractString`: The location of zero-shot prompts for the LLM. Specialize these prompts to your domain for better performance. (default: "prompts/")
- `idea_database::Vector{AbstractString}`: A list of natural-language concept “ideas” for seeding the LLM. (default: [])
- `api_key::AbstractString`: An API key for an OpenAI-compatible server. Required.
- `model::AbstractString`: The LLM model name for the OpenAI-compatible server. Required.
- `api_kwargs::Dict`: Additional keyword arguments for the LLM's API. Must include `url::AbstractString` to specify the endpoint. (default: `Dict("url" => "...")`)
    - For example: `Dict("url" => "http://localhost:11440/v1", "max_tokens" => 1000)`
- `http_kwargs::Dict`: Additional keyword arguments for HTTP requests.
    - `retries::Int`: Number of retries. (default: 3)
    - `readtimeout::Int`: Read timeout in seconds. (default: 3600)
- `verbose::Bool`: Whether to print tokens and additional debugging info for each LLM call. (default: true)
"""

# Constructor with both sets of parameters:
"""
    LaSROptions(;kws...) <: SymbolicRegression.AbstractOptions

Construct options for `equation_search` and other functions. Also read: SymbolicRegression.Options

# Arguments
$(LASR_OPTIONS_DESCRIPTION)
"""
@unstable @save_kwargs LASR_DEFAULT_OPTIONS function LaSROptions(;
    # LLM Search Options.
    ## 1. LaSR Ablation Modifiers
    use_llm::Bool=false,
    use_concepts::Bool=false,
    use_concept_evolution::Bool=false,
    @nospecialize(
        lasr_mutation_weights::Union{
            LaSRMutationWeights,AbstractVector,NamedTuple,Nothing
        } = nothing
    ),
    @nospecialize(
        llm_operation_weights::Union{
            LLMOperationWeights,AbstractVector,NamedTuple,Nothing
        } = nothing
    ),

    ## 2. LaSR Performance Modifiers
    num_pareto_context::Integer=5,
    num_generated_equations::Integer=5,
    num_generated_concepts::Integer=5,
    max_concepts::Integer=30,
    is_parametric::Bool=false,
    llm_context::Union{String,Nothing}=nothing,

    ## 3. LaSR Bookkeeping Utilities
    llm_recorder_dir::Union{String,Nothing}=nothing,
    variable_names::Union{Dict,Nothing}=nothing,
    prompts_dir::Union{String,Nothing}=nothing,
    idea_database::Union{Vector{AbstractString},Nothing}=nothing,

    ## 4. LaSR LLM API Options
    api_key::Union{String,Nothing}=nothing,
    model::Union{String,Nothing}=nothing,
    api_kwargs::Union{Dict,Nothing}=nothing,
    http_kwargs::Union{Dict,Nothing}=nothing,
    verbose::Bool=true,
    kws...,
)
    if use_llm
        if isnothing(model)
            @warn "model is not set in LaSROptions. This might be required for your LLM Inference backend."
        end

        if isnothing(api_key)
            @warn "api_key is not set in LaSROptions. This might be required for your LLM Inference backend."
        end

        if isnothing(api_kwargs) ||
            !haskey(api_kwargs, "max_tokens") ||
            isnothing(api_kwargs["max_tokens"])
            @warn "api_kwargs.max_tokens is not set in LaSROptions. Defaulting to 1000."
        end

        if isnothing(api_kwargs) ||
            !haskey(api_kwargs, "url") ||
            isnothing(api_kwargs["url"])
            error(
                "api_kwargs.url is not set in LaSROptions. No backend URL to send the request to.",
            )
        end
    end

    #################################
    #### Supply defaults ############
    #! format: off
    _default_options = default_options()
    use_llm = something(use_llm, _default_options.use_llm)
    use_concepts = something(use_concepts, _default_options.use_concepts)
    use_concept_evolution = something(use_concept_evolution, _default_options.use_concept_evolution)
    lasr_mutation_weights = something(lasr_mutation_weights, _default_options.lasr_mutation_weights)
    llm_operation_weights = something(llm_operation_weights, _default_options.llm_operation_weights)
    num_pareto_context = something(num_pareto_context, _default_options.num_pareto_context)
    num_generated_equations = something(num_generated_equations, _default_options.num_generated_equations)
    num_generated_concepts = something(num_generated_concepts, _default_options.num_generated_concepts)
    max_concepts = something(max_concepts, _default_options.max_concepts)
    is_parametric = something(is_parametric, _default_options.is_parametric)
    llm_context = something(llm_context, _default_options.llm_context)
    llm_recorder_dir = something(llm_recorder_dir, _default_options.llm_recorder_dir)
    variable_names = something(variable_names, _default_options.variable_names)
    prompts_dir = something(prompts_dir, _default_options.prompts_dir)
    idea_database = something(idea_database, _default_options.idea_database)
    api_key = something(api_key, _default_options.api_key)
    model = something(model, _default_options.model)
    api_kwargs = something(api_kwargs, _default_options.api_kwargs)
    http_kwargs = something(http_kwargs, _default_options.http_kwargs)
    verbose = something(verbose, _default_options.verbose)
    #! format: on
    #################################

    if !isdir(prompts_dir)
        @warn "Prompts directory does not exist. Creating one at $prompts_dir."
        mkdir(prompts_dir)
    end

    if !isdir(llm_recorder_dir)
        @warn "LLM Recorder directory does not exist. Creating one at $llm_recorder_dir."
        mkdir(llm_recorder_dir)
    end

    set_lasr_mutation_weights = create_lasr_mutation_weights(lasr_mutation_weights)
    set_llm_operation_weights = create_llm_operation_weights(llm_operation_weights)

    llm_options = LLMOptions(
        use_llm,
        use_concepts,
        use_concept_evolution,
        set_lasr_mutation_weights,
        set_llm_operation_weights,
        num_pareto_context,
        num_generated_equations,
        num_generated_concepts,
        max_concepts,
        is_parametric,
        llm_context,
        llm_recorder_dir,
        variable_names,
        prompts_dir,
        idea_database,
        api_key,
        model,
        api_kwargs,
        http_kwargs,
        verbose,
    )
    sr_options_keys = filter(k -> !(k in LLM_OPTIONS_KEYS), keys(kws))
    sr_options = SymbolicRegression.Options(;
        NamedTuple(sr_options_keys .=> Tuple(kws[k] for k in sr_options_keys))...
    )
    return LaSROptions(llm_options, sr_options)
end

# Make all `Options` available while also making `llm_options` accessible
function Base.getproperty(options::LaSROptions, k::Symbol)
    if k in LLM_OPTIONS_KEYS
        return getproperty(getfield(options, :llm_options), k)
    else
        return getproperty(getfield(options, :sr_options), k)
    end
end

# Add setproperty! for `Options` and `llm_options`
function Base.setproperty!(options::LaSROptions, k::Symbol, v)
    if k in LLM_OPTIONS_KEYS
        return setproperty!(getfield(options, :llm_options), k, v)
    else
        return setproperty!(getfield(options, :sr_options), k, v)
    end
end

function Base.propertynames(options::LaSROptions)
    return (LLM_OPTIONS_KEYS..., fieldnames(SymbolicRegression.Options)...)
end

function default_options()
    return (;
        # LaSR Ablation Modifiers
        use_llm=false,
        use_concepts=false,
        use_concept_evolution=false,
        lasr_mutation_weights=LaSRMutationWeights(; llm_mutate=0.0, llm_randomize=0.0),
        llm_operation_weights=LLMOperationWeights(; llm_crossover=0.0),

        # LaSR Performance Modifiers
        num_pareto_context=5,
        num_generated_equations=5,
        num_generated_concepts=5,
        max_concepts=30,
        is_parametric=false,
        llm_context="",

        # LaSR Bookkeeping Utilities
        llm_recorder_dir="lasr_runs/",
        variable_names=Dict(),
        prompts_dir="prompts/",
        idea_database=Vector{AbstractString}(),

        # LaSR LLM API Options
        api_key="",
        model="",
        api_kwargs=Dict("max_tokens" => 1000),
        http_kwargs=Dict("retries" => 3, "readtimeout" => 3600),
        verbose=true,
    )
end

end # module
