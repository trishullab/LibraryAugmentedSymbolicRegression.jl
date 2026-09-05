module ClientModule

using DispatchDoctor: @unstable
using SymbolicRegression: AbstractOptions
using ..SuggestionCacheModule: cache_key, take_suggestion!, store_suggestions!
using ..CallBudgetModule: claim_call!
using ..PluginModule: lasr_context
using ..PromptsModule: convertDict, get_vars, get_ops
using ..LaSRLoggerModule: log_generation!
using PromptingTools: CustomOpenAISchema, render
using JSON: parse
using UUIDs: uuid1

export _clean, render_conversation

const _TRIM = (' ', '\n', '"', ',', '.', '[', ']')
_clean(s) = String(strip(s, _TRIM))

# `@unstable`: `join` over PromptingTools' loosely-typed rendered `content` infers to
# `Union{Nothing,AnnotatedString{String},String}`. This is a cold prompt-marshalling helper
# (the result is only fed to logging/cache-key), so exempt it from DispatchDoctor's
# `error` mode the same way `safe_literal_parse`/`take_suggestion!` are. Inline in the
# operators this was invisible; extracting it created a checked function boundary.
@unstable function render_conversation(conversation, options; N::Integer, kws...)
    return join(
        [
            x["content"] for x in render(
                CustomOpenAISchema(),
                conversation;
                variables=get_vars(options),
                operators=get_ops(options),
                N=N,
                no_system_message=false,
                kws...,
            )
        ],
        "\n",
    )
end

"""
    request_suggestions(options, mode, conversation, n; rendered_msg, template_vars...)

Send one rendered conversation to the model and return `(candidates, gen_id)`.

Every LLM operation funnels through here, so logging, error handling, and response
parsing exist once rather than being duplicated per operator. On any failure --- a
transport error, or a response with nothing parseable in it --- this returns an empty
candidate list and the caller applies its own fallback.
"""
function request_suggestions(
    options::AbstractOptions,
    mode::AbstractString,
    conversation,
    n::Integer;
    rendered_msg::Union{AbstractString,Nothing}=nothing,
    template_vars...,
)
    gen_id = uuid1()
    if isnothing(rendered_msg)
        log_generation!(options.lasr_logger; id=gen_id, mode=mode)
    else
        log_generation!(options.lasr_logger; id=gen_id, mode=mode, llm_input=rendered_msg)
    end

    # The rendered prompt is exactly what determines the answer, so it is the cache key.
    # Requests that never render one (no `rendered_msg`) simply do not participate.
    cache = options.suggestion_cache
    key = if isnothing(cache) || isnothing(rendered_msg)
        nothing
    else
        cache_key(mode, rendered_msg)
    end

    if !isnothing(key)
        pooled = take_suggestion!(cache, key)
        if !isnothing(pooled)
            log_generation!(options.lasr_logger; id=gen_id, mode=mode, cached=pooled)
            return String[pooled], gen_id
        end
    end

    # Nothing pooled, so this would be a real round trip: check the allowance first.
    # Refusing here rather than at each operator means every LLM entry point is bounded
    # by construction.
    if !claim_call!(options.call_budget)
        log_generation!(options.lasr_logger; id=gen_id, mode=mode, failed="budget")
        return String[], gen_id
    end

    msg = try
        options.llm_generate(
            CustomOpenAISchema(),
            conversation;
            N=n,
            api_key=options.api_key,
            model=options.model,
            api_kwargs=convertDict(options.api_kwargs),
            http_kwargs=convertDict(options.http_kwargs),
            template_vars...,
        )
    catch e
        log_generation!(
            options.lasr_logger; id=gen_id, mode=mode, failed="None." * string(e)
        )
        return String[], gen_id
    end

    log_generation!(
        options.lasr_logger; id=gen_id, mode=mode, llm_output=string(msg.content)
    )
    candidates = parse_msg_content(String(msg.content), options)
    # A well-formed response with nothing parseable in it is still a failure for the
    # caller; record it the same way a transport error is recorded.
    isempty(candidates) &&
        log_generation!(options.lasr_logger; id=gen_id, mode=mode, failed="None")
    # Bank everything past the first; the caller consumes from the front.
    if !isnothing(key) && length(candidates) > 1
        store_suggestions!(cache, key, candidates[2:end])
    end
    return candidates, gen_id
end

"""
    safe_literal_parse(s::AbstractString)

Read `s` as a Julia *literal* vector/dict of strings and numbers, without evaluating it.

Only array/tuple/dict literals and their scalar elements are accepted; anything that
would require running code (function calls, `:call` nodes, interpolation, ...) is
rejected. This is the safe counterpart to `eval(Meta.parse(s))`, which would execute
arbitrary code contained in an LLM response.
"""
@unstable function safe_literal_parse(s::AbstractString)
    return _literal_value(Meta.parse(strip(s)))
end

@unstable function _literal_value(x)
    # Bare literals parsed by `Meta.parse` come back as plain values.
    (x isa String || x isa Number || x isa Bool) && return x
    x isa QuoteNode && return _literal_value(x.value)
    x isa Symbol && throw(ArgumentError("refusing to resolve symbol `$(x)`"))
    x isa Expr || throw(ArgumentError("unsupported literal of type $(typeof(x))"))

    if x.head === :vect || x.head === :tuple || x.head === :hcat || x.head === :vcat
        return Any[_literal_value(a) for a in x.args]
    elseif x.head === :call && !isempty(x.args) && x.args[1] === :Dict
        d = Dict{Any,Any}()
        for a in x.args[2:end]
            a isa Expr && a.head === :call && a.args[1] === :(=>) ||
                throw(ArgumentError("unsupported Dict entry"))
            d[_literal_value(a.args[2])] = _literal_value(a.args[3])
        end
        return d
    end
    throw(ArgumentError("refusing to evaluate expression head `$(x.head)`"))
end

@unstable function try_capture(pattern::Regex, text::String)::Union{Nothing,AbstractString}
    m = match(pattern, text)
    return m === nothing ? nothing : get(m.captures, 1, nothing)
end

function parse_msg_content(msg_content::String, options::AbstractOptions)::Vector{String}
    options = lasr_context(options)
    # Attempt extraction with several patterns in order
    patterns = [r"```json(.*?)```"s, r"```(.*?)```"s, r"(\[.*?\])"s]

    content = nothing
    for pat in patterns
        content = try_capture(pat, msg_content)
        content !== nothing && break
    end

    content = content === nothing ? msg_content : content

    out = nothing
    try
        out = parse(content)
    catch
        if options.verbose
            @debug "Failed to parse content: $content"
        end
    end

    # Fall back to a *literal-only* reader for Julia-style vectors (e.g. `["x + y"]`)
    # that are not valid JSON. This deliberately never evaluates the model output:
    # LLM responses are untrusted input, and `eval` on them is remote code execution.
    if isnothing(out)
        try
            out = safe_literal_parse(msg_content)
        catch
            if options.verbose
                @debug "Failed to read content as a literal: $content"
            end
        end
    end

    if out isa Dict && all(x -> isa(x, String), values(out))
        return collect(values(out))
    elseif out isa Vector && all(x -> isa(x, String), out)
        return out
    end
    return String[]
end

end # module
