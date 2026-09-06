module ClientModule

using DispatchDoctor: @unstable
using SymbolicRegression: AbstractOptions
using ..SuggestionCacheModule: cache_key, take_suggestion!, store_suggestions!
using ..CallBudgetModule: claim_call!
using ..PluginModule: lasr_context
using ..PromptsModule:
    convertDict, get_vars, get_ops, load_prompt, prompt_path, construct_prompt
using ..LaSRLoggerModule: log_generation!
using PromptingTools: CustomOpenAISchema, render, SystemMessage, UserMessage
using JSON: parse
using UUIDs: uuid1

export _clean, ask

const _TRIM = (' ', '\n', '"', ',', '.', '[', ']')
_clean(s) = String(strip(s, _TRIM))

@unstable function _render(conversation, n::Integer; kws...)
    return join(
        [x["content"] for x in render(CustomOpenAISchema(), conversation; N=n, kws...)],
        "\n",
    )
end

"""
    ask(options, mode, n, slots...; prompt=mode, use_cache=true, template_vars...)

Fill the `<prompt>_system.prompt` and `<prompt>_user.prompt` templates, send them, and
return `(candidates, gen_id)`.

Each entry of `slots` is an `elements => tag` pair that `construct_prompt` writes into the
user template, in the order given. `prompt` names the template pair when it differs from
`mode`, which is the label used for logging and for the suggestion pool.
"""
function ask(
    options::AbstractOptions,
    mode::AbstractString,
    n::Integer,
    slots::Pair...;
    prompt::AbstractString=mode,
    use_cache::Bool=true,
    template_vars...,
)
    dir = options.prompts_dir
    user = load_prompt(prompt_path(dir, prompt * "_user.prompt"))
    for (elements, tag) in slots
        user = construct_prompt(user, elements, tag)
    end
    conversation = [
        SystemMessage(load_prompt(prompt_path(dir, prompt * "_system.prompt"))),
        UserMessage(user),
    ]
    return request_suggestions(
        options, mode, conversation, n; use_cache=use_cache, template_vars...
    )
end

"""
    request_suggestions(options, mode, conversation, n; use_cache=true, template_vars...)

Send one conversation to the model and return `(candidates, gen_id)`. Rendering, caching,
budgeting, logging, and response parsing all happen here.

Pass `use_cache=false` to keep a mode out of the suggestion pool. The pool hands back one
suggestion at a time, so a caller that needs the whole batch must opt out.
"""
function request_suggestions(
    options::AbstractOptions,
    mode::AbstractString,
    conversation,
    n::Integer;
    use_cache::Bool=true,
    template_vars...,
)
    gen_id = uuid1()
    vars = (;
        variables=get_vars(options),
        operators=get_ops(options),
        no_system_message=false,
        template_vars...,
    )
    rendered_msg = _render(conversation, n; vars...)
    log_generation!(options.lasr_logger; id=gen_id, mode=mode, llm_input=rendered_msg)

    cache = options.suggestion_cache
    key = (use_cache && !isnothing(cache)) ? cache_key(mode, rendered_msg) : nothing

    if !isnothing(key)
        pooled = take_suggestion!(cache, key)
        if !isnothing(pooled)
            log_generation!(options.lasr_logger; id=gen_id, mode=mode, cached=pooled)
            return String[pooled], gen_id
        end
    end

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
            verbose=options.verbose,
            vars...,
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
    candidates = parse_msg_content(string(msg.content), options)
    isempty(candidates) &&
        log_generation!(options.lasr_logger; id=gen_id, mode=mode, failed="None")

    if !isnothing(key) && length(candidates) > 1
        store_suggestions!(cache, key, candidates[2:end])
    end
    return candidates, gen_id
end

"""
    parse_msg_content(msg_content, options) -> Vector{String}

Read the expression strings out of one model response.

The reader looks for a fenced or bracketed JSON payload and falls back to the whole
message. It is JSON-only, apart from a retry that drops a trailing comma. It never
evaluates the response: model output is untrusted input, and `eval` on it would be remote
code execution.
"""
function parse_msg_content(msg_content::String, options::AbstractOptions)::Vector{String}
    options = lasr_context(options)
    content = msg_content
    for pat in (r"```json(.*?)```"s, r"```(.*?)```"s, r"(\[.*?\])"s)
        m = match(pat, msg_content)
        m === nothing && continue
        cap = m.captures[1]
        cap === nothing && continue
        content = String(cap)
        break
    end

    out = try
        parse(content)
    catch
        # A trailing comma is the one non-JSON slip a model makes often enough to be worth
        # handling (`["x + y",]`). Retrying only on already-failed content means a
        # well-formed payload never reaches this rewrite.
        try
            parse(replace(content, r",(\s*[\]\}])" => s"\1"))
        catch
            options.verbose && @debug "Failed to parse content: $content"
            nothing
        end
    end

    if out isa Dict
        return String[v for v in values(out) if v isa String]
    elseif out isa Vector
        return String[x for x in out if x isa String]
    end
    return String[]
end

end # module
