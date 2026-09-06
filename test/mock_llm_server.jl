# A minimal OpenAI-compatible chat-completions server used by the LLM integration
# tests. It lets CI exercise the full LLM code path (prompt construction -> HTTP ->
# response parsing -> expression insertion) deterministically, without downloading a
# model or reaching the network.

module MockLLMServer

using HTTP: HTTP
using JSON3: JSON3
using Sockets: Sockets

const EQUATIONS = [
    "2 * cos(x) + y * y - 2",
    "cos(x) * C + y ^ 2",
    "C * cos(x) - y",
    "x + y * C",
    "cos(x + y) * C",
]

const CONCEPTS = [
    "The expression likely involves a cosine of the first variable.",
    "A quadratic term in the second variable improves the fit.",
    "Additive structure separates the two variables.",
]

"""Counts of how many times each LaSR operation asked the server for a completion."""
const CALL_COUNTS = Dict{String,Int}()
const COUNTS_LOCK = ReentrantLock()

count_for(mode) = lock(COUNTS_LOCK) do
    return get(CALL_COUNTS, mode, 0)
end
total_calls() = lock(COUNTS_LOCK) do
    return sum(values(CALL_COUNTS); init=0)
end
reset_counts!() = lock(COUNTS_LOCK) do
    return empty!(CALL_COUNTS)
end

"""Identify which LaSR operation a prompt came from, via its template's markers."""
function classify(prompt::AbstractString)
    p = lowercase(prompt)
    # Concept prompts ask for "hypotheses"; equation prompts ask for "expressions".
    if occursin("hypotheses", p)
        return occursin("good expression", p) ? "generate_concepts" : "concept_evolution"
    end
    (occursin("reference expression 1", p) && occursin("reference expression 2", p)) &&
        return "crossover"
    occursin("reference expression:", p) && return "mutate"
    return "gen_random"
end

function payload_for(mode, n)
    items = if mode in ("generate_concepts", "concept_evolution")
        CONCEPTS[1:min(n, length(CONCEPTS))]
    else
        [EQUATIONS[mod1(i, length(EQUATIONS))] for i in 1:max(n, 1)]
    end
    return "```json\n" * JSON3.write(items) * "\n```"
end

function handle(req::HTTP.Request)
    if req.method == "GET"
        return HTTP.Response(
            200, ["Content-Type" => "application/json"]; body=JSON3.write((; object="list"))
        )
    end

    body = JSON3.read(String(req.body))
    prompt = join([get(m, :content, "") for m in get(body, :messages, [])], "\n")
    mode = classify(prompt)
    lock(COUNTS_LOCK) do
        return CALL_COUNTS[mode] = get(CALL_COUNTS, mode, 0) + 1
    end

    text = payload_for(mode, 5)
    resp = (;
        id="chatcmpl-mock",
        object="chat.completion",
        created=0,
        model=get(body, :model, "mock-model"),
        choices=[(;
            index=0, message=(; role="assistant", content=text), finish_reason="stop"
        )],
        usage=(; prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    return HTTP.Response(
        200, ["Content-Type" => "application/json"]; body=JSON3.write(resp)
    )
end

"""
    with_server(f, port)

Run `f(url)` with a mock OpenAI-compatible server listening on `port`, shutting it
down fully afterwards.

The shutdown must be complete, not merely requested. `TestItemRunner` executes every
test item in one process, so a surviving accept loop outlives this test file. Under a
single-threaded `julia`, that leaked task starves the cooperative scheduler and later
searches spin at 100% CPU while evaluating nothing -- the suite appears to hang. So we
`close` *and* `wait` for the server, then confirm the port is actually free.
"""
function with_server(f::Function, port::Int)
    reset_counts!()
    server = HTTP.serve!(handle, "127.0.0.1", port; verbose=false)
    try
        return f("http://127.0.0.1:$(port)/v1")
    finally
        HTTP.forceclose(server)
        try
            wait(server)
        catch
            # `wait` throws if the server task was already torn down; that is the
            # outcome we want, so it is not an error.
        end
        _await_port_release(port)
    end
end

"""Block briefly until nothing is listening on `port`, so the next test can bind it."""
function _await_port_release(port::Int; timeout_s::Real=10.0)
    deadline = time() + timeout_s
    while time() < deadline
        sock = try
            s = Sockets.connect("127.0.0.1", port)
            close(s)
            s
        catch
            return true  # refused => nothing is listening
        end
        sleep(0.05)
    end
    @warn "mock LLM server port $port still accepting connections after shutdown"
    return false
end

end # module
