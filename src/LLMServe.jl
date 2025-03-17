module LLMServeModule
using Downloads
using Logging
using Base: dirname, isdir, mkdir, atexit

const LLAMAFILE_MODEL = get(ENV, "LLAMAFILE_MODEL", "gemma-2-2b-it.Q6_K")
const LLAMAFILE_PATH = get(
    ENV, "LLAMAFILE_PATH", abspath("$(@__DIR__)/../llamafiles/gemma-2-2b-it.Q6_K.llamafile")
)
const LLAMAFILE_URL = get(
    ENV,
    "LLAMAFILE_URL",
    "https://huggingface.co/Mozilla/gemma-2-2b-it-llamafile/resolve/main/gemma-2-2b-it.Q6_K.llamafile",
)
const LLM_PORT = parse(Int, get(ENV, "LLM_PORT", "11449"))

# print the project dir
@info "Project directory: $(@__DIR__)"

"""
    download_llm(llm_url::String, llm_path::String)

Downloads an LLM “llamafile” from `llm_url` to the local path `llm_path`, creating
parent directories if necessary. On non-Windows platforms it marks the file as
executable. On Windows it renames the file by appending ".exe" to the path.
Returns the final local path to the downloaded (and possibly renamed) llamafile.

Throws an exception if the download or permission changes fail.
"""
function download_llm(llm_url::String, llm_path::String)
    if isfile(llm_path)
        @info "LLM already downloaded to $llm_path"
        return llm_path
    end

    @info "Preparing to download llamafile..." url = llm_url path = llm_path

    dir = dirname(llm_path)
    if !isdir(dir)
        @info "Creating directory for LLM: $dir"
        mkdir(dir)
    end

    @info "Downloading llamafile from $llm_url to $llm_path"
    Downloads.download(llm_url, llm_path)

    final_path = llm_path
    if Sys.iswindows()
        if !endswith(lowercase(llm_path), ".exe")
            final_path = llm_path * ".exe"
            mv(llm_path, final_path; force=true)
        end
    else
        run(`chmod +x $llm_path`)
    end

    @info "LLM downloaded successfully to $final_path"
    return final_path
end

"""
    serve_llm(llm_path::String, port::Int=11443; waitfor::Bool=false)

Launches a llamafile-based LLM server on the specified port (default is 11443).
By default, it spawns the llamafile in a **non-blocking** manner, returning a
`Process` object that you can wait on or kill. If you prefer blocking behavior,
pass `waitfor=true`.

The command line used is something like:
`./llm_path --server --host 0.0.0.0 --port 11443 ...`

If you need additional flags (e.g. `--v2`, `--temp`, etc.), either modify
this function or create your own variant.
"""
function serve_llm(llm_path::String, port::Int=LLM_PORT; waitfor::Bool=false)
    local_exe = abspath(llm_path)

    if !Sys.iswindows()
        run(`chmod +x $local_exe`)
    end

    cmd = `$local_exe --server --nobrowser --port $port`
    @info "Starting LLM server at $llm_path on port $port" path = llm_path port = port

    if waitfor
        # Blocking run
        run(cmd; wait=true)
        @info "LLM server has exited (blocking mode)."
        return nothing
    else
        # Non-blocking run
        proc = run(cmd; wait=false)
        @info "LLM server spawned asynchronously" pid = getpid(proc)
        return proc
    end
    return proc
end

"""
    async_run_llm_server(llm_url::String=DEFAULT_LLAMAFILE_URL, 
                         llm_path::String=DEFAULT_LLAMAFILE_PATH,
                         port::Int=DEFAULT_PORT)

Downloads the llamafile from `llm_url` (if needed) into `llm_path`, ensures it is
executable (or renamed on Windows), and then **asynchronously** launches the
server on the given `port`. Returns the process object. The server process
is also registered to shut down when the Julia process exits (using `atexit`).

Example:
```
proc = async_run_llm_server()
# do stuff ...
wait(proc)  # Wait for server to end
```
"""
function async_run_llm_server(
    llm_url::String=LLAMAFILE_URL, llm_path::String=LLAMAFILE_PATH, port::Int=LLM_PORT
)
    local_exe = download_llm(llm_url, llm_path)
    proc = serve_llm(local_exe, port; waitfor=false)
    atexit() do
        try
            if isopen(proc)
                @info "Shutting down LLM server (pid: $(getpid(proc))) at exit."
                kill(proc, Base.SIGTERM)
                wait(proc)
                @info "LLM server closed."
            end
        catch e
            @warn "Error while closing LLM server at exit: $e"
        end
    end
    return proc
end

end # module LLMServeModule
