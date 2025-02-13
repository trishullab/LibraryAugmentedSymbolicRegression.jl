module LLMServeModule
using Downloads
using Logging
using Base: dirname, isdir, mkdir

const DEFAULT_LLAMAFILE_MODEL = "gemma-2-2b-it.Q6_K"
const DEFAULT_LLAMAFILE_PATH = abspath(
    "$(@__DIR__)/../llamafiles/gemma-2-2b-it.Q6_K.llamafile"
)
const DEFAULT_LLAMAFILE_URL = "https://huggingface.co/Mozilla/gemma-2-2b-it-llamafile/resolve/main/gemma-2-2b-it.Q6_K.llamafile"
const DEFAULT_PORT = 11443

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
function serve_llm(llm_path::String, port::Int=DEFAULT_PORT; waitfor::Bool=false)
    @info "Starting LLM server at $llm_path on port $port" path = llm_path port = port

    local_exe = abspath(llm_path)

    if !Sys.iswindows()
        run(`chmod +x $local_exe`)
    end

    cmd = `$local_exe --server --v2 -l 0.0.0.0:$port`

    if waitfor
        @info "Running server in blocking mode (will not return until process exits)."
        run(cmd)
        return nothing
    else
        proc = run(cmd; wait=false)
        pid = getpid(proc)
        @info "LLM server spawned asynchronously (pid: $(pid))."
        return proc
    end
end

"""
    async_run_llm_server(llm_url::String=DEFAULT_LLAMAFILE_URL, 
                         llm_path::String=DEFAULT_LLAMAFILE_PATH,
                         port::Int=DEFAULT_PORT)

Downloads the llamafile from `llm_url` (if needed) into `llm_path`, ensures it is
executable (or renamed on Windows), and then **asynchronously** launches the
server on the given `port`. Returns the process object.

Example:
```
proc = async_run_llm_server()
# do stuff ...
wait(proc)  # Wait for server to end
```
"""
function async_run_llm_server(
    llm_url::String=DEFAULT_LLAMAFILE_URL,
    llm_path::String=DEFAULT_LLAMAFILE_PATH,
    port::Int=DEFAULT_PORT,
)
    local_exe = download_llm(llm_url, llm_path)
    return serve_llm(local_exe, port; waitfor=false)
end

end # module LLMServeModule
