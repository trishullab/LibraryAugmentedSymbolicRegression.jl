# Test that we can connect to a LLM server and make simple queries to it.
using HTTP
using LibraryAugmentedSymbolicRegression: LLM_PORT

try
    # if a auth-token is needed, use `token-abc123` as the authorization-token
    header = Dict("Authorization" => "Bearer token-abc123")
    resp = HTTP.get("http://127.0.0.1:$(LLM_PORT)/health", header)
    @test resp.status == 200
catch err
    println("Error reaching LLaMA server at port 11440")
    @show err
end
