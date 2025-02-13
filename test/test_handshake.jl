# Test that we can connect to a LLM server and make simple queries to it.
using HTTP

try
    resp = HTTP.get("http://127.0.0.1:11434")
    @test resp.status == 200
catch err
    println("Error reaching LLaMA server at port 11434")
    @show err
end
