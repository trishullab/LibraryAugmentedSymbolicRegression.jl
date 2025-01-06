# LaSR uses an LLM to generate expressions. This test case ensures that the parser can correctly format the LLM outputs.
println("Testing LaSR llm output parser")
using LibraryAugmentedSymbolicRegression: parse_msg_content

include("static/sample_outputs.jl")

for (i, (llm_output, parsed_output)) in
    enumerate(zip(sample_llm_outputs, sample_parsed_outputs))
    @test parse_msg_content(llm_output) == parsed_output
end
println("Passed.")
