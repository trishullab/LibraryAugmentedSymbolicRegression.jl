# LaSR uses an LLM to generate expressions. This test case ensures that the parser can correctly format the LLM outputs.
println("Testing LaSR llm output parser")
using LibraryAugmentedSymbolicRegression: LaSROptions, parse_msg_content

include("test_params.jl")
options = LaSROptions(;
    default_params..., binary_operators=[+, *, ^, -], unary_operators=[sin, cos, exp]
)

include("static/sample_outputs.jl")

for (i, (llm_output, parsed_output)) in
    enumerate(zip(sample_llm_outputs, sample_parsed_outputs))
    @test parse_msg_content(llm_output, options) == parsed_output
end
println("Passed.")

for (i, (llm_output, parsed_output)) in
    enumerate(zip(expert_sample_llm_outputs, expert_sample_parsed_outputs))
    @test parse_msg_content(llm_output, options) == parsed_output
end
println("Passed.")
