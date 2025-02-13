# using LibraryAugmentedSymbolicRegression: LLMOptions, Options
# """
# Test if the wrappers around the llm generation commands process the data properly.
    
# Instead of importing aigenerate from the PromptingTools package, we will use a custom aigenerate
# function that will return a predetermined string and a predetermined success value.

# Each stringparsing function will be required to properly convert the predetermined generation
# string into the correct success value.
# """

# # # test that we can partially specify LLMOptions
# # op1 = LLMOptions(; use_llm=false)
# # @test op1.use_llm == false

# # # test that we can fully specify LLMOptions
# # op2 = LLMOptions(;
# #     use_llm=true,
# #     lasr_weights=LLMWeights(; llm_mutate=0.5, llm_crossover=0.3, llm_gen_random=0.2),
# #     num_pareto_context=5,
# #     use_concept_evolution=true,
# #     use_concepts=true,
# #     api_key="vllm_api.key",
# #     model="modelx",
# #     api_kwargs=Dict("url" => "http://localhost:11440/v1"),
# #     http_kwargs=Dict("retries" => 3, "readtimeout" => 3600),
# #     llm_recorder_dir="test/",
# #     llm_context="test",
# #     variable_names=nothing,
# #     max_concepts=30,
# # )
# # @test op2.use_llm == true

# # # test that we can pass LLMOptions to Options
# # llm_opt = LLMOptions(; use_llm=false)
# # op = Options(;
# #     optimizer_options=(iterations=16, f_calls_limit=100, x_tol=1e-16), llm_options=llm_opt
# # )
# # @test isa(op.llm_options, LLMOptions)
# # println("Passed.")
