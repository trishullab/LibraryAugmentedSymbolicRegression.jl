# LaSR uses prompt templates specified as txt files to generate LLM prompts. This test ensures that the parser can correctly parse these templates.
println("Testing LaSR prompt construction")
using LibraryAugmentedSymbolicRegression: construct_prompt, load_prompt
using PromptingTools:
    render, CustomOpenAISchema, SystemMessage, UserMessage, AbstractChatMessage

all_prompts = Dict(
    replace(filepath, ".txt" => "") => load_prompt("static/test_prompts/" * filepath) for
    filepath in readdir("static/test_prompts/"; join=false)
)

# loads expected_constructions
include("static/sample_constructed_prompts.jl")

function render_conversation(system_message, user_messages; kws...)
    conversation = AbstractChatMessage[]
    push!(conversation, SystemMessage(system_message))
    for user_message in user_messages
        push!(conversation, UserMessage(user_message))
    end
    rendered_msg = strip(
        join(
            [
                x["content"] for x in render(
                    CustomOpenAISchema(),
                    conversation;
                    no_system_message=false,
                    # other kwargs should be passed here
                    kws...,
                )
            ],
            "\n\n",
        ),
    )
    return rendered_msg
end

function nlist(prefix, n)
    return [prefix * string(i) for i in 1:n]
end

# crossover: Handle >3 suggestions and references
system_message = all_prompts["crossover_system"]
user_messages = [
    construct_prompt(
        construct_prompt(all_prompts["crossover_user"], nlist("expression-#", 3), "expr"),
        nlist("idea-#", 4),
        "assump",
    ),
]

@test render_conversation(system_message, user_messages) == expected_constructions[1]

# crossover: Handle <3 suggestions and references
user_messages = [
    construct_prompt(
        construct_prompt(all_prompts["crossover_user"], nlist("expression-#", 2), "expr"),
        nlist("idea-#", 2),
        "assump",
    ),
]

@test render_conversation(system_message, user_messages) == expected_constructions[2]

# crossover: Handle 1 suggestion and 2 references
user_messages = [
    construct_prompt(
        construct_prompt(all_prompts["crossover_user"], nlist("expression-#", 2), "expr"),
        nlist("idea-#", 1),
        "assump",
    ),
]

@test render_conversation(system_message, user_messages) == expected_constructions[3]

# crossover: Handle 1 suggestion and 2 references with default arguments
@test render_conversation(
    system_message,
    user_messages;
    operators="<DEFAULT OPERATORS>",
    variables="<DEFAULT VARIABLES>",
    N="<DEFAULT N>",
) == expected_constructions[4]

# Idea generation: Handle default case
system_message = all_prompts["extract_idea_system"]
user_messages = [
    construct_prompt(
        construct_prompt(
            all_prompts["extract_idea_user"], nlist("good-expression-#", 5), "gexpr"
        ),
        nlist("bad-expression-#", 5),
        "bexpr",
    ),
]

@test render_conversation(system_message, user_messages) == expected_constructions[5]

# Gen random: Handle default case
system_message = all_prompts["gen_random_system"]
user_messages = [
    construct_prompt(all_prompts["gen_random_user"], nlist("suggestion-#", 3), "assump")
]

@test render_conversation(system_message, user_messages) == expected_constructions[6]

# Prompt evolution: Handle default case
system_message = all_prompts["prompt_evol_system"]
user_messages = [
    construct_prompt(all_prompts["prompt_evol_user"], nlist("idea-#", 3), "idea")
]

@test render_conversation(system_message, user_messages) == expected_constructions[7]

# Prompt evolution: Handle no ideas.
user_messages = [construct_prompt(all_prompts["prompt_evol_user"], [], "idea")]

@test render_conversation(system_message, user_messages) == expected_constructions[8]
println("All tests passed!")
