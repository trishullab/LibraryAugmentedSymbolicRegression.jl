module DefaultPrompts
using Base: @kwdef

@kwdef mutable struct Messages
    system_message::AbstractString
    user_messages::Vector{AbstractString}
end

setfield!(m::Messages, field::Symbol, value) = setfield!(m, field, value)
getfield(m::Messages, field::Symbol) = getfield(m, field)

MUTATION_PROMPT = Messages(;
    system_message="""
You are a helpful assistant that mutates a mathematical expression by following a few provided suggestions. You will be given three suggestions and a single reference expression to mutate.
An expression must consist of the following variables: {{variables}}. All constants will be represented with the symbol C. Each expression will only use these operators: {{operators}}.
  """ |> strip,
    user_messages=[
        """
Suggestion 1: {{assump1}}
Suggestion 2: {{assump2}}
Suggestion 3: {{assump3}}
Reference Expression: {{expr}}

Propose a new expression that would be appropriate given the suggestions and references. Provide short commentary for your decision. End with the proposed expression.
""" |> strip,
    ],
)

CROSSOVER_PROMPT = Messages(;
    system_message="""You are a helpful assistant that recombines two mathematical expressions by following a few provided suggestions. You will be given three suggestions and two reference expressions to recombine.
An expression must consist of the following variables: {{variables}}. All constants will be represented with the symbol C. Each expression will only use these operators: {{operators}}.
    """ |> strip,
    user_messages=[
        """
Suggestion 1: {{assump1}}
Suggestion 2: {{assump2}}
Suggestion 3: {{assump3}}
Reference Expression 1: {{expr1}}
Reference Expression 2: {{expr2}}

Propose {{N}} expressions that would be appropriate given the suggestions and references. Provide short commentary for each of your decisions. End with a JSON list that enumerates the proposed expressions following this format:
```json
["expr1",
 "expr2",
 ...
 "expr{{N}}"
]
```""" |> strip,
    ],
)

GEN_RANDOM_PROMPT = Messages(;
    system_message="""You are a helpful assistant that proposes a mathematical expression by following three provided suggestions.
An expression must consist of the following variables: {{variables}}. All constants will be represented with the symbol C. Each expression will only use these operators: {{operators}}.
    """ |> strip,
    user_messages=[
        """Suggestion 1: {{assump1}}
Suggestion 2: {{assump2}}
Suggestion 3: {{assump3}}

Propose {{N}} expressions that would be appropriate given the suggestions. Provide short commentary for each of your decisions. End with a JSON list that enumerates the proposed expressions following this format:
```json
["expr1",
 "expr2",
 ...
 "expr{{N}}"
]
```""" |> strip,
    ],
)

CONCEPT_EVOLUTION_PROMPT = Messages(;
    system_message="""You are a helpful assistant that merges and refines ideas about a set of hidden mathematical expression in new, interesting, and diverse ways. My ultimate goal is to discover the underlying properties of these hidden expressions. The resulting ideas should be a nontrivial conclusion given the previous ideas.
    """ |> strip,
    user_messages=[
        """Idea 1: {{idea1}}
Idea 2: {{idea2}}
Idea 3: {{idea3}}
Idea 4: {{idea4}}
Idea 5: {{idea5}}

Propose {{N}} hypotheses that would be appropriate given the ideas. Provide short commentary for each of your decisions. Do not talk about topics related to the simplicity or complexity of the expressions. I want ideas that are unique and interesting enough to amaze the world's best mathematicians. End with a JSON list that enumerates the proposed hypotheses following this format:
```json
["hyp1",
 "hyp2",
 ...
 "hyp{{N}}"
]
```""" |> strip,
    ],
)

GENERATE_CONCEPT_PROMPT = Messages(;
    system_message="""You are a helpful assistant that hypothesizes about the underlying assumptions that generated a list of good and bad mathematical expressions in detailed ways. My ultimate goal is to discover what assumptions generated the observed good mathematical expressions and excludes the bad mathematical expressions. Focus more on the good expressions, their mathematical structure, and any relation to physical concepts. Note that capital C represents an arbitrary constant.
    """ |> strip,
    user_messages=[
        """Good Expression 1: {{gexpr1}}
Good Expression 2: {{gexpr2}}
Good Expression 3: {{gexpr3}}
Good Expression 4: {{gexpr4}}
Good Expression 5: {{gexpr5}}

Bad Expression 1: {{bexpr1}}
Bad Expression 2: {{bexpr2}}
Bad Expression 3: {{bexpr3}}
Bad Expression 4: {{bexpr4}}
Bad Expression 5: {{bexpr5}}

Propose {{N}} hypotheses that would be appropriate given the expressions. Provide short commentary for each of your decisions. Do not talk about topics related to the simplicity or complexity of the expressions. I want ideas that are unique and interesting enough to amaze the world's best mathematicians. End with a JSON list that enumerates the proposed hypotheses following this format:
```json
["hyp1",
 "hyp2",
 ...
 "hyp{{N}}"
]
```""" |> strip,
    ],
)

@kwdef mutable struct Prompts
    mutation::Messages = MUTATION_PROMPT
    crossover::Messages = CROSSOVER_PROMPT
    gen_random::Messages = GEN_RANDOM_PROMPT
    concept_evolution::Messages = CONCEPT_EVOLUTION_PROMPT
    generate_concept::Messages = GENERATE_CONCEPT_PROMPT
end

const DEFAULT_PROMPTS = Prompts()

end
