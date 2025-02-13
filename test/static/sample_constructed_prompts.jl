
# Handle >3 suggestions and references
expected_crossover_1 = """
You are a helpful assistant that recombines two mathematical expressions by following a few provided suggestions. You will be given three suggestions and two reference expressions to recombine.
An expression must consist of the following variables: {{variables}}. All constants will be represented with the symbol C. Each expression will only use these operators: {{operators}}.

Suggestion 1: idea-#1
Suggestion 2: idea-#2
Suggestion 3: idea-#3
Suggestion 4: idea-#4
Reference Expression 1: expression-#1
Reference Expression 2: expression-#2
Reference Expression 3: expression-#3

Propose {{N}} expressions that would be appropriate given the suggestions and references. Provide short commentary for each of your decisions. End with a JSON list that enumerates the proposed expressions following this format:
```json
["expr1",
 "expr2",
 ...
 "expr{{N}}"
]
```
"""

# Handle <3 suggestions and references
expected_crossover_2 = """
You are a helpful assistant that recombines two mathematical expressions by following a few provided suggestions. You will be given three suggestions and two reference expressions to recombine.
An expression must consist of the following variables: {{variables}}. All constants will be represented with the symbol C. Each expression will only use these operators: {{operators}}.

Suggestion 1: idea-#1
Suggestion 2: idea-#2
Reference Expression 1: expression-#1
Reference Expression 2: expression-#2

Propose {{N}} expressions that would be appropriate given the suggestions and references. Provide short commentary for each of your decisions. End with a JSON list that enumerates the proposed expressions following this format:
```json
["expr1",
 "expr2",
 ...
 "expr{{N}}"
]
```
"""

# Handle 1 suggestion and 2 references
expected_crossover_3 = """
You are a helpful assistant that recombines two mathematical expressions by following a few provided suggestions. You will be given three suggestions and two reference expressions to recombine.
An expression must consist of the following variables: {{variables}}. All constants will be represented with the symbol C. Each expression will only use these operators: {{operators}}.

Suggestion 1: idea-#1
Reference Expression 1: expression-#1
Reference Expression 2: expression-#2

Propose {{N}} expressions that would be appropriate given the suggestions and references. Provide short commentary for each of your decisions. End with a JSON list that enumerates the proposed expressions following this format:
```json
["expr1",
 "expr2",
 ...
 "expr{{N}}"
]
```
"""

# Handle 1 suggestion and 2 references with arguments
expected_crossover_4 = """
You are a helpful assistant that recombines two mathematical expressions by following a few provided suggestions. You will be given three suggestions and two reference expressions to recombine.
An expression must consist of the following variables: <DEFAULT VARIABLES>. All constants will be represented with the symbol C. Each expression will only use these operators: <DEFAULT OPERATORS>.

Suggestion 1: idea-#1
Reference Expression 1: expression-#1
Reference Expression 2: expression-#2

Propose <DEFAULT N> expressions that would be appropriate given the suggestions and references. Provide short commentary for each of your decisions. End with a JSON list that enumerates the proposed expressions following this format:
```json
["expr1",
 "expr2",
 ...
 "expr<DEFAULT N>"
]
```
"""

# Handle idea-generation
expected_idea_1 = """
You are a helpful assistant that hypothesizes about the underlying assumptions that generated a list of good and bad mathematical expressions in detailed ways. My ultimate goal is to discover what assumptions generated the observed good mathematical expressions and excludes the bad mathematical expressions. Focus more on the good expressions, their mathematical structure, and any relation to physical concepts. Note that capital C represents an arbitrary constant.

Good Expression 1: good-expression-#1
Good Expression 2: good-expression-#2
Good Expression 3: good-expression-#3
Good Expression 4: good-expression-#4
Good Expression 5: good-expression-#5

Bad Expression 1: bad-expression-#1
Bad Expression 2: bad-expression-#2
Bad Expression 3: bad-expression-#3
Bad Expression 4: bad-expression-#4
Bad Expression 5: bad-expression-#5

Propose {{N}} hypotheses that would be appropriate given the expressions. Provide short commentary for each of your decisions. Do not talk about topics related to the simplicity or complexity of the expressions. I want ideas that are unique and interesting enough to amaze the world's best mathematicians. End with a JSON list that enumerates the proposed hypotheses following this format:
```json
["hyp1",
 "hyp2",
 ...
 "hyp{{N}}"
]
```
"""

# Handle gen-random
expected_gen_random_1 = """
You are a helpful assistant that proposes a mathematical expression by following three provided suggestions.
An expression must consist of the following variables: {{variables}}. All constants will be represented with the symbol C. Each expression will only use these operators: {{operators}}.

Suggestion 1: suggestion-#1
Suggestion 2: suggestion-#2
Suggestion 3: suggestion-#3

Propose {{N}} expressions that would be appropriate given the suggestions. Provide short commentary for each of your decisions. End with a JSON list that enumerates the proposed expressions following this format:
```json
["expr1",
 "expr2",
 ...
 "expr{{N}}"
]
```
"""

# Handle prompt evolution
expected_prompt_evol_1 = """
You are a helpful assistant that merges and refines ideas about a set of hidden mathematical expression in new, interesting, and diverse ways. My ultimate goal is to discover the underlying properties of these hidden expressions. The resulting ideas should be a nontrivial conclusion given the previous ideas.

Idea 1: idea-#1
Idea 2: idea-#2
Idea 3: idea-#3

Propose {{N}} hypotheses that would be appropriate given the ideas. Provide short commentary for each of your decisions. Do not talk about topics related to the simplicity or complexity of the expressions. I want ideas that are unique and interesting enough to amaze the world's best mathematicians. End with a JSON list that enumerates the proposed hypotheses following this format:
```json
["hyp1",
 "hyp2",
 ...
 "hyp{{N}}"
]
```
"""

# Handle prompt evolution: no ideas.
expected_prompt_evol_2 = """
You are a helpful assistant that merges and refines ideas about a set of hidden mathematical expression in new, interesting, and diverse ways. My ultimate goal is to discover the underlying properties of these hidden expressions. The resulting ideas should be a nontrivial conclusion given the previous ideas.


Propose {{N}} hypotheses that would be appropriate given the ideas. Provide short commentary for each of your decisions. Do not talk about topics related to the simplicity or complexity of the expressions. I want ideas that are unique and interesting enough to amaze the world's best mathematicians. End with a JSON list that enumerates the proposed hypotheses following this format:
```json
["hyp1",
 "hyp2",
 ...
 "hyp{{N}}"
]
```
"""

expected_constructions = [
    expected_crossover_1,
    expected_crossover_2,
    expected_crossover_3,
    expected_crossover_4,
    expected_idea_1,
    expected_gen_random_1,
    expected_prompt_evol_1,
    expected_prompt_evol_2,
]
# strip newlines
expected_constructions = [strip(x) for x in expected_constructions]
