sample_llm_outputs = [
    # Correct formatting
    """```json
    [
        "x1 / x2",
        "x1 ^ x2",
        "(x1 + x2) / x3 * (x4 ^ x5)"
    ]
    ```
    """,
    # Slightly incorrect formatting
    """[
        "x1 / x2",
        "x1 ^ x2",
        "(x1 + x2) / x3 * (x4 ^ x5)"
    ]
    """,
    # Incorrect formatting
    """{
    "x1 / x2",
    "x1 ^ x2",
    "(x1 + x2) / x3 * (x4 ^ x5)"
    }
    """,

    # Incorrect formatting
    """[
        "x1 / x2",
        "x1 ^ x2",
        "(x1 + x2) / x3 * (x4 ^ x5)"
    """,

    # Incorrect formatting (recovery possible)
    """{
        "equations": [
            "x1 / x2",
            "x1 ^ x2",
            "(x1 + x2) / x3 * (x4 ^ x5)"
        ]
    }
    """,

    # Incorrect formatting (recovery impossible)
    """{
        "equations": {
            "1" : "x1 / x2",
            "2" : "x1 ^ x2",
            "3" : "(x1 + x2) / x3 * (x4 ^ x5)"
        }
    }""",
]

sample_parsed_outputs = [
    ["x1 / x2", "x1 ^ x2", "(x1 + x2) / x3 * (x4 ^ x5)"],
    ["x1 / x2", "x1 ^ x2", "(x1 + x2) / x3 * (x4 ^ x5)"],
    String[],
    String[],
    ["x1 / x2", "x1 ^ x2", "(x1 + x2) / x3 * (x4 ^ x5)"],
    String[],
]
