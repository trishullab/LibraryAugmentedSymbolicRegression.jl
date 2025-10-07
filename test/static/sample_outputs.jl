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

expert_sample_llm_outputs = [
    """```json
    [
        "v = c * sqrt(1 - (m_0 / m)^2)",
        "v = c * (1 - m_0/m)",
        "v = c * (m - m_0) / m",
        "v = c * sqrt(1 - (m_0^2 / m^2))",
        "v = c / sqrt((m / m_0)^2 - 1)"
    ]
    ```""",
    """```json
    [
    "c * sqrt(1 - (m_0/m)^2)",
    "c / sqrt(1 + (m/m_0)^2)",
    "c * sin(arccos(m_0/m))",
    "c * sqrt(1 - exp(2 * log(m_0/m)))",
    "c * sqrt(1 - (m_0^2/m^2))"
    ]
    ```""",
    """["cos(c / sqrt(((m / m_0) ^ 2.0) * 1.0))"]""",
    """["c * sin(cos(m_0 / m))"]""",
    """["C",
        "m",
        "m_0",
        "c",
        "v",
        "cos(c)",
        "sin(c)",
        "sqrt(C)",
        "log(C)",
        "sqrt(m_0)",
        "cos(m)",
        "cos(m_0)",
        "sin(m_0)",
        "log(m)",
        "log(m_0)",
        "sqrt(m)",
        "exp(m)",
        "inv(c)",
        "inv(m)",
        "inv(inv(m))"]""",
]

expert_sample_parsed_outputs = [
    [
        "v = c * sqrt(1 - (m_0 / m)^2)",
        "v = c * (1 - m_0/m)",
        "v = c * (m - m_0) / m",
        "v = c * sqrt(1 - (m_0^2 / m^2))",
        "v = c / sqrt((m / m_0)^2 - 1)",
    ],
    [
        "c * sqrt(1 - (m_0/m)^2)",
        "c / sqrt(1 + (m/m_0)^2)",
        "c * sin(arccos(m_0/m))",
        "c * sqrt(1 - exp(2 * log(m_0/m)))",
        "c * sqrt(1 - (m_0^2/m^2))",
    ],
    ["cos(c / sqrt(((m / m_0) ^ 2.0) * 1.0))"],
    ["c * sin(cos(m_0 / m))"],
    [
        "C",
        "m",
        "m_0",
        "c",
        "v",
        "cos(c)",
        "sin(c)",
        "sqrt(C)",
        "log(C)",
        "sqrt(m_0)",
        "cos(m)",
        "cos(m_0)",
        "sin(m_0)",
        "log(m)",
        "log(m_0)",
        "sqrt(m)",
        "exp(m)",
        "inv(c)",
        "inv(m)",
        "inv(inv(m))",
    ],
]
