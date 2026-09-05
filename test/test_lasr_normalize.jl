using Test
using LibraryAugmentedSymbolicRegression.NormalizeModule:
    NormalizationRule, apply_string_rules, apply_expr_rules,
    rule_pipe_abs, rule_subscript_var, rule_pow_star, rule_c_placeholder,
    rule_unary_sign, rule_strip_lhs

@testset "NormalizationRule + Pair convenience" begin
    r = NormalizationRule(r"x_(\d+)" => s"x\1")
    @test r.stage === :string
    @test r.apply("x_0 + x_1") == "x0 + x1"
end

@testset "LaSR-specific string rules (faithful port)" begin
    strs = [rule_c_placeholder(), rule_pipe_abs(), rule_subscript_var(), rule_pow_star()]
    @test apply_string_rules(strs, "|x0|") == "abs(x0)"           # pipe_abs
    @test apply_string_rules(strs, "x_0 + x_1") == "x0 + x1"      # subscript_var (new; fixes live bug)
    @test apply_string_rules(strs, "2 ** x0") == "2 ^ x0"         # pow_star
    @test occursin("1.0", apply_string_rules(strs, "C * x0"))     # c_placeholder
end

@testset "LaSR-specific expr rules (faithful port)" begin
    exprs = [rule_strip_lhs(), rule_unary_sign()]
    @test apply_expr_rules(exprs, Meta.parse("pow(x0, 2)")) == Meta.parse("x0 ^ 2")   # unary_sign
    @test apply_expr_rules(exprs, Meta.parse("-x0")) == Meta.parse("0.0 - x0")        # unary_sign
    @test apply_expr_rules(exprs, Meta.parse("y = x0 + x1")) == Meta.parse("x0 + x1") # strip_lhs
end

using LibraryAugmentedSymbolicRegression.NormalizeModule:
    rule_implicit_multiplication, rule_implicit_application, rule_function_exponentiation
@testset "inherited SymPy rules" begin
    ss = [rule_implicit_multiplication(), rule_implicit_application(), rule_function_exponentiation()]
    @test apply_string_rules(ss, "3 x y") == "3*x*y" || occursin("3*x*y", replace(apply_string_rules(ss,"3 x y')")," "=>""))
    @test occursin("sin(x)", apply_string_rules([rule_implicit_multiplication(), rule_implicit_application()], "sin x"))
    @test occursin("cos(t)^2", replace(apply_string_rules([rule_function_exponentiation()], "cos^2(t)"), " " => ""))
end

using LibraryAugmentedSymbolicRegression: LaSRPlugin
using LibraryAugmentedSymbolicRegression.NormalizeModule: resolve_rules, DEFAULT_RULES
@testset "extension API" begin
    p = LaSRPlugin(; use_llm=false, parse_rules=[NormalizationRule(r"θ" => "theta")])
    @test length(p.parse_rules) == 1
    resolved = resolve_rules(DEFAULT_RULES, p.parse_rules)
    @test length(resolved) == length(DEFAULT_RULES) + 1
    @test apply_string_rules(resolved, "sin(θ)") == apply_string_rules(resolved, "sin(theta)")
end

using LibraryAugmentedSymbolicRegression: Options, parse_expr, string_tree
using LibraryAugmentedSymbolicRegression.LLMOptionsModule: lasr_context
@testset "extension API: parse_expr honors plugin.parse_rules regardless of options form" begin
    # Regression test for a bug where `parse_expr` resolved the active plugin via
    # `applicable(lasr_plugin, options)` on the pre-conversion argument: every real
    # call site in LLMFunctions.jl passes an already-converted `LaSRContext` (not a
    # raw `Options`), for which `lasr_plugin` has no method, so `applicable` returned
    # false and scientist-registered `parse_rules` were silently dropped.
    options = Options(;
        binary_operators=[+, -, *, /],
        unary_operators=[sin, cos],
        plugins=(
            LaSRPlugin(;
                use_llm=false,
                variable_names=Dict('x' * string(i) => ('x' * string(i)) for i in 0:2),
                parse_rules=[NormalizationRule(r"θ" => "x1")],
            ),
        ),
    )

    t_raw = parse_expr(Float64, "sin(θ)", options)
    t_ctx = parse_expr(Float64, "sin(θ)", lasr_context(options))

    # Must not fall back to DEFAULT_RULES (which would fail to parse `θ` and yield the
    # constant-1 fallback node) in either call form.
    @test string_tree(t_raw, options) == "sin(x1)"
    @test string_tree(t_ctx, options) == "sin(x1)"
end

@testset "indexed constants C1/C2/... -> optimizable constants (LLM dialect)" begin
    # Observed live: the 27B emits multi-coefficient expressions like `C1*x0 + C2*x1`,
    # `C1*x0^2 + C2*x1^2 + C3*x0*x1`. The bare-`C` `c_placeholder` rule does NOT match
    # `C1` (its `(?!\w)` lookahead fails on the trailing digit), so `case_fold` lowercased
    # `C1`->`c1`, which `parse_expression` then rejected as an unknown variable -> every such
    # proposal was silently dropped. `indexed_const` maps each `Ck` to a numeric literal so
    # it becomes an independent optimizable constant.
    strs = filter(r -> r.stage === :string, DEFAULT_RULES)
    out = apply_string_rules(strs, "C1*x0 + C2*x1")
    @test !occursin(r"[Cc]\d", out)          # no surviving indexed-constant token
    @test occursin("x0", out) && occursin("x1", out)

    # End-to-end: `C1*x0 + C2*x1` must parse to a REAL two-variable tree, not the
    # constant-1 fallback (a lone constant node).
    options = Options(;
        binary_operators=[+, -, *, /],
        unary_operators=[sin, cos],
        plugins=(LaSRPlugin(;
            use_llm=false,
            variable_names=Dict("x0" => "x0", "x1" => "x1"),
        ),),
    )
    ex = parse_expr(Float64, "C1*x0 + C2*x1", options)
    rendered = string_tree(ex, options)
    @test occursin("x0", rendered) && occursin("x1", rendered)   # both vars survived
    @test !occursin(r"[Cc]\d", rendered)                          # no stray cN variable
end

@testset "function_exponentiation must not mangle VARIABLES (regression)" begin
    # The rule is for `sin^2 x` -> `sin(x)^2`. It must fire ONLY on known function names, never
    # on a variable: `t^2 + v` must stay `t^2 + v` (the old `[A-Za-z]+` matched `t`, grabbing
    # `+` as its argument -> `t(+)^2 v`, breaking every polynomial once variables are single-char
    # domain symbols like t, v).
    strs = filter(r -> r.stage === :string, DEFAULT_RULES)
    for s in ["C*t^2 + C*v^2", "t^2 + v^2", "C*sqrt(t^2 + v^2)", "cos(t)^2 + v"]
        out = apply_string_rules(strs, s)
        @test !occursin("(+)", out)          # no mangled operator
        @test !occursin(r"[a-z]\(\+\)", out)
    end
    # real function exponentiation still works
    @test occursin("sin(t)^2", apply_string_rules(strs, "sin^2(t)"))
end

@testset "ln -> log (universal natural-log dialect)" begin
    # LLMs write `ln` for natural log across every domain; SR's operator is `log`. Without
    # this rule `ln(x)` is an unknown function and the whole proposal is dropped.
    strs = filter(r -> r.stage === :string, DEFAULT_RULES)
    @test occursin("log(", apply_string_rules(strs, "ln(x0)"))
    @test occursin("log(", apply_string_rules(strs, "LN(x0)"))       # case-insensitive
    @test !occursin(r"\bln\b", apply_string_rules(strs, "ln(x0) + ln(x1)"))
    # word-bounded: must NOT corrupt an identifier that merely contains "ln"
    @test occursin("lnk", apply_string_rules(strs, "lnk + x0"))
end

@testset "case_fold runs before implicit_application (mixed-case function w/o parens)" begin
    # Regression test: `case_fold` used to run *after* `implicit_application` in
    # `DEFAULT_RULES`, so a mixed-case function name written without parens (e.g. an
    # LLM's `Sin x`) was never recognized by `implicit_application` (its operator-name
    # set is lowercase-only) and so never got wrapped as `sin(x)` -- it fell through to
    # `Meta.parse` unparseable and hit the constant-1 fallback. `case_fold` now runs
    # before `implicit_multiplication`/`implicit_application` so this parses correctly.
    options = Options(;
        binary_operators=[+, -, *, /],
        unary_operators=[sin],
        plugins=(LaSRPlugin(; use_llm=false, variable_names=Dict("x" => "x")),),
    )
    ex = parse_expr(Float64, "Sin x", options)
    @test string_tree(ex, options) == "sin(x)"   # NOT the constant-1 fallback
end

using LibraryAugmentedSymbolicRegression.NormalizeModule:
    ParseFailure, ParseFailureStore, record_parse_failure!, parse_failures, parse_failure_summary
@testset "failure store" begin
    store = ParseFailureStore(; cap=3)
    record_parse_failure!(store, ParseFailure("|x", "abs(x", :meta_parse, "unbalanced"))
    record_parse_failure!(store, ParseFailure("|x", "abs(x", :meta_parse, "unbalanced"))
    @test length(parse_failures(store)) == 2
    top = parse_failure_summary(store; n=1)
    @test top[1][1] == "|x" && top[1][2] == 2                       # (raw, count)
    for _ in 1:5; record_parse_failure!(store, ParseFailure("z","z",:tree_parse,"x")); end
    @test length(parse_failures(store)) <= 3                        # bounded
end
