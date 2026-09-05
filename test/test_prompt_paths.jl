# Prompt-template location and resolution.
#
# LaSR ships its templates inside the installed package, which is READ-ONLY on a
# `Pkg.add` install (files land mode 444) and whose absolute path must never be
# frozen into the precompile cache -- a `const` computed from `pkgdir` at
# precompile time survives a depot move/copy as a stale path while the pkgimage
# stays valid, and every prompt load then fails with "No such file or directory".
# So: the default is a FUNCTION, user dirs are joined (not concatenated), a
# missing template falls back to the packaged copy, and a bad dir fails loudly at
# construction instead of minutes into a search.
using Test
import LibraryAugmentedSymbolicRegression as LaSR
using LibraryAugmentedSymbolicRegression:
    LaSRPlugin,
    default_prompts_dir,
    prompt_path,
    copy_prompts,
    load_prompt,
    construct_prompt,
    llm_mutate_tree
using SymbolicRegression: Options
using DynamicExpressions: Node

@testset "default_prompts_dir is resolved at call time, not baked at precompile" begin
    @test default_prompts_dir isa Function
    d = default_prompts_dir()
    @test isdir(d)
    @test isfile(joinpath(d, "mutate_system.prompt"))
    @test d == normpath(joinpath(pkgdir(LaSR), "prompts"))
end

@testset "prompt_path prefers a template from the user's directory" begin
    mktempdir() do tmp
        write(joinpath(tmp, "mutate_system.prompt"), "custom system prompt")
        @test prompt_path(tmp, "mutate_system.prompt") ==
            joinpath(tmp, "mutate_system.prompt")
    end
end

@testset "prompt_path falls back to the packaged template per file" begin
    mktempdir() do tmp
        write(joinpath(tmp, "mutate_system.prompt"), "custom system prompt")
        # only `mutate_system` was overridden; the rest come from the package
        @test prompt_path(tmp, "mutate_user.prompt") ==
            joinpath(default_prompts_dir(), "mutate_user.prompt")
    end
end

@testset "prompt_path ignores a trailing separator" begin
    d = default_prompts_dir()
    @test prompt_path(d * "/", "crossover_user.prompt") ==
        prompt_path(d, "crossover_user.prompt")
end

@testset "prompt_path reports an unknown template clearly" begin
    err = try
        prompt_path(default_prompts_dir(), "no_such_template.prompt")
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("no_such_template.prompt", err.msg)
end

@testset "LaSRPlugin defaults prompts_dir to the packaged templates" begin
    plugin = LaSRPlugin(; use_llm=false, variable_names=Dict(1 => "x1"))
    @test plugin.prompts_dir == default_prompts_dir()
end

@testset "LaSRPlugin normalizes prompts_dir with or without a trailing slash" begin
    mktempdir() do tmp
        kws = (; use_llm=false, variable_names=Dict(1 => "x1"))
        @test LaSRPlugin(; kws..., prompts_dir=tmp).prompts_dir ==
            LaSRPlugin(; kws..., prompts_dir=tmp * "/").prompts_dir ==
            normpath(tmp)
    end
end

@testset "LaSRPlugin rejects a nonexistent prompts_dir at construction" begin
    bad = joinpath(tempdir(), "lasr_no_such_prompts_dir")
    err = try
        LaSRPlugin(; use_llm=false, variable_names=Dict(1 => "x1"), prompts_dir=bad)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin(bad, err.msg)
end

@testset "copy_prompts materializes writable templates for editing" begin
    mktempdir() do tmp
        dest = joinpath(tmp, "my_prompts")
        @test copy_prompts(dest) == dest
        for name in readdir(default_prompts_dir())
            endswith(name, ".prompt") || continue
            copied = joinpath(dest, name)
            @test isfile(copied)
            # the packaged originals are mode 444 on a Pkg.add install; copies must be editable
            @test Base.Filesystem.uperm(copied) & 0x02 != 0
            open(io -> write(io, "\nedited"), copied, "a")
        end
    end
end

@testset "copy_prompts does not clobber edits unless forced" begin
    mktempdir() do tmp
        copy_prompts(tmp)
        edited = joinpath(tmp, "mutate_system.prompt")
        write(edited, "my edited prompt")
        copy_prompts(tmp)
        @test read(edited, String) == "my edited prompt"
        copy_prompts(tmp; force=true)
        @test read(edited, String) ==
            read(joinpath(default_prompts_dir(), "mutate_system.prompt"), String)
    end
end

@testset "prompt loading in a search uses the override and falls back for the rest" begin
    mktempdir() do tmp
        write(joinpath(tmp, "mutate_system.prompt"), "custom system prompt")
        captured = Ref{Any}(nothing)
        mock_llm = function (args...; kwargs...)
            captured[] = args[2]
            return (; content="[\"x1 + 1\"]")
        end
        plugin = LaSRPlugin(;
            llm_generate=mock_llm,
            variable_names=Dict(1 => "x1"),
            prompts_dir=tmp,
            use_concepts=false,
        )
        options = Options(; plugins=(plugin,), binary_operators=[+, *])
        llm_mutate_tree(Node{Float64}(; feature=1), options)

        conversation = captured[]
        @test conversation[1].content == "custom system prompt"
        @test conversation[2].content == construct_prompt(
            load_prompt(joinpath(default_prompts_dir(), "mutate_user.prompt")), [], "assump"
        )
    end
end
