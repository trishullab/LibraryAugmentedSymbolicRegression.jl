# Static analysis of LaSR's own code.
#
# `report_package` without `target_modules` also walks into Optim, NLSolversBase,
# DynamicExpressions and PromptingTools, which report 24 errors LaSR cannot fix. Scoping the
# analysis to this package's modules keeps the signal (0 findings as of 2026-09-01) without
# failing on third-party internals.
if !(VERSION >= v"1.10.0")
    exit(0)
end

dir = mktempdir()

@info "Starting test_jet.jl" dir

using Pkg
@info "Creating environment..."
Pkg.activate(dir; io=devnull)
Pkg.develop(; path=dirname(@__DIR__), io=devnull)
Pkg.add(["JET", "Preferences", "DynamicExpressions"]; io=devnull)
@info "Done!"

using Preferences
cd(dir)
Preferences.set_preferences!(
    "LibraryAugmentedSymbolicRegression", "dispatch_doctor_mode" => "disable"; force=true
)
Preferences.set_preferences!(
    "DynamicExpressions", "dispatch_doctor_mode" => "disable"; force=true
)

using LibraryAugmentedSymbolicRegression
using JET

@info "Running tests..."
JET.test_package(
    LibraryAugmentedSymbolicRegression;
    target_modules=(LibraryAugmentedSymbolicRegression,),
)
@info "Done!"

@info "test_jet.jl finished"
