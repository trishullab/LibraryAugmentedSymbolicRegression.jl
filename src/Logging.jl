module LoggingModule

using Base: UUID
using Logging: Logging as LG
using SymbolicRegression
using SymbolicRegression.LoggingModule: should_log

"""
    LaSRLogger(logger::SRLogger; **kwargs)

A logger for LaSR that wraps the base SymbolicRegression.jl logger.

# Arguments
- `logger`: The base logger to wrap
"""
Base.@kwdef struct LaSRLogger{L<:SymbolicRegression.LoggingModule.SRLogger} <:
                   SymbolicRegression.AbstractSRLogger
    logger::L
end
# LaSRLogger(logger::SymbolicRegression.LoggingModule.SRLogger; kws...) = LaSRLogger(; logger, kws...)

function log_generation!(
    logger::Union{SymbolicRegression.AbstractSRLogger,Nothing};
    id::UUID,
    mode::String="debug",
    kws...,
)
    if !isnothing(logger) && should_log(logger.logger)
        primary_key = string(id, "/", mode)
        LG.with_logger(logger.logger) do
            for (key, value) in kws
                @info(primary_key, key = value)
            end
        end
    end
end

end
