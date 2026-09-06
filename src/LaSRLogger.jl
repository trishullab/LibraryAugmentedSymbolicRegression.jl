module LaSRLoggerModule

using Base: UUID
using Logging: Logging as LG
using SymbolicRegression
using SymbolicRegression.LoggingModule: should_log

"""
    LaSRLogger(logger)

A logger for LaSR. It wraps `logger`, the base SymbolicRegression.jl `SRLogger`.
"""
Base.@kwdef struct LaSRLogger{L<:SymbolicRegression.LoggingModule.SRLogger} <:
                   SymbolicRegression.AbstractSRLogger
    logger::L
end

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
                if isnothing(value) || length(value) == 0
                    continue
                end
                @info(primary_key, key = value)
            end
        end
    end
end

end
