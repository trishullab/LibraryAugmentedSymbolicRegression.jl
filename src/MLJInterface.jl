module MLJInterfaceModule

using SymbolicRegression: SRRegressor, MultitargetSRRegressor
using ..LLMOptionsStructModule: LaSRPlugin

function LaSRRegressor(;
    plugin::LaSRPlugin=LaSRPlugin(), plugins::Union{Tuple,AbstractVector}=(), kws...
)
    return SRRegressor(; kws..., plugins=(Tuple(plugins)..., plugin))
end

function MultitargetLaSRRegressor(;
    plugin::LaSRPlugin=LaSRPlugin(), plugins::Union{Tuple,AbstractVector}=(), kws...
)
    return MultitargetSRRegressor(; kws..., plugins=(Tuple(plugins)..., plugin))
end

end
