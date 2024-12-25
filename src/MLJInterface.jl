module MLJInterfaceModule
using Optim: Optim
using LineSearches: LineSearches
using Logging: AbstractLogger
using MLJModelInterface: MLJModelInterface as MMI
using ADTypes: AbstractADType
using DynamicExpressions:
    eval_tree_array,
    string_tree,
    AbstractExpressionNode,
    AbstractExpression,
    Node,
    Expression,
    default_node_type,
    get_tree
using DynamicQuantities:
    QuantityArray,
    UnionAbstractQuantity,
    AbstractDimensions,
    SymbolicDimensions,
    Quantity,
    DEFAULT_DIM_BASE_TYPE,
    ustrip,
    dimension
using LossFunctions: SupervisedLoss
using SymbolicRegression.InterfaceDynamicQuantitiesModule: get_dimensions_type
using SymbolicRegression.CoreModule:
    Options, Dataset, AbstractMutationWeights, MutationWeights, LOSS_TYPE, ComplexityMapping
using SymbolicRegression.CoreModule.OptionsModule: DEFAULT_OPTIONS, OPTION_DESCRIPTIONS
using SymbolicRegression.ComplexityModule: compute_complexity
using SymbolicRegression.HallOfFameModule: HallOfFame, format_hall_of_fame
using SymbolicRegression.UtilsModule: subscriptify, @ignore
using SymbolicRegression.LoggingModule: AbstractSRLogger
using SymbolicRegression.MLJInterfaceModule: modelexpr, AbstractSRRegressor

# For static analysis tools:
@ignore mutable struct LaSRRegressor <: AbstractSRRegressor
    selection_method::Function
end

@ignore mutable struct MultitargetLaSRRegressor <: AbstractSRRegressor
    selection_method::Function
end

eval(modelexpr(:LaSRRegressor))
eval(modelexpr(:MultitargetLaSRRegressor))

end
