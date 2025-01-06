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
import SymbolicRegression.MLJInterfaceModule: AbstractMultitargetSRRegressor, 
    AbstractSingletargetSRRegressor,
    AbstractMultitargetSRRegressor,
    modelexpr,
    getsymb,
    get_options,
    full_report,
    unwrap_units_single,
    get_matrix_and_info,
    format_input_for,
    validate_variable_names,
    validate_units,
    prediction_warn,
    eval_tree_mlj,
    choose_best,
    dispatch_selection_for,
    get_equation_strings_for,
    dimension_with_fallback,
    clean_units,
    compat_ustrip,
    SRFitResult

@ignore mutable struct LaSRRegressor <: AbstractSingletargetSRRegressor
    selection_method::Function
end

@ignore mutable struct MultitargetLaSRRegressor <: AbstractMultitargetSRRegressor
    selection_method::Function
end

eval(modelexpr(:LaSRRegressor, :AbstractSingletargetSRRegressor))
eval(modelexpr(:MultitargetLaSRRegressor, :AbstractMultitargetSRRegressor))

MMI.metadata_pkg(
    LaSRRegressor;
    name="LibraryAugmentedSymbolicRegression",
    uuid="158930c3-947c-4174-974b-74b39e64a28f",
    url="https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl",
    julia=true,
    license="Apache-2.0",
    is_wrapper=false,
)

MMI.metadata_pkg(
    MultitargetLaSRRegressor;
    name="LibraryAugmentedSymbolicRegression",
    uuid="158930c3-947c-4174-974b-74b39e64a28f",
    url="https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl",
    julia=true,
    license="Apache-2.0",
    is_wrapper=false,
)

const input_scitype = Union{
    MMI.Table(MMI.Continuous),
    AbstractMatrix{<:MMI.Continuous},
    MMI.Table(MMI.Continuous, MMI.Count),
}

# TODO: Allow for Count data, and coerce it into Continuous as needed.
MMI.metadata_model(
    LaSRRegressor;
    input_scitype,
    target_scitype=AbstractVector{<:Any},
    supports_weights=true,
    reports_feature_importances=false,
    load_path="LibraryAugmentedSymbolicRegression.MLJInterfaceModule.LaSRRegressor",
    human_name="Symbolic Regression accelerated with LLM guidance",
)
MMI.metadata_model(
    MultitargetLaSRRegressor;
    input_scitype,
    target_scitype=Union{MMI.Table(Any),AbstractMatrix{<:Any}},
    supports_weights=true,
    reports_feature_importances=false,
    load_path="LibraryAugmentedSymbolicRegression.MLJInterfaceModule.MultitargetLaSRRegressor",
    human_name="Multi-Target Symbolic Regression accelerated with LLM guidance",
)

end
