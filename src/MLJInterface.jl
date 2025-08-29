module MLJInterfaceModule
using Optim: Optim
using LineSearches: LineSearches
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
    get_tree,
    AbstractOperatorEnum

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
using SymbolicRegression
using SymbolicRegression.InterfaceDynamicQuantitiesModule: get_dimensions_type
using SymbolicRegression.CoreModule:
    Options, Dataset, AbstractMutationWeights, MutationWeights, LOSS_TYPE, ComplexityMapping
using SymbolicRegression.CoreModule.OptionsModule: DEFAULT_OPTIONS, OPTION_DESCRIPTIONS
using SymbolicRegression.ComplexityModule: compute_complexity
using SymbolicRegression.HallOfFameModule: HallOfFame, format_hall_of_fame
using SymbolicRegression.UtilsModule: subscriptify
using SymbolicRegression.LoggingModule: AbstractSRLogger
import SymbolicRegression.MLJInterfaceModule:
    AbstractSymbolicRegressor,
    AbstractSingletargetSRRegressor,
    AbstractMultitargetSRRegressor,
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
    AbstractExpressionSpec,
    SRFitResultTypes

using ..CoreModule:
    LaSRMutationWeights, LLMOperationWeights, LLMOptions, LaSROptions, LASR_DEFAULT_OPTIONS

using ..UtilsModule: @ignore, unique_by_argname

@ignore mutable struct LaSRRegressor <: AbstractSingletargetSRRegressor
    selection_method::Function
end

@ignore mutable struct MultitargetLaSRRegressor <: AbstractMultitargetSRRegressor
    selection_method::Function
end

@ignore mutable struct LaSRTestRegressor <: AbstractSingletargetSRRegressor
    selection_method::Function
end

@ignore mutable struct MultitargetLaSRTestRegressor <: AbstractMultitargetSRRegressor
    selection_method::Function
end

"""Generate an `SRRegressor` struct containing all the fields in `Options`."""
function modelexpr(
    model_name::Symbol,
    parent_type::Symbol=:AbstractSymbolicRegressor;
    default_niterations=100,
)
    struct_def =
        :(Base.@kwdef mutable struct $(model_name){D<:AbstractDimensions,L} <: $parent_type
            niterations::Int = $(default_niterations)
            parallelism::Symbol = :multithreading
            numprocs::Union{Int,Nothing} = nothing
            procs::Union{Vector{Int},Nothing} = nothing
            addprocs_function::Union{Function,Nothing} = nothing
            heap_size_hint_in_bytes::Union{Integer,Nothing} = nothing
            worker_timeout::Union{Real,Nothing} = nothing
            worker_imports::Union{Vector{Symbol},Nothing} = nothing
            logger::Union{AbstractSRLogger,Nothing} = nothing
            runtests::Bool = true
            run_id::Union{String,Nothing} = nothing
            loss_type::Type{L} = Nothing
            guesses::Union{AbstractVector,AbstractVector{<:AbstractVector},Nothing} =
                nothing
            selection_method::Function = choose_best
            dimensions_type::Type{D} = SymbolicDimensions{DEFAULT_DIM_BASE_TYPE}
        end)
    # TODO: store `procs` from initial run if parallelism is `:multiprocessing`
    fields = last(last(struct_def.args).args).args

    # @TODO: Make a default options for LaSR and use it here instead.
    DEFAULT_OPTIONS_MERGED = unique_by_argname([LASR_DEFAULT_OPTIONS; DEFAULT_OPTIONS])

    # Add everything from `Options` constructor directly to struct:
    for (i, option) in enumerate(DEFAULT_OPTIONS_MERGED)
        insert!(fields, i, Expr(:(=), option.args...))
    end

    # We also need to create the `get_options` function, based on this:
    constructor = :(LaSROptions(;))
    constructor_fields = last(constructor.args).args

    for option in DEFAULT_OPTIONS_MERGED
        symb = getsymb(first(option.args))
        push!(constructor_fields, Expr(:kw, symb, Expr(:(.), :m, Core.QuoteNode(symb))))
    end

    return quote
        $struct_def
        function get_options(m::$(model_name))
            return $constructor
        end
    end
end

eval(modelexpr(:LaSRRegressor, :AbstractSingletargetSRRegressor))
eval(modelexpr(:MultitargetLaSRRegressor, :AbstractMultitargetSRRegressor))

eval(modelexpr(:LaSRTestRegressor, :AbstractSingletargetSRRegressor; default_niterations=1))
eval(
    modelexpr(
        :MultitargetLaSRTestRegressor,
        :AbstractMultitargetSRRegressor;
        default_niterations=1,
    ),
)

const input_scitype = Union{
    MMI.Table(MMI.Continuous),
    AbstractMatrix{<:MMI.Continuous},
    MMI.Table(MMI.Continuous, MMI.Count),
}

for model in [:LaSRRegressor, :LaSRTestRegressor]
    @eval begin
        MMI.metadata_model(
            $model;
            input_scitype,
            target_scitype=AbstractVector{<:Any},
            supports_weights=true,
            reports_feature_importances=false,
            load_path=(
                $("LibraryAugmentedSymbolicRegression.MLJInterfaceModule." * string(model))
            ),
            human_name="Symbolic Regression accelerated with LLM guidance",
        )
    end
end

for model in [:MultitargetLaSRRegressor, :MultitargetLaSRTestRegressor]
    @eval begin
        MMI.metadata_model(
            $model;
            input_scitype,
            target_scitype=Union{
                MMI.Table(MMI.Continuous),AbstractMatrix{<:MMI.Continuous}
            },
            supports_weights=true,
            reports_feature_importances=false,
            load_path=(
                $("LibraryAugmentedSymbolicRegression.MLJInterfaceModule." * string(model))
            ),
            human_name="Multi-Target Symbolic Regression accelerated with LLM guidance",
        )
    end
end

end
