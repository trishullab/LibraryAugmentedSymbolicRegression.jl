module MutateModule

using DispatchDoctor: @unstable
using DynamicExpressions: copy_into!, with_contents, allocate_container, get_tree
using UUIDs: uuid1

using SymbolicRegression
using SymbolicRegression:
    max_features,
    check_constraints,
    compute_complexity,
    condition_mutation_weights!,
    sample_mutation,
    eval_cost,
    crossover_trees,
    combine_operators,
    simplify_tree!,
    gen_random_tree_fixed_size

import SymbolicRegression: AbstractPopMember
import SymbolicRegression.PopMemberModule: create_child

using SymbolicRegression.CoreModule: dataset_fraction
using SymbolicRegression.MutateModule: AbstractMutationResult
using .SymbolicRegression: @recorder

# It's important that we explicitly import the mutate! function from SymbolicRegression
# so Julia knows that we're extending it.
import SymbolicRegression: mutate!, MutationResult, crossover_trees
import SymbolicRegression.MutateModule:
    next_generation, _dispatch_mutations!, crossover_generation

using ..CoreModule: LaSROptions
using ..LLMFunctionsModule: llm_mutate_tree, llm_crossover_trees, llm_randomize_tree
using ..LoggingModule: log_generation!
using ..ParseModule: render_expr, parse_expr

function check_constant(tree::AbstractExpressionNode)::Bool
    return (tree.degree == 0) && tree.constant
end

function check_constant(tree::AbstractExpression)::Bool
    return check_constant(get_tree(tree))
end

function mutate!(
    tree::N,
    member::P,
    ::Val{:llm_mutate},
    ::SymbolicRegression.AbstractMutationWeights,
    options::SymbolicRegression.AbstractOptions;
    recorder::SymbolicRegression.RecordType,
    kws...,
) where {T,N<:SymbolicRegression.AbstractExpression{T},P<:AbstractPopMember}
    tree = llm_mutate_tree(tree, options)
    @recorder recorder["type"] = "llm_mutate"
    return MutationResult{N,P}(; tree=tree)
end

function mutate!(
    tree::N,
    member::P,
    ::Val{:llm_randomize},
    ::SymbolicRegression.AbstractMutationWeights,
    options::SymbolicRegression.AbstractOptions;
    recorder::SymbolicRegression.RecordType,
    curmaxsize,
    nfeatures,
    kws...,
) where {T,N<:SymbolicRegression.AbstractExpression{T},P<:AbstractPopMember}
    tree = llm_randomize_tree(tree, curmaxsize, options, nfeatures)
    @recorder recorder["type"] = "llm_randomize"
    return MutationResult{N,P}(; tree=tree)
end

"""
Generate a generation via crossover of two members.
"""
function crossover_generation(
    member1::P,
    member2::P,
    dataset::D,
    curmaxsize::Int,
    options::LaSROptions;
    recorder::SymbolicRegression.RecordType=SymbolicRegression.RecordType(),
)::Tuple{
    P,P,Bool,Float64
} where {T,L,D<:SymbolicRegression.Dataset{T,L},N,P<:AbstractPopMember{T,L,N}}
    llm_skip = false
    num_evals = 0.0

    if options.use_llm && (rand() < options.llm_operation_weights.llm_crossover)
        tree1 = member1.tree
        tree2 = member2.tree

        # add simplification for crossover
        tree1 = simplify_tree!(tree1, options.operators)
        tree1 = combine_operators(tree1, options.operators)
        tree2 = simplify_tree!(tree2, options.operators)
        tree2 = combine_operators(tree2, options.operators)

        nfeatures = dataset.nfeatures

        # if it is constant, replace with random
        if check_constant(tree1)
            tree1 = with_contents(
                tree1, gen_random_tree_fixed_size(rand(1:curmaxsize), options, nfeatures, T)
            )
        end
        if check_constant(tree2)
            tree2 = with_contents(
                tree2, gen_random_tree_fixed_size(rand(1:curmaxsize), options, nfeatures, T)
            )
        end

        child_tree1, child_tree2 = llm_crossover_trees(tree1, tree2, options)

        child_tree1 = simplify_tree!(child_tree1, options.operators)
        child_tree1 = combine_operators(child_tree1, options.operators)
        child_tree2 = simplify_tree!(child_tree2, options.operators)
        child_tree2 = combine_operators(child_tree2, options.operators)

        afterSize1 = compute_complexity(child_tree1, options)
        afterSize2 = compute_complexity(child_tree2, options)

        successful_crossover =
            (!check_constant(child_tree1)) &&
            (!check_constant(child_tree2)) &&
            check_constraints(child_tree1, options, curmaxsize, afterSize1) &&
            check_constraints(child_tree2, options, curmaxsize, afterSize2)

        # Generate a new UUID for logging this crossover attempt
        gen_id = uuid1()
        recorder_str =
            render_expr(child_tree1, options) * " && " * render_expr(child_tree2, options)

        if successful_crossover
            log_generation!(
                options.lasr_logger; id=gen_id, mode="crossover", chosen=recorder_str
            )
            llm_skip = true
        else
            log_generation!(
                options.lasr_logger; id=gen_id, mode="crossover", failed=recorder_str
            )
            child_tree1, child_tree2 = crossover_trees(tree1, tree2)
        end
    end

    if !llm_skip
        return crossover_generation(
            member1, member2, dataset, curmaxsize, options.sr_options; recorder=recorder
        )
    else
        after_cost1, after_loss1 = eval_cost(
            dataset, child_tree1, options; complexity=afterSize1
        )
        after_cost2, after_loss2 = eval_cost(
            dataset, child_tree2, options; complexity=afterSize2
        )
        num_evals += 2.0 * dataset_fraction(dataset)
        baby1 = create_child(
            (member1, member2),
            child_tree1::AbstractExpression,
            after_cost1,
            after_loss1,
            options;
            complexity=afterSize1,
            parent_ref=member1.ref,
            mutation_choice=:llm_crossover,
        )::P
        baby2 = create_child(
            (member1, member2),
            child_tree2::AbstractExpression,
            after_cost2,
            after_loss2,
            options;
            complexity=afterSize2,
            parent_ref=member2.ref,
            mutation_choice=:llm_crossover,
        )::P
        @recorder begin
            recorder["result"] = "accept"
            recorder["reason"] = "pass"
        end

        crossover_accepted = true
        return baby1, baby2, crossover_accepted, num_evals
    end
end

end
