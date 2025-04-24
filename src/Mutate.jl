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
    gen_random_tree_fixed_size,
    PopMember

using SymbolicRegression.CoreModule: dataset_fraction
using SymbolicRegression.MutateModule: AbstractMutationResult
using .SymbolicRegression: @recorder

# It's important that we explicitly import the mutate! function from SymbolicRegression
# so Julia knows that we're extending it.
import SymbolicRegression: mutate!, MutationResult
import SymbolicRegression.MutateModule:
    next_generation, _dispatch_mutations!, crossover_generation

using ..CoreModule: LaSROptions
using ..LLMFunctionsModule: llm_mutate_tree, llm_crossover_trees, llm_randomize_tree
using ..LoggingModule: log_generation!
using ..TrackedPopMemberModule: TrackedPopMember
using ..ParseModule: render_expr, parse_expr

struct LLMMutationResult{N<:AbstractExpression,P<:AbstractPopMember} <:
       AbstractMutationResult{N,P}
    tree::Union{N,Nothing}
    member::Union{P,Nothing}
    num_evals::Float64
    return_immediately::Bool
    using_llm::Bool

    # Explicit constructor with keyword arguments
    function LLMMutationResult{_N,_P}(;
        tree::Union{_N,Nothing}=nothing,
        member::Union{_P,Nothing}=nothing,
        num_evals::Float64=0.0,
        return_immediately::Bool=false,
        using_llm::Bool=false,
    ) where {_N<:AbstractExpression,_P<:AbstractPopMember}
        @assert(
            (tree === nothing) ⊻ (member === nothing),
            "Mutation result must return either a tree or a pop member, not both"
        )
        return new{_N,_P}(tree, member, num_evals, return_immediately, using_llm)
    end
end

function check_constant(tree::AbstractExpressionNode)::Bool
    return (tree.degree == 0) && tree.constant
end

function check_constant(tree::AbstractExpression)::Bool
    return check_constant(get_tree(tree))
end

@unstable function next_generation(
    dataset::D,
    member::P,
    temperature,
    curmaxsize::Int,
    running_search_statistics::SymbolicRegression.RunningSearchStatistics,
    options::LaSROptions;
    tmp_recorder::SymbolicRegression.RecordType,
)::Tuple{
    P,Bool,Float64
} where {T,L,D<:Dataset{T,L},N<:AbstractExpression{T},P<:AbstractPopMember{T,L,N}}
    parent_ref = member.ref
    num_evals = 0.0

    #TODO - reconsider this
    before_cost, before_loss = member.cost, member.loss

    nfeatures = max_features(dataset, options)

    weights = copy(options.mutation_weights)

    condition_mutation_weights!(weights, member, options, curmaxsize)

    mutation_choice = sample_mutation(weights)

    successful_mutation = false
    attempts = 0
    max_attempts = 10
    node_storage = allocate_container(member.tree)

    #############################################
    # Mutations
    #############################################
    # local tree
    old_contribution = [
        member.llm_contribution, member.sr_contribution, member.total_contribution
    ]
    new_contribution = deepcopy(old_contribution)
    rtree = Ref{N}()
    while (!successful_mutation) && attempts < max_attempts
        rtree[] = copy_into!(node_storage, member.tree)

        mutation_result = _dispatch_mutations!(
            rtree[],
            member.pm,
            mutation_choice,
            options.mutation_weights,
            options;
            recorder=tmp_recorder,
            temperature,
            dataset,
            cost=before_cost,
            loss=before_loss,
            parent_ref,
            curmaxsize,
            nfeatures,
        )
        if options.tracking && !isnothing(mutation_result.member)
            # If the mutation result is a PopMember, we need to convert it to a TrackedPopMember
            # MR = MutationResult{N,TrackedPopMember{T,L,N}} 
            wrapped_member = TrackedPopMember(mutation_result.member, old_contribution...)
        else
            wrapped_member = mutation_result.member
        end
        mutation_result = if mutation_result isa LLMMutationResult{N,P}
            LLMMutationResult{N,TrackedPopMember{T,L,N}}(;
                tree=mutation_result.tree,
                member=wrapped_member,
                num_evals=mutation_result.num_evals,
                return_immediately=mutation_result.return_immediately,
                using_llm=true,
            )
        else
            MutationResult{N,TrackedPopMember{T,L,N}}(;
                tree=mutation_result.tree,
                member=wrapped_member,
                num_evals=mutation_result.num_evals,
                return_immediately=mutation_result.return_immediately,
            )
        end
        mutation_result::AbstractMutationResult{N,P}
        num_evals += mutation_result.num_evals::Float64

        if mutation_result.return_immediately
            @assert(
                mutation_result.member isa P,
                "Mutation result must return a `PopMember` if `return_immediately` is true"
            )
            if options.tracking
                if mutation_result isa LLMMutationResult{N,P}
                    # increment the llm contribution
                    mutation_result.member.llm_contribution += 1
                else
                    mutation_result.member.sr_contribution += 1
                end
                mutation_result.member.total_contribution += 1
            end

            return mutation_result.member::P, true, num_evals
        else
            @assert(
                mutation_result.tree isa N,
                "Mutation result must return a tree if `return_immediately` is false"
            )
            rtree[] = mutation_result.tree::N
            successful_mutation = check_constraints(rtree[], options, curmaxsize)
            attempts += 1
            if options.tracking
                new_contribution = deepcopy(old_contribution)
                if mutation_result isa LLMMutationResult{N,P}
                    # increment the llm contribution
                    new_contribution[1] += 1
                else
                    # increment the sr contribution
                    new_contribution[2] += 1
                end
                new_contribution[3] += 1
            end
        end
    end

    tree = rtree[]

    if !successful_mutation
        @recorder begin
            tmp_recorder["result"] = "reject"
            tmp_recorder["reason"] = "failed_constraint_check"
        end
        mutation_accepted = false
        ret = if options.tracking
            TrackedPopMember(
                PopMember(
                    copy_into!(node_storage, member.tree),
                    before_cost,
                    before_loss,
                    options,
                    compute_complexity(member, options);
                    parent=parent_ref,
                    deterministic=options.deterministic,
                ),
                old_contribution...,
            )
        else
            PopMember(
                copy_into!(node_storage, member.tree),
                before_cost,
                before_loss,
                options,
                compute_complexity(member, options);
                parent=parent_ref,
                deterministic=options.deterministic,
            )
        end
        return (ret, mutation_accepted, num_evals)
    end

    after_cost, after_loss = eval_cost(dataset, tree, options)
    num_evals += dataset_fraction(dataset)

    if isnan(after_cost)
        @recorder begin
            tmp_recorder["result"] = "reject"
            tmp_recorder["reason"] = "nan_loss"
        end
        mutation_accepted = false
        ret = if options.tracking
            TrackedPopMember(
                PopMember(
                    copy_into!(node_storage, member.tree),
                    before_cost,
                    before_loss,
                    options,
                    compute_complexity(member, options);
                    parent=parent_ref,
                    deterministic=options.deterministic,
                ),
                old_contribution...,
            )
        else
            PopMember(
                copy_into!(node_storage, member.tree),
                before_cost,
                before_loss,
                options,
                compute_complexity(member, options);
                parent=parent_ref,
                deterministic=options.deterministic,
            )
        end
        return (ret, mutation_accepted, num_evals)
    end

    probChange = 1.0
    if options.annealing
        delta = after_cost - before_cost
        probChange *= exp(-delta / (temperature * options.alpha))
    end
    newSize = -1
    if options.use_frequency
        oldSize = compute_complexity(member, options)
        newSize = compute_complexity(tree, options)
        old_frequency = if (0 < oldSize <= options.maxsize)
            running_search_statistics.normalized_frequencies[oldSize]
        else
            1e-6
        end
        new_frequency = if (0 < newSize <= options.maxsize)
            running_search_statistics.normalized_frequencies[newSize]
        else
            1e-6
        end
        probChange *= old_frequency / new_frequency
    end

    if probChange < rand()
        @recorder begin
            tmp_recorder["result"] = "reject"
            tmp_recorder["reason"] = "annealing_or_frequency"
        end
        mutation_accepted = false
        ret = if options.tracking
            TrackedPopMember(
                PopMember(
                    copy_into!(node_storage, member.tree),
                    before_cost,
                    before_loss,
                    options,
                    compute_complexity(member, options);
                    parent=parent_ref,
                    deterministic=options.deterministic,
                ),
                old_contribution...,
            )
        else
            PopMember(
                copy_into!(node_storage, member.tree),
                before_cost,
                before_loss,
                options,
                compute_complexity(member, options);
                parent=parent_ref,
                deterministic=options.deterministic,
            )
        end
        return (ret, mutation_accepted, num_evals)
    else
        @recorder begin
            tmp_recorder["result"] = "accept"
            tmp_recorder["reason"] = "pass"
        end
        mutation_accepted = true
        ret = if options.tracking
            TrackedPopMember(
                PopMember(
                    tree,
                    after_cost,
                    after_loss,
                    options,
                    newSize;
                    parent=parent_ref,
                    deterministic=options.deterministic,
                ),
                new_contribution...,
            )
        else
            PopMember(
                tree,
                after_cost,
                after_loss,
                options,
                newSize;
                parent=parent_ref,
                deterministic=options.deterministic,
            )
        end

        return (ret, mutation_accepted, num_evals)
    end
end

function mutate!(
    tree::N,
    member::P,
    ::Val{:llm_mutate},
    ::SymbolicRegression.AbstractMutationWeights,
    options::SymbolicRegression.AbstractOptions;
    recorder::SymbolicRegression.RecordType,
    kws...,
) where {
    T,N<:SymbolicRegression.AbstractExpression{T},P<:SymbolicRegression.AbstractPopMember
}
    tree = llm_mutate_tree(tree, options)
    @recorder recorder["type"] = "llm_mutate"
    return LLMMutationResult{N,P}(; tree=tree)
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
) where {
    T,N<:SymbolicRegression.AbstractExpression{T},P<:SymbolicRegression.AbstractPopMember
}
    tree = llm_randomize_tree(tree, curmaxsize, options, nfeatures)
    @recorder recorder["type"] = "llm_randomize"
    return LLMMutationResult{N,P}(; tree=tree)
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
} where {
    T,L,D<:SymbolicRegression.Dataset{T,L},N,P<:SymbolicRegression.AbstractPopMember{T,L,N}
}
    llm_skip = false

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

    contribution = [
        [member1.llm_contribution, member1.sr_contribution, member1.total_contribution],
        [member2.llm_contribution, member2.sr_contribution, member2.total_contribution],
    ]

    if !llm_skip
        # Fall back to the default SR approach
        m1 = if member1 isa TrackedPopMember
            member1.pm
        else
            member1
        end
        m2 = if member2 isa TrackedPopMember
            member2.pm
        else
            member2
        end
        new_member1, new_member2, crossover_accepted, num_evals = crossover_generation(
            m1, m2, dataset, curmaxsize, options.sr_options; recorder=recorder
        )
    else
        # If we used the LLM for crossover, we need to evaluate the cost of the new trees
        after_cost1, after_loss1 = eval_cost(dataset, child_tree1, options)
        after_cost2, after_loss2 = eval_cost(dataset, child_tree2, options)

        num_evals = dataset_fraction(dataset) * 2

        new_member1 = PopMember(
            child_tree1,
            after_cost1,
            after_loss1,
            options,
            afterSize1;
            parent=member1.ref,
            deterministic=options.deterministic,
        )
        new_member2 = PopMember(
            child_tree2,
            after_cost2,
            after_loss2,
            options,
            afterSize2;
            parent=member2.ref,
            deterministic=options.deterministic,
        )
        crossover_accepted = true
        contribution = [
            [
                member1.llm_contribution + 1,
                member1.sr_contribution,
                member1.total_contribution + 1,
            ],
            [
                member2.llm_contribution + 1,
                member2.sr_contribution,
                member2.total_contribution + 1,
            ],
        ]
    end

    if options.tracking
        new_member1 = TrackedPopMember(new_member1, contribution[1]...)
        new_member2 = TrackedPopMember(new_member2, contribution[2]...)
    end

    return new_member1, new_member2, crossover_accepted, num_evals
end

end
