module LibraryAugmentedSymbolicRegression

# Types
export Population,
    PopMember,
    HallOfFame,
    Options,
    Dataset,
    MutationWeights,
    Node,
    LaSRRegressor,
    MultitargetLaSRRegressor,
    # Options:
    LLMOperationWeights,
    LLMOptions,
    LaSROptions,
    LaSRMutationWeights,
    # Functions:
    llm_randomize_tree,
    llm_crossover_trees,
    llm_mutate_tree,
    crossover_trees,
    concept_evolution,
    generate_concepts,
    mutate!,
    crossover_generation,
    # Utilities:
    LaSRLogger,
    render_expr,
    parse_expr,
    parse_msg_content,
    construct_prompt,
    load_prompt,
    # LLM Server options
    LLAMAFILE_MODEL,
    LLAMAFILE_PATH,
    LLAMAFILE_URL,
    LLM_PORT

using Distributed
using PackageExtensionCompat: @require_extensions
using Pkg: Pkg
using TOML: parsefile
using Reexport

# https://discourse.julialang.org/t/how-to-find-out-the-version-of-a-package-from-its-module/37755/15
const PACKAGE_VERSION = try
    root = pkgdir(@__MODULE__)
    if root == String
        let project = parsefile(joinpath(root, "Project.toml"))
            VersionNumber(project["version"])
        end
    else
        VersionNumber(0, 0, 0)
    end
catch
    VersionNumber(0, 0, 0)
end

using DispatchDoctor: @stable
@reexport using SymbolicRegression
using .SymbolicRegression: @recorder, @sr_spawner

import SymbolicRegression: _main_search_loop!

@stable default_mode = "disable" begin
    include("Utils.jl")
    include("Parse.jl")
    include("LLMServe.jl")
    include("MutationWeights.jl")
    include("Logging.jl")
    include("LLMOptionsStruct.jl")
    include("LLMOptions.jl")
    include("Core.jl")
    include("LLMUtils.jl")
    include("LLMFunctions.jl")
    include("Mutate.jl")
end

using .CoreModule:
    LLMOperationWeights, LLMOptions, LaSROptions, LaSRMutationWeights, LaSRLogger,
    async_run_llm_server,
    LLAMAFILE_MODEL,
    LLAMAFILE_PATH,
    LLAMAFILE_URL,
    LLM_PORT

using .UtilsModule: is_anonymous_function, recursive_merge, json3_write, @ignore
using .LLMFunctionsModule:
    llm_randomize_tree,
    llm_mutate_tree,
    crossover_trees,
    llm_crossover_trees,
    concept_evolution,
    parse_msg_content,
    generate_concepts
using .LLMUtilsModule: load_prompt, construct_prompt
using .ParseModule: render_expr, parse_expr
import .MutateModule: mutate!, crossover_generation

using .LoggingModule: log_generation!
using UUIDs: uuid1

"""
@TODO: Modularize _main_search_loop! function so that I don't have to change the
entire function to accomodate prompt evolution.
"""
function _main_search_loop!(
    state::SymbolicRegression.AbstractSearchState{T,L,N},
    datasets,
    ropt::SymbolicRegression.AbstractRuntimeOptions,
    options::LaSROptions,
) where {T,L,N}
    ropt.verbosity > 0 && @info "Started!"
    if !isnothing(ropt.logger)
        options.lasr_logger = LaSRLogger(ropt.logger)
    end
    nout = length(datasets)
    start_time = time()
    progress_bar = if ropt.progress
        #TODO: need to iterate this on the max cycles remaining!
        sum_cycle_remaining = sum(state.cycles_remaining)
        SymbolicRegression.WrappedProgressBar(
            sum_cycle_remaining, ropt.niterations; barlen=options.terminal_width
        )
    else
        nothing
    end

    last_print_time = time()
    last_speed_recording_time = time()
    num_evals_last = sum(sum, state.num_evals)
    num_evals_since_last = sum(sum, state.num_evals) - num_evals_last  # i.e., start at 0
    print_every_n_seconds = 5
    equation_speed = Float32[]

    if ropt.parallelism in (:multiprocessing, :multithreading)
        for j in 1:nout, i in 1:(options.populations)
            # Start listening for each population to finish:
            t = @async put!(state.channels[j][i], fetch(state.worker_output[j][i]))
            push!(state.tasks[j], t)
        end
    end
    kappa = 0
    resource_monitor = SymbolicRegression.ResourceMonitor(;
        # Storing n times as many monitoring intervals as populations seems like it will
        # help get accurate resource estimates:
        max_recordings=options.populations * 100 * nout,
        start_reporting_at=options.populations * 3 * nout,
        window_size=options.populations * 2 * nout,
    )
    n_iterations = 0

    # Replaces: llm_recorder(options.llm_options, string(div(n_iterations, options.populations)), "n_iterations")
    begin
        gen_id = uuid1()
        log_generation!(
            options.lasr_logger;
            id=gen_id,
            mode="n_iterations",
            chosen=string(div(n_iterations, options.populations)),
        )
    end

    worst_members = Vector{PopMember}()
    while sum(state.cycles_remaining) > 0
        kappa += 1
        if kappa > options.populations * nout
            kappa = 1
        end
        # nout, populations:
        j, i = state.task_order[kappa]

        # Check if error on population:
        if ropt.parallelism in (:multiprocessing, :multithreading)
            if istaskfailed(state.tasks[j][i])
                fetch(state.tasks[j][i])
                error("Task failed for population")
            end
        end
        # Non-blocking check if a population is ready:
        population_ready = if ropt.parallelism in (:multiprocessing, :multithreading)
            # TODO: Implement type assertions based on parallelism.
            isready(state.channels[j][i])
        else
            true
        end
        SymbolicRegression.record_channel_state!(resource_monitor, population_ready)

        # Don't start more if this output has finished its cycles:
        # TODO - this might skip extra cycles?
        population_ready &= (state.cycles_remaining[j] > 0)
        if population_ready
            if n_iterations % options.populations == 0
                worst_members = Vector{PopMember}()
            end
            n_iterations += 1

            # Take the fetch operation from the channel since it's ready
            (cur_pop, best_seen, cur_record, cur_num_evals) = if ropt.parallelism in
                (
                :multiprocessing, :multithreading
            )
                take!(
                    state.channels[j][i]
                )
            else
                state.worker_output[j][i]
            end::SymbolicRegression.DefaultWorkerOutputType{
                Population{T,L,N},HallOfFame{T,L,N}
            }
            state.last_pops[j][i] = copy(cur_pop)
            state.best_sub_pops[j][i] = SymbolicRegression.best_sub_pop(
                cur_pop; topn=options.topn
            )
            @recorder state.record[] = SymbolicRegression.recursive_merge(
                state.record[], cur_record
            )
            state.num_evals[j][i] += cur_num_evals
            dataset = datasets[j]
            cur_maxsize = state.cur_maxsizes[j]

            for member in cur_pop.members
                size = SymbolicRegression.compute_complexity(member, options)
                SymbolicRegression.update_frequencies!(
                    state.all_running_search_statistics[j]; size
                )
            end
            #! format: off
            SymbolicRegression.update_hall_of_fame!(state.halls_of_fame[j], cur_pop.members, options)
            SymbolicRegression.update_hall_of_fame!(state.halls_of_fame[j], best_seen.members[best_seen.exists], options)
            #! format: on

            # Dominating pareto curve - must be better than all simpler equations
            dominating = SymbolicRegression.calculate_pareto_frontier(
                state.halls_of_fame[j]
            )

            worst_member = nothing
            for member in cur_pop.members
                if worst_member === nothing || member.loss > worst_member.loss
                    worst_member = member
                end
            end

            if worst_member !== nothing && worst_member.loss > dominating[end].loss
                push!(worst_members, worst_member)
            end

            if options.use_llm &&
                options.use_concept_evolution &&
                (n_iterations % options.populations == 0)
                generate_concepts(dominating, worst_members, options)
            end

            if options.save_to_file
                SymbolicRegression.save_to_file(dominating, nout, j, dataset, options, ropt)
            end

            ##################################################
            # Migration
            ##################################################
            if options.migration
                best_of_each = SymbolicRegression.Population([
                    member for pop in state.best_sub_pops[j] for member in pop.members
                ])
                SymbolicRegression.migrate!(
                    best_of_each.members => cur_pop, options; frac=options.fraction_replaced
                )
            end
            if options.hof_migration && length(dominating) > 0
                SymbolicRegression.migrate!(
                    dominating => cur_pop, options; frac=options.fraction_replaced_hof
                )
            end
            ##################################################

            state.cycles_remaining[j] -= 1
            if state.cycles_remaining[j] == 0
                break
            end
            worker_idx = SymbolicRegression.assign_next_worker!(
                state.worker_assignment;
                out=j,
                pop=i,
                parallelism=ropt.parallelism,
                state.procs,
            )
            iteration = if options.use_recorder
                key = "out$(j)_pop$(i)"
                SymbolicRegression.find_iteration_from_record(key, state.record[]) + 1
            else
                0
            end

            c_rss = deepcopy(state.all_running_search_statistics[j])
            in_pop = copy(cur_pop::SymbolicRegression.Population{T,L,N})
            state.worker_output[j][i] = @sr_spawner(
                begin
                    SymbolicRegression._dispatch_s_r_cycle(
                        in_pop,
                        dataset,
                        options;
                        pop=i,
                        out=j,
                        iteration,
                        ropt.verbosity,
                        cur_maxsize,
                        running_search_statistics=c_rss,
                    )
                end,
                parallelism = ropt.parallelism,
                worker_idx = worker_idx
            )
            if ropt.parallelism in (:multiprocessing, :multithreading)
                state.tasks[j][i] = @async put!(
                    state.channels[j][i], fetch(state.worker_output[j][i])
                )
            end

            total_cycles = ropt.niterations * options.populations
            state.cur_maxsizes[j] = SymbolicRegression.get_cur_maxsize(;
                options, total_cycles, cycles_remaining=state.cycles_remaining[j]
            )
            SymbolicRegression.move_window!(state.all_running_search_statistics[j])
            if !isnothing(progress_bar)
                head_node_occupation = SymbolicRegression.estimate_work_fraction(
                    resource_monitor
                )
                SymbolicRegression.update_progress_bar!(
                    progress_bar,
                    only(state.halls_of_fame),
                    only(datasets),
                    options,
                    equation_speed,
                    head_node_occupation,
                    ropt.parallelism,
                )
            end
            if ropt.logger !== nothing
                SymbolicRegression.logging_callback!(
                    ropt.logger; state, datasets, ropt, options
                )
            end
        end
        yield()

        ###############################################################
        ## Search statistics
        elapsed_since_speed_recording = time() - last_speed_recording_time
        if elapsed_since_speed_recording > 1.0
            num_evals_since_last, num_evals_last = let s = sum(sum, state.num_evals)
                s - num_evals_last, s
            end
            current_speed = num_evals_since_last / elapsed_since_speed_recording
            push!(equation_speed, current_speed)
            average_over_m_measurements = 20 # 20 second running average
            if length(equation_speed) > average_over_m_measurements
                deleteat!(equation_speed, 1)
            end
            last_speed_recording_time = time()
        end
        ###############################################################

        ###############################################################
        ## Printing code
        elapsed = time() - last_print_time
        # Update if time has passed
        if elapsed > print_every_n_seconds
            if ropt.verbosity > 0 && !ropt.progress && length(equation_speed) > 0

                # Dominating pareto curve - must be better than all simpler equations
                head_node_occupation = SymbolicRegression.estimate_work_fraction(
                    resource_monitor
                )
                total_cycles = ropt.niterations * options.populations
                SymbolicRegression.print_search_state(
                    state.halls_of_fame,
                    datasets;
                    options,
                    equation_speed,
                    total_cycles,
                    state.cycles_remaining,
                    head_node_occupation,
                    parallelism=ropt.parallelism,
                    width=options.terminal_width,
                )
            end
            last_print_time = time()
        end
        ###############################################################

        ###############################################################
        ## Early stopping code
        if any((
            SymbolicRegression.check_for_loss_threshold(state.halls_of_fame, options),
            SymbolicRegression.check_for_user_quit(state.stdin_reader),
            SymbolicRegression.check_for_timeout(start_time, options),
            SymbolicRegression.check_max_evals(state.num_evals, options),
        ))
            break
        end
        ###############################################################
    end
    if !isnothing(progress_bar)
        SymbolicRegression.finish!(progress_bar)
    end
    return nothing
end

include("MLJInterface.jl")
using .MLJInterfaceModule:
    LaSRRegressor, LaSRTestRegressor, MultitargetLaSRRegressor, MultitargetLaSRTestRegressor

function __init__()
    @require_extensions

    should_start_llamafile =
        get(ENV, "START_LLAMASERVER", "false") == "true" ||
        # if testing_mode is online_llamafile
        get(ENV, "SYMBOLIC_REGRESSION_TEST_SUITE", "online,online_llamafile,offline") == "online_llamafile"
    if should_start_llamafile
        @info "Starting LLM server..."
        async_run_llm_server(LLAMAFILE_URL, LLAMAFILE_PATH, LLM_PORT)
    end
end

# Hack to get static analysis to work from within tests:
@ignore include("../test/runtests.jl")

# TODO: Hack to force ConstructionBase version
using ConstructionBase: ConstructionBase as _

include("precompile.jl")
redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        do_precompilation(Val(:precompile))
    end
end

end # module SR
