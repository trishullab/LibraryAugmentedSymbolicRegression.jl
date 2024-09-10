# activate MyProject
using Pkg
Pkg.activate("lasr")
Pkg.instantiate()
# Pkg.develop(path="PromptingTools.jl")
# Pkg.develop(path="SymbolicRegression.jl")
# Pkg.add("Plots")
# Pkg.add("DataFrames")
# Pkg.add("CSV")
# Pkg.add("Random")
# Pkg.add("StatsBase")
# Pkg.add("LoopVectorization")
# Pkg.add("Bumper")
# Pkg.instantiate()
using SymbolicRegression
using Plots: Plots as plt
using DataFrames
using CSV: CSV
using Random: MersenneTwister
using StatsBase: sample
using LoopVectorization
using Bumper

df = let df = CSV.read("data/bigbench_records.csv", DataFrame)
    df[!, [:task, :subtask, :score_fn, :model, :n_shots, :n_params, :norm_score, :score]]
end

# names(df): "task", "subtask", "score_fn", "model", "n_shots", "n_params", "score", "norm_score"
# ^ Only want to model n_shots, n_params, score
# unique_score_fn = unique(df.score_fn)


# Assign each group an id
grouped_df = groupby(df, [:task, :subtask, :score_fn, :model])
num_groups = 50
df.group = groupindices(grouped_df)
groups_to_train_on =
    Vector{Int}(sample(MersenneTwister(0), unique(df.group), num_groups; replace = false))
df2 = filter(row -> row.group ∈ groups_to_train_on, df)

# Re-index the classes so they count from 1 up
df3 = let df3 = deepcopy(df2), old_to_new_group_map = Dict(unique(df2.group) .=> 1:length(unique(df2.group)))
    df3.group = map(g -> old_to_new_group_map[g], df2.group)
    df3
end


X = copy(Matrix{Float32}(df3[!, [:n_shots, :n_params]])');
classes = Vector{Int}(df3[!, :group]);
y = Vector{Float32}(df3[!, :score]);
dataset = Dataset(X, y, extra = (; classes), variable_names = ["n_shots", "n_params"]);
my_loss(prediction, target) = abs((prediction - target))
p=0.01

llm_recorder_dir = "bigbench_logs/"
prompts_dir = "prompts/"

# Create directories if they don't exist
isdir(llm_recorder_dir) || mkdir(llm_recorder_dir)
isdir(prompts_dir) || mkdir(prompts_dir)


options = Options(
    populations=256,
    ncycles_per_iteration=1000,
    binary_operators = (+, *, -, /, ^),
    unary_operators = (cos, log),
    nested_constraints = [(^) => [(^) => 0, cos => 0, log => 0], (/) => [(/) => 1], (cos) => [cos => 0, log => 0], log => [log => 0, cos => 0, (^) => 0]],
    constraints = [(^) => (3, 1), log => 5, cos => 7],
    node_type = ParametricNode,
    expression_type = ParametricExpression,
    # elementwise_loss="L1DistLoss()",
    # elementwise_loss=myloss(predicted, target) = abs(predicted - target),
    expression_options = (; max_parameters = 3),
    elementwise_loss = my_loss,
    optimizer_nrestarts = 0,
    optimizer_iterations = 2,
    mutation_weights=MutationWeights(optimize=0.01),
    llm_options=LLMOptions(;
        active=true,
        prompt_evol=true,
        prompt_concepts=true,
        weights=LLMWeights(
            llm_mutate=p,
            llm_crossover=p,
            llm_gen_random=p
        ),
        num_pareto_context=5,
        api_key="token-abc123",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        api_kwargs = Dict(
            "max_tokens" => 1024,
            "url" => "https://avior.mlfoundry.com/vllm/v1",
            # "url" => "http://localhost:11440/v1",
            "temperature" => 0.0, # 0.0 is greedy sampling. 
            # Change this if low performance.
            # It might perform better with higher temperature.
        ),
        http_kwargs = Dict(
            "retries" => 5,
            "readtimeout" => 360, # Wait 6 minutes for a response.
        ),
        llm_recorder_dir,
        prompts_dir,
        idea_threshold = 30,
        var_order=Dict(
            "x0" => "n_shots",
            "x1" => "n_params",
        ),
        is_parametric=true,
    ),
    maxsize=30,
    turbo=true,
    bumper=true,
    # batching = true,
    # batch_size = 1000,
);
# hof = equation_search(dataset; options, niterations = 1000);
hof = equation_search(dataset; options, niterations = 1000, parallelism=:multiprocessing, numprocs=Sys.CPU_THREADS);