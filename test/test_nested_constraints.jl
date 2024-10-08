println("Test operator nesting and flagging.")
using LibraryAugmentedSymbolicRegression

function create_options(nested_constraints)
    return Options(;
        binary_operators=(+, *, /, -),
        unary_operators=(cos, exp),
        nested_constraints=nested_constraints,
    )
end

options = create_options(nothing)
# Count max nests:
tree = cos(exp(exp(exp(exp(Node("x1"))))))
degree_of_exp = 1
index_of_exp = findfirst(isequal(exp), options.operators.unaops)
@test 4 == LibraryAugmentedSymbolicRegression.CheckConstraintsModule.count_max_nestedness(
    tree, degree_of_exp, index_of_exp
)

tree = cos(exp(Node("x1")) + exp(exp(exp(exp(Node("x1"))))))
@test 4 == LibraryAugmentedSymbolicRegression.CheckConstraintsModule.count_max_nestedness(
    tree, degree_of_exp, index_of_exp
)

degree_of_plus = 2
index_of_plus = findfirst(isequal(+), options.operators.binops)
tree = cos(exp(Node("x1")) + exp(exp(Node("x1") + exp(exp(exp(Node("x1")))))))
@test 2 == LibraryAugmentedSymbolicRegression.CheckConstraintsModule.count_max_nestedness(
    tree, degree_of_plus, index_of_plus
)

# Test checking for illegal nests:
x1 = Node("x1")
options = create_options(nothing)
tree = cos(cos(x1)) + cos(x1) + exp(cos(x1))
@test !LibraryAugmentedSymbolicRegression.CheckConstraintsModule.flag_illegal_nests(
    tree, options
)

options = create_options([cos => [cos => 0]])
@test LibraryAugmentedSymbolicRegression.CheckConstraintsModule.flag_illegal_nests(
    tree, options
)

options = create_options([cos => [cos => 1]])
@test !LibraryAugmentedSymbolicRegression.CheckConstraintsModule.flag_illegal_nests(
    tree, options
)

options = create_options([cos => [exp => 0]])
@test !LibraryAugmentedSymbolicRegression.CheckConstraintsModule.flag_illegal_nests(
    tree, options
)

options = create_options([exp => [cos => 0]])
@test LibraryAugmentedSymbolicRegression.CheckConstraintsModule.flag_illegal_nests(
    tree, options
)

options = create_options([(+) => [(+) => 0]])
@test LibraryAugmentedSymbolicRegression.CheckConstraintsModule.flag_illegal_nests(
    tree, options
)

println("Passed.")
