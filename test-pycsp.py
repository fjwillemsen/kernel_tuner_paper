from pycsp3 import *


def prnt(var: Var):
    print(var)
    print(var.id)
    print(var.dom)
    print()


array = [1, 2, 4, 8, 16]
x = Var(range(5), id="x")
y = Var("hello", "world", id="y")
z = Var(array, id="z")

prnt(x)
prnt(y)

satisfy(x > 0, z > 0, AllDifferent(x, z))

# # solve for a single configuration
# if solve() is SAT:
#     print(values(x))

# solve for all configurations to get the feasible region
if solve(sols=ALL) is SAT:
    num_solutions: int = n_solutions()  # number of solutions
    print(f"{num_solutions=}")
    solutions = list(
        values(x, sol=i) for i in range(num_solutions)
    )  # list of solutions
    print(f"{solutions=}")
    for solution in solutions:
        print(x[solution])
