# construct the searchspace of https://github.com/HiPerCoRe/KTT/blob/master/Examples/CoulombSum2d/CoulombSum2d.cpp

import constraint

problem = constraint.Problem()

# // Add several parameters to tuned kernel, some of them utilize constraint function and thread modifiers.
problem.addVariable("INNER_UNROLL_FACTOR", [0, 1, 2, 4, 8, 16, 32]);
problem.addVariable("USE_CONSTANT_MEMORY", [0, 1]);
problem.addVariable("VECTOR_TYPE", [1, 2, 4, 8]);
problem.addVariable("USE_SOA", [0, 1, 2]);

# // Using vectorized SoA only makes sense when vectors are longer than 1.
problem.addConstraint(lambda a, b: a > 1 or b != 2, ["VECTOR_TYPE", "USE_SOA"]);

# // Divide NDRange in dimension x by OUTER_UNROLL_FACTOR.
problem.addVariable("OUTER_UNROLL_FACTOR", [1, 2, 4, 8]);

# // Multiply work-group size in dimensions x and y by the following parameters (effectively setting work-group size to their values).
problem.addVariable("WORK_GROUP_SIZE_X", [4, 8, 16, 32]);
problem.addVariable("WORK_GROUP_SIZE_Y", [1, 2, 4, 8, 16, 32]);

if __name__ == "__main__":
    solutions = problem.getSolutions()
    print(len(solutions))
    # print(solutions)
