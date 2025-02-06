# construct the searchspace of https://github.com/HiPerCoRe/KTT/blob/master/Examples/Reduction/Reduction.cpp

import constraint

problem = constraint.Problem()

cus = 16 # assumes di.GetMaxComputeUnits() = 16 as for GTX 1070 used in paper (as per https://compubench.com/device.jsp?benchmark=compu20&D=NVIDIA+GeForce+GTX+1070&testgroup=info)
problem.addVariable("WORK_GROUP_SIZE_X", [32, 64, 96, 128, 160, 196, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024])   # assumes useDenseParameters && useWideParameters
problem.addVariable("WG_NUM", [0, cus, cus * 2, cus * 3, cus * 4, cus * 5, cus * 6, cus * 7, cus * 8, cus * 10, cus * 12, cus * 14, cus * 16, cus * 20, cus * 24, cus * 28, cus * 32, cus * 40, cus * 48, cus * 56, cus * 64])  # assumes useDenseParameters && useWideParameters
problem.addVariable("VECTOR_SIZE", [1, 2, 4, 8, 16])    # assumes computeApi == ktt::ComputeApi::OpenCL
problem.addVariable("UNBOUNDED_WG", [0, 1]);
problem.addVariable("USE_ATOMICS", [0, 1]);

problem.addConstraint(lambda a, b: (a == 1 and b == 0) or (a != 1 and b > 0), ["UNBOUNDED_WG", "WG_NUM"])
problem.addConstraint(lambda a, b: a == 1 or (a == 0 and b == 1), ["UNBOUNDED_WG", "USE_ATOMICS"])
problem.addConstraint(lambda a, b: a != 1 or b >= 32, ["UNBOUNDED_WG", "WORK_GROUP_SIZE_X"])

if __name__ == "__main__":
    solutions = problem.getSolutions()
    print(len(solutions))
    # print(solutions)
