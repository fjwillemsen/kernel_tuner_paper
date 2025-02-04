# construct the searchspace of https://github.com/CNugteren/CLTune/blob/8a56a4a314be7ccef56ad8f55e8a34a37dda0545/samples/gemm/gemm.cc

import constraint

problem = constraint.Problem()

M = 20
N = 1
K = 576
problem.addVariable("WGD", [8, 16, 32, 64, 128]);
problem.addVariable("MDIMCD", [8, 16, 32]);
problem.addVariable("NDIMCD", [8, 16, 32]);
problem.addVariable("MDIMAD", [8, 16, 32]);
problem.addVariable("NDIMBD", [8, 16, 32]);
problem.addVariable("KWID", [2, 8, 16]);
problem.addVariable("VWMD", [1, 2, 4, 8]);
problem.addVariable("VWND", [1, 2, 4, 8]);
problem.addVariable("PADA", [0, 1]);
problem.addVariable("PADB", [0, 1]);

# // Helper function to determine whether or not 'a' is a multiple of 'b'
def IsMultiple(a, b) -> bool:
  return True if ((a/b)*b == a) else False

def MultipleOfX(a, x) -> bool:
    return IsMultiple(a, x)

def MultipleOfXMulY(a, x, y) -> bool:
    return IsMultiple(a, x*y)

def MultipleOfXMulYDivZ(a, x, y, z) -> bool:
    return IsMultiple(a, (x*y)/z)

# def LocalMemorySize(a, b, c, d, e, f, g, h) -> bool:
#     size_of_float = 4   # given 32 bit precision
#     return (((a * b * c / d) + (e * f * g / h)) * size_of_float);

# // Requirement for unrolling the WGD loop
problem.addConstraint(MultipleOfX, ["WGD", "KWID"]);
# // Required for integer MWID and NWID
problem.addConstraint(MultipleOfXMulY, ["WGD", "MDIMCD", "VWMD"]);
problem.addConstraint(MultipleOfXMulY, ["WGD", "NDIMCD", "VWND"]);
# // Required for integer MWIAD and NWIBD
problem.addConstraint(MultipleOfXMulY, ["WGD", "MDIMAD", "VWMD"]);
problem.addConstraint(MultipleOfXMulY, ["WGD", "NDIMBD", "VWND"]);
# // WGD has to be a multiple of KDIMAD = ((MDIMCD*NDIMCD)/(MDIMAD)) and KDIMBD = (...)
problem.addConstraint(MultipleOfXMulYDivZ, ["WGD", "MDIMCD", "NDIMCD", "MDIMAD"]);
problem.addConstraint(MultipleOfXMulYDivZ, ["WGD", "MDIMCD", "NDIMCD", "NDIMBD"]);

# problem.addConstraint(LocalMemorySize, ["SA", "KWG", "MWG", "VWM", "SB", "KWG", "NWG", "VWN"]);


if __name__ == "__main__":
    solutions = problem.getSolutions()
    print(len(solutions))
    # print(solutions)
