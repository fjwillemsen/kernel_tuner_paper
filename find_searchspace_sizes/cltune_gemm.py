# construct the searchspace of https://github.com/CNugteren/CLTune/blob/8a56a4a314be7ccef56ad8f55e8a34a37dda0545/samples/gemm/gemm.cc

import constraint

problem = constraint.Problem()

problem.addVariable("MWG", [16, 32, 64, 128]);
problem.addVariable("NWG", [16, 32, 64, 128]);
problem.addVariable("KWG", [16, 32]);
problem.addVariable("MDIMC", [8, 16, 32]);
problem.addVariable("NDIMC", [8, 16, 32]);
problem.addVariable("MDIMA", [8, 16, 32]);
problem.addVariable("NDIMB", [8, 16, 32]);
problem.addVariable("KWI", [2, 8]);
problem.addVariable("VWM", [1, 2, 4, 8]);
problem.addVariable("VWN", [1, 2, 4, 8]);
problem.addVariable("STRM", [0, 1]);
problem.addVariable("STRN", [0, 1]);
problem.addVariable("SA", [0, 1]);
problem.addVariable("SB", [0, 1]);

# // Helper function to determine whether or not 'a' is a multiple of 'b'
def IsMultiple(a, b) -> bool:
  return True if ((a/b)*b == a) else False

def MultipleOfX(a, x) -> bool:
    return IsMultiple(a, x)

def MultipleOfXMulY(a, x, y) -> bool:
    return IsMultiple(a, x*y)

def MultipleOfXMulYDivZ(a, x, y, z) -> bool:
    return IsMultiple(a, (x*y)/z)

def LocalMemorySize(a, b, c, d, e, f, g, h) -> bool:
    size_of_float = 4   # given 32 bit precision
    return (((a * b * c / d) + (e * f * g / h)) * size_of_float);

# // Sets constraints: Requirement for unrolling the KWG loop
problem.addConstraint(MultipleOfX, ["KWG", "KWI"]);

#   // Sets constraintequired for integer MWI and NWI
problem.addConstraint(MultipleOfXMulY, ["MWG", "MDIMC", "VWM"]);
problem.addConstraint(MultipleOfXMulY, ["NWG", "NDIMC", "VWN"]);

#   // Sets constraintequired for integer MWIA and NWIB
problem.addConstraint(MultipleOfXMulY, ["MWG", "MDIMA", "VWM"]);
problem.addConstraint(MultipleOfXMulY, ["NWG", "NDIMB", "VWN"]);

# // Sets constraints: has to be a multiple of KDIMA = ((MDIMC*NDIMC)/(MDIMA)) and KDIMB = (...)
problem.addConstraint(MultipleOfXMulYDivZ, ["KWG", "MDIMC", "NDIMC", "MDIMA"]);
problem.addConstraint(MultipleOfXMulYDivZ, ["KWG", "MDIMC", "NDIMC", "NDIMB"]);

problem.addConstraint(LocalMemorySize, ["SA", "KWG", "MWG", "VWM", "SB", "KWG", "NWG", "VWN"]);


if __name__ == "__main__":
    solutions = problem.getSolutions()
    print(len(solutions))
    # print(solutions)
