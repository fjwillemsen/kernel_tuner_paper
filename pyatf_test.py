import numpy as np
from pyatf import TP, Interval, Set, Tuner
from pyatf.cost_functions.generic import CostFunction
from pyatf.search_techniques import Exhaustive

# input size
M = 8
N = 8
K = 8

# Step 1: Generate the Search Space
MWG = TP("MWG", Set(*np.arange(0.5, M).flatten()), lambda MWG: M % MWG == 0)
NWG = TP("NWG", Interval(1, N), lambda NWG: N % NWG == 0)
KWG = TP("KWG", Interval(1, K), lambda KWG: K % KWG == 0)

MDIMC = TP(
    "MDIMC",
    Interval(1, M),
    lambda MDIMC, MWG: MWG % MDIMC == 0,
)
NDIMC = TP(
    "NDIMC",
    Interval(1, N),
    lambda NDIMC, NWG, MDIMC: NWG % NDIMC == 0,
)
MDIMA = TP(
    "MDIMA",
    Interval(1, M),
    lambda MDIMA, MWG, NDIMC, KWG, MDIMC: MWG % MDIMA == 0
    and (MDIMC * NDIMC) % MDIMA == 0
    and KWG % ((MDIMC * NDIMC) / MDIMA) == 0,
)
NDIMB = TP(
    "NDIMB",
    Interval(1, N),
    lambda NDIMB, NWG, NDIMC, KWG, MDIMC: NWG % NDIMB == 0
    and (MDIMC * NDIMC) % NDIMB == 0
    and KWG % ((MDIMC * NDIMC) / NDIMB) == 0,
)

KWI = TP("KWI", Interval(1, K), lambda KWI, KWG: KWG % KWI == 0)

VWM = TP(
    "VWM",
    Set(1, 2, 4, 8),
    lambda VWM, MWG, MDIMC, MDIMA: (MWG / MDIMC) % VWM == 0
    and (MWG / MDIMA) % VWM == 0,
)
VWN = TP(
    "VWN",
    Set(1, 2, 4, 8),
    lambda VWN, NWG, NDIMC, NDIMB: (NWG / NDIMC) % VWN == 0
    and (NWG / NDIMB) % VWN == 0,
)

STRM = TP("STRM", Set(0, 1))
STRN = TP("STRN", Set(0, 1))

SA = TP("SA", Set(0, 1))
SB = TP(
    "SB",
    Set(0, 1),
)  # restriction of local memory

# Step 2: Implement a Cost Function
costfunc = CostFunction(":")  # bash no-op

# Step 3: Explore the Search Space
tuning_result = (
    Tuner()
    .tuning_parameters(
        MWG, NWG, KWG, MDIMC, NDIMC, MDIMA, NDIMB, KWI, VWM, VWN, STRM, STRN, SA, SB
    )
    .search_technique(Exhaustive())
    .tune(costfunc)
)
print(tuning_result)
