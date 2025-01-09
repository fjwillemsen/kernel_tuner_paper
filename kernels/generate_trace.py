import json
import os
import time
from pathlib import Path

import kernel_tuner
import numpy as np
from common import get_device_name, get_fallback, get_metrics
from kernel_tuner.observers.nvml import NVMLObserver


def ops(m, n, k):
    return (m * n * k * 2 + 2 * m * k)/1e9


def tune(inputs, optimization_algorithm: str, allotted_time_seconds = None, max_fevals = None, device=0, searchspace_set=2):
    path = os.path.dirname(os.path.realpath(__file__)) + "/gemm_cltune_opencl/"
    device_name = get_device_name(device)

    # kernel string
    kernel_string = ''
    files = ["common.opencl", "xgemm_part1.opencl", "xgemm_part2.opencl", "xgemm_part3.opencl", "xgemm_part4.opencl"]
    for f in files:
        with open(path + f, "r") as fp:
            kernel_string += fp.read()

    #n = np.int32(32)
    #m = np.int32(16)
    #k = np.int32(32)
    m, n, k = [np.int32(i) for i in inputs]

    #// Matrices are accessed as follows:
    #// A: [k*M + m], with 'k' ranging from 0:K and 'm' from 0:M (m,k,m)
    #// B: [k*N + n], with 'k' ranging from 0:K and 'n' from 0:N (n,k,n)
    #// C: [n*M + m], with 'n' ranging from 0:N and 'm' from 0:M (m,n,m)

    # input / output array intialization
    print("array initialization (may take a while)")
    A = np.array(np.random.randn(m, k), order='F').astype(np.float32)
    B = np.array(np.random.randn(k, n), order='F').astype(np.float32)
    #C = np.array(np.random.randn(m, n), order='F').astype(np.float32)
    C = np.zeros((m, n), order='F').astype(np.float32)
    alpha, beta = np.random.randn(2).astype(np.float32)
    alpha, beta = np.array([1.0, 1.0]).astype(np.float32)

    # tunable parameters
    print("setting tunable parameters")
    tune_params = dict()
    tune_params["nvml_gr_clock"] = [840]   # A4000: (base+boost)/2 = 1147, largest supported in range is 1140
    tune_params["nvml_mem_clock"] = [6501]  # A4000: nvidia-smi --query-supported-clocks=mem --format=csv

    if searchspace_set == 0:
        # original Kernel Tuner parameters
        tune_params["GEMMK"] = [0]
        tune_params["MWG"] = [16, 32, 64, 128]
        tune_params["NWG"] = [16, 32, 64, 128]
        tune_params["KWG"] = [32]
        tune_params["MDIMC"] = [8, 16, 32]
        tune_params["NDIMC"] = [8, 16, 32]
        tune_params["MDIMA"] = [8, 16, 32]
        tune_params["NDIMB"] = [8, 16, 32]
        tune_params["KWI"] = [2]
        tune_params["VWM"] = [1, 2, 4, 8]
        tune_params["VWN"] = [1, 2, 4, 8]
        tune_params["STRM"] = [0]
        tune_params["STRN"] = [0]
        tune_params["SA"] = [0, 1]
        tune_params["SB"] = [0, 1]
        tune_params["PRECISION"] = [32]
    elif searchspace_set == 1:
        # CLTune parameters, limited subset
        tune_params["GEMMK"] = [0]
        tune_params["MWG"] = [16, 32, 64]
        tune_params["NWG"] = [16, 32, 64]
        tune_params["KWG"] = [32]
        tune_params["MDIMC"] = [8, 16, 32]
        tune_params["NDIMC"] = [8, 16, 32]
        tune_params["MDIMA"] = [8, 16, 32]
        tune_params["NDIMB"] = [8, 16, 32]
        tune_params["KWI"] = [2]
        tune_params["VWM"] = [1, 2, 4]
        tune_params["VWN"] = [1, 2, 4]
        tune_params["STRM"] = [0]
        tune_params["STRN"] = [0]
        tune_params["SA"] = [0, 1]
        tune_params["SB"] = [0, 1]
        tune_params["KREG"] = [1]
        tune_params["PRECISION"] = [32]
    elif searchspace_set == 2 or searchspace_set == 3:
        tune_params["GEMMK"] = [0]
        tune_params["MWG"] = [16, 32, 64, 128]
        tune_params["NWG"] = [16, 32, 64, 128]
        tune_params["KWG"] = [16, 32]
        tune_params["MDIMC"] = [8, 16, 32]
        tune_params["NDIMC"] = [8, 16, 32]
        tune_params["MDIMA"] = [8, 16, 32]
        tune_params["NDIMB"] = [8, 16, 32]
        tune_params["KWI"] = [2]
        tune_params["VWM"] = [1, 2, 4, 8]
        tune_params["VWN"] = [1, 2, 4, 8]
        tune_params["STRM"] = [0, 1]
        tune_params["STRN"] = [0, 1]
        tune_params["SA"] = [0, 1]
        tune_params["SB"] = [0, 1]
        tune_params["KREG"] = [1]
        tune_params["PRECISION"] = [32]
    elif searchspace_set == 3:
        # for an even larger searchspace, precision can be tuned as well
        tune_params["PRECISION"] = [16, 32, 64, 3232, 6464]
    else:
        raise ValueError(f"Invalid {searchspace_set=}")

    # restrictions
    restrict = []
    restrict += ["KWG % KWI == 0"]
    restrict += ["MWG % (MDIMC * VWM) == 0"]
    restrict += ["NWG % (NDIMC * VWN) == 0"]
    restrict += ["MWG % (MDIMA * VWM) == 0"]
    restrict += ["NWG % (NDIMB * VWN) == 0"]
    restrict += ["KWG % ((MDIMC * NDIMC)/MDIMA) == 0"]
    restrict += ["KWG % ((MDIMC * NDIMC)/NDIMB) == 0"]
    restrict += ["not (MWG == 128 and NWG == 128 and MDIMC == 8 and NDIMC == 8)"]

    # observer for the frequencies and temperature
    nvmlobserver = NVMLObserver(
        [
            "core_freq",
            "mem_freq",
            "temperature",
            # "nvml_energy", 
        ],
        save_all=True,
        nvidia_smi_fallback=get_fallback(),
        use_locked_clocks=True
    )

    # additional arguments
    observers = [nvmlobserver]
    args = [m, n, k, alpha, beta, A, B, C, np.int32(0), np.int32(0)]
    problem_size = (m, n)
    grid_div_x = ["MWG"]
    grid_div_y = ["NWG"]
    block_size_names = ["MDIMC", "NDIMC", "block_size_z"]
    total_flops = ops(*inputs)
    metrics = get_metrics(total_flops)
    filename = f"outputdata/simulation_mode/gemm_cltune_opencl_iter=7_{device_name}_size-{m}x{n}x{k}"
    cachefile = Path(f"{filename}_cache.json")
    tracefile = Path(f"tracefile_{optimization_algorithm}.json")

    # set the optimization algorithm options
    strategy_options=dict()
    strategy_options['tracefile_path'] = str(tracefile.resolve())
    if max_fevals is not None:
        strategy_options['max_fevals'] = max_fevals
    if allotted_time_seconds is not None:
        strategy_options['time_limit'] = allotted_time_seconds
    if max_fevals is not None and allotted_time_seconds is not None:
        raise ValueError("Can't have 'max_fevals' and 'alloted_time_seconds' at the same time")

    # start tuning
    print(f"Starting tuning, {filename=}")
    results, env = kernel_tuner.tune_kernel("Xgemm", kernel_string, problem_size, args, tune_params, block_size_names=block_size_names,
                             lang="opencl", restrictions=restrict, verbose=False, compiler_options=["-I"+path],
                             grid_div_x=grid_div_x, grid_div_y=grid_div_y, observers=observers,
                             device=device, platform=0, iterations=7, metrics=metrics,
                             cache=str(cachefile), simulation_mode=True, strategy=optimization_algorithm, 
                             strategy_options=strategy_options, quiet=True)
    print(f"Wrote tracefile to {tracefile.resolve()}")

if __name__ == "__main__":
    # # get arguments
    # parser = argparse.ArgumentParser(
    #     prog="GEMM OpenCL Kernel tuning",
    #     description="Tuning script to tune the CLTune GEMM kernel. Based on https://github.com/CNugteren/CLBlast/blob/master/src/kernels/level3/xgemm_part1.opencl.",
    # )
    # parser.add_argument(
    #     "-s", "--size", type=int, nargs=1, default=[4096], required=False
    # )
    # args = parser.parse_args()

    # set constants
    size = 4096
    m = n = k = size

    max_fevals = None
    # allotted_time = 2*60
    max_fevals = 100
    optimization_algorithm = "random_sample"
    allotted_time = None
    tune([m,n,k], allotted_time_seconds=allotted_time, max_fevals=max_fevals, optimization_algorithm=optimization_algorithm, device=0)