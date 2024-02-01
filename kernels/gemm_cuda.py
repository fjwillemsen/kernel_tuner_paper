#!/usr/bin/env python

"""Tuning script to tune the GEMM CUDA kernel. Based on https://github.com/NTNU-HPC-Lab/BAT/tree/master/batbench/benchmarks/GEMM."""


import argparse
import time
import os
import json

import numpy as np
import kernel_tuner

from common import get_metrics, get_device_name, get_fallback
from kernel_tuner.observers.nvml import NVMLObserver


def ops(m, n, k):
    return (m * n * k) / 1e9
    # return (m * n * k * 2 + 2 * m * k)/1e9


def tune(inputs, device=0):
    device_name = get_device_name(device)
    print(device_name)

    # n = np.int32(32)
    # m = np.int32(16)
    # k = np.int32(32)
    m, n, k = [np.int32(i) for i in inputs]

    #// Matrices are accessed as follows:
    #// A: [k*M + m], with 'k' ranging from 0:K and 'm' from 0:M (m,k,m)
    #// B: [k*N + n], with 'k' ranging from 0:K and 'n' from 0:N (n,k,n)
    #// C: [n*M + m], with 'n' ranging from 0:N and 'm' from 0:M (m,n,m)

    # input / output array intialization
    print("array initialization (may take a while)")
    A = np.array(np.random.randn(m, k), order='F').astype(np.float32)
    B = np.array(np.random.randn(k, n), order='F').astype(np.float32)
    # C = np.array(np.random.randn(m, n), order='F').astype(np.float32)
    C = np.zeros((m, n), order='F').astype(np.float32)
    alpha, beta = np.random.randn(2).astype(np.float32)
    alpha, beta = np.array([1.0, 1.0]).astype(np.float32)

    # tunable parameters
    print("setting tunable parameters")
    tune_params = dict()
    tune_params["nvml_gr_clock"] = [1560]   # fix the core clock frequency at the A4000 boost clock
    tune_params["MWG"] = [16, 32, 64, 128]
    tune_params["NWG"] = [16, 32, 64, 128]
    tune_params["KWG"] = [16, 32]
    tune_params["MDIMC"] = [8, 16, 32]
    tune_params["NDIMC"] = [8, 16, 32]
    tune_params["MDIMA"] = [8, 16, 32]
    tune_params["NDIMB"] = [8, 16, 32]
    tune_params["KWI"] = [2, 8]
    tune_params["VWM"] = [1]
    tune_params["VWN"] = [1]
    tune_params["STRM"] = [0, 1]
    tune_params["STRN"] = [0, 1]
    tune_params["SA"] = [0]
    tune_params["SB"] = [0]
    tune_params["PRECISION"] = [32]

    # restrictions
    restrict = []
    restrict += ["(KWG % KWI) == 0"]
    restrict += ["(MWG % (MDIMC * VWM)) == 0"]
    restrict += ["(NWG % (NDIMC * VWN)) == 0"]
    restrict += ["(MWG % (MDIMA * VWM)) == 0"]
    restrict += ["(NWG % (NDIMB * VWN)) == 0"]
    restrict += ["(KWG % ((MDIMC * NDIMC) / MDIMA)) == 0"]
    restrict += ["(KWG % ((MDIMC * NDIMC) / NDIMB)) == 0"]

    # observer for the frequencies and temperature
    nvmlobserver = NVMLObserver(
        [
            "core_freq",
            "mem_freq",
            "temperature",
            "nvml_energy", 
        ],
        save_all=True,
        nvidia_smi_fallback=get_fallback(),
        use_locked_clocks=True
    )

    # additional arguments
    observers = [nvmlobserver]
    args = [m, n, k, alpha, beta, A, B, C]
    problem_size = (m, n)
    grid_div_x = ["MWG"]
    grid_div_y = ["NWG"]
    block_size_names = ["MDIMC", "NDIMC", "block_size_z"]
    total_flops = ops(*inputs)
    metrics = get_metrics(total_flops)
    filename = f"outputdata/gemm_cuda/gemm_cuda_{device_name}_size-{m}x{n}x{k}"

    # start tuning
    print(f"Starting tuning, {filename=}")
    start = time.time()
    results, env = kernel_tuner.tune_kernel("gemm_fast", "gemm_cuda/gemm.cu", problem_size, args, tune_params, block_size_names=block_size_names,
                             lang="cupy", restrictions=restrict, verbose=True, 
                             grid_div_x=grid_div_x, grid_div_y=grid_div_y, observers=observers, 
                             device=device, platform=0, iterations=32, metrics=metrics,
                             cache=filename + "_cache.json", simulation_mode=False)
    end = time.time()
    env['execution_time'] = end-start

    # write outputs
    with open(filename + "_output.json", 'w') as fh:
        json.dump(results, fh)
    with open(filename + "_env.json", 'w') as fh:
        json.dump(env, fh)
    return results, env


if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser(
        prog="GEMM CUDA Kernel tuning",
        description="Tuning script to tune the convolution kernel. Based on https://github.com/NTNU-HPC-Lab/BAT/tree/master/batbench/benchmarks/GEMM.",
    )
    parser.add_argument(
        "-s", "--size", type=int, nargs=1, default=[16384], required=False
    )
    args = parser.parse_args()
    size = int(args.size[0])

    # start tuning process
    m = n = k = size
    results, env = tune([m,n,k], device=0)
