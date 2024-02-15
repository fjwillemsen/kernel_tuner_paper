#!/usr/bin/env python
"""Script to test whether and where the OpenCL backend may be leaking time."""


import argparse
import time
import os
import json

import numpy as np
import kernel_tuner

from common import get_metrics, get_device_name, get_fallback
from kernel_tuner.observers.nvml import NVMLObserver
from pathlib import Path


def ops(m, n, k):
    return (m * n * k * 2 + 2 * m * k)/1e9


def tune(inputs, device=0):
    path = os.path.dirname(os.path.realpath(__file__)) + "/gemm_opencl/"
    device_name = get_device_name(device)

    # kernel string
    kernel_string = ""
    files = ["common.opencl", "xgemm_part1.opencl", "xgemm_part2.opencl", "xgemm_part3.opencl"]
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
    tune_params = {
        "nvml_gr_clock": [840],     # A4000: (base+boost)/2 = 1147, largest supported in range is 1140
        "nvml_mem_clock": [6501],    # A4000: nvidia-smi --query-supported-clocks=mem --format=csv
        "MWG": [32, 64, 128],
        "NWG": [64, 128],
        "MDIMC": [16],
        "NDIMC": [8],
        "MDIMA": [32],
        "NDIMB": [32],
        "VWM": [4],
        "VWN": [2], # differs from CUDA: 4
        "SA": [1],
        "SB": [1],
        "KWG": [32],
        "KWI": [2],
        "STRM": [0],
        "STRN": [0],
        "PRECISION": [32],
    }

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
    filename = f"outputdata/test_gemm_opencl_times/gemm_opencl_{device_name}_size-{m}x{n}x{k}"
    cache_filename = filename + "_cache.json"

    # start tuning
    print(f"Starting tuning, {filename=}")
    start = time.time()
    results, env = kernel_tuner.tune_kernel("Xgemm", kernel_string, problem_size, args, tune_params, block_size_names=block_size_names,
                             lang="OpenCL", restrictions=restrict, verbose=False, compiler_options=["-I"+path],
                             grid_div_x=grid_div_x, grid_div_y=grid_div_y, observers=observers,
                             device=device, platform=0, iterations=32, metrics=metrics,
                             cache=cache_filename, simulation_mode=False)
    end = time.time()
    env['execution_time'] = end-start

    # # write outputs
    # with open(filename + "_output.json", 'w') as fh:
    #     json.dump(results, fh)
    # with open(filename + "_env.json", 'w') as fh:
    #     json.dump(env, fh)

    # delete cachefile
    Path(filename + "_cache.json").unlink()
    return results, env


if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser(
        prog="GEMM OpenCL Kernel tuning",
        description="Tuning script to tune the convolution kernel. Based on https://github.com/benvanwerkhoven/energy_experiments/blob/master/algorithm/gemm.py.",
    )
    parser.add_argument(
        "-s", "--size", type=int, nargs=1, default=[4096], required=False
    )
    args = parser.parse_args()
    size = int(args.size[0])

    # start tuning process
    m = n = k = size
    results, env = tune([m,n,k], device=0)

