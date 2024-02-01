#!/usr/bin/env python

"""Tuning script to tune the GEMM CUDA kernel. Based on https://github.com/NTNU-HPC-Lab/BAT/tree/master/batbench/benchmarks/GEMM."""


import argparse
import time
import os
import json

import numpy as np
import kernel_tuner

from common import get_metrics, get_device_name, get_fallback, get_nvcc_cuda_version_string, get_pycuda_cuda_version_string, check_pycuda_version_matches_cuda
from kernel_tuner.observers import BenchmarkObserver
from kernel_tuner.observers.nvml import NVMLObserver


def ops(m, n, k):
    return (m * n * k * 2 + 2 * m * k)/1e9


def tune(inputs, backends, device=0):
    device_name = get_device_name(device)
    print(device_name)

    # check selected backends
    if "CUDA" in backends:
        assert (
            check_pycuda_version_matches_cuda()
        ), f"PyCUDA was compiled against a different CUDA version ({get_pycuda_cuda_version_string()}) than the current CUDA version ({get_nvcc_cuda_version_string()})"

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
    tune_params["nvml_gr_clock"] = [1560]   # fix the clock frequency at the A4000 boost clock
    if m == n == k == 16384:
        tune_params = {
            "MWG": [128],
            "NWG": [128],
            "KWG": [32],
            "MDIMC": [16],
            "NDIMC": [8],
            "MDIMA": [8],
            "NDIMB": [16],
            "KWI": [2],
            "VWM": [1],
            "VWN": [1],
            "STRM": [0],
            "STRN": [1],
            "SA": [0],
            "SB": [0],
            "PRECISION": [32],
        }
        tune_params["REPEAT"] = [i for i in range(2000)]
    else:
        raise ValueError(f"Tune params for {m=}x{n=}x{k=} not set")

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

    # observer for counting the number of registers
    class RegisterObserver(BenchmarkObserver):
        """Observer for counting the number of registers."""

        def get_results(self):
            return {
                "num_regs": self.dev.current_module.get_function(
                    "convolution_kernel"
                ).num_regs
            }

    # additional arguments
    observers = [nvmlobserver, RegisterObserver()]
    args = [m, n, k, alpha, beta, A, B, C]
    problem_size = (m, n)
    grid_div_x = ["MWG"]
    grid_div_y = ["NWG"]
    block_size_names = ["MDIMC", "NDIMC", "block_size_z"]
    total_flops = ops(*inputs)
    metrics = get_metrics(total_flops)

    # CUDA and backend selection
    cuda_version = get_nvcc_cuda_version_string()
    assert cuda_version in ["11.2", "12.3"]
    for backend in backends:
        filename = f"outputdata/gemm_cuda/gemm_cuda_{device_name}_size-{m}x{n}x{k}_noisetest_backend-{backend}_CUDA-{cuda_version}"

        # start tuning
        print(f"Starting tuning, {filename=}")
        start = time.time()
        results, env = kernel_tuner.tune_kernel("gemm_fast", "gemm_cuda/gemm.cu", problem_size, args, tune_params, block_size_names=block_size_names,
                                lang=backend, restrictions=restrict, verbose=True, 
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


if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser(
        prog="GEMM CUDA Kernel tuning",
        description="Tuning script to tune the convolution kernel. Based on https://github.com/NTNU-HPC-Lab/BAT/tree/master/batbench/benchmarks/GEMM.",
    )
    parser.add_argument(
        "-b", "--backends", nargs="+", default=["CUDA", "CUPY", "NVCUDA"], required=True
    )
    parser.add_argument(
        "-s", "--size", type=int, nargs=1, default=[16384], required=False
    )
    args = parser.parse_args()
    backends = args.backends
    size = int(args.size[0])

    # start tuning process
    m = n = k = size
    results, env = tune([m,n,k], backends=backends, device=0)
