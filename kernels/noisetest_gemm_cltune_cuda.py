#!/usr/bin/env python

"""Tuning script to tune the GEMM OpenCL kernel using a CUDA header. Based on https://github.com/CNugteren/CLBlast/blob/master/src/kernels/level3/xgemm_part1.opencl."""


import argparse
import time
import os
import json

import numpy as np
import kernel_tuner

from common import get_metrics, get_device_name, get_fallback, get_nvcc_cuda_version_string, get_pycuda_cuda_version_string, check_pycuda_version_matches_cuda
from kernel_tuner.observers.register import RegisterObserver
from kernel_tuner.observers.nvml import NVMLObserver
from kernel_tuner.observers import BenchmarkObserver


def ops(m, n, k):
    return (m * n * k * 2 + 2 * m * k)/1e9


def tune(inputs, backends, device=0):
    path = os.path.dirname(os.path.realpath(__file__)) + "/gemm_cltune_cuda/"
    device_name = get_device_name(device)
    print(device_name)

    # check selected backends
    if "CUDA" in backends:
        assert (
            check_pycuda_version_matches_cuda()
        ), f"PyCUDA was compiled against a different CUDA version ({get_pycuda_cuda_version_string()}) than the current CUDA version ({get_nvcc_cuda_version_string()})"

    # kernel string
    kernel_string = '#include "../cl_to_cuda.h"'
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
    tune_params = dict()
    tune_params["nvml_gr_clock"] = [840]   # A4000: (base+boost)/2 = 1147, largest supported in range is 1140
    tune_params["nvml_mem_clock"] = [6501]  # A4000: nvidia-smi --query-supported-clocks=mem --format=csv
    if m == n == k == 4096:
        tune_params = {
            "nvml_gr_clock": [840],     # A4000: (base+boost)/2 = 1147, largest supported in range is 1140
            "nvml_mem_clock": [6501],   # A4000: nvidia-smi --query-supported-clocks=mem --format=csv
            "GEMMK": [0],
            "MWG": [128],
            "NWG": [128],
            "KWG": [32],
            "MDIMC": [16],
            "NDIMC": [8],
            "MDIMA": [8],
            "NDIMB": [32],
            "KWI": [2],
            "VWM": [4],
            "VWN": [2],
            "STRM": [1],
            "STRN": [0],
            "SA": [1],
            "SB": [1],
            "KREG": [1],
            "PRECISION": [32],
        }
        tune_params["REPEAT"] = [i for i in range(1000)]
    else:
        raise ValueError(f"Tune params for {m=}x{n=}x{k=} not set")

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

    class ResetL2Observer(BenchmarkObserver):

        def __init__(self, args):
            self.args = args

        def before_start(self):
            for i, arg in enumerate(self.args):
                self.dev.memcpy_htod(self.dev.allocations[i], arg)

        def get_results(self):
            return {}

    # additional arguments
    args = [m, n, k, alpha, beta, A, B, C, np.int32(0), np.int32(0)]  
    observers = [nvmlobserver, RegisterObserver(), ResetL2Observer(args[-3:])]
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
        filename = f"outputdata/gemm_cltune_cuda/gemm_cltune_cuda_{device_name}_size-{m}x{n}x{k}_noisetest_backend-{backend}_CUDA-{cuda_version}"

        # start tuning
        print(f"Starting tuning, {filename=}")
        start = time.time()
        results, env = kernel_tuner.tune_kernel("Xgemm", kernel_string, problem_size, args, tune_params, block_size_names=block_size_names,
                                lang=backend, restrictions=restrict, verbose=False, compiler_options=["-I"+path],
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
        prog="GEMM OpenCL Kernel tuning using CUDA header",
        description="Tuning script to tune the GEMM kernel. Based on https://github.com/CNugteren/CLBlast/blob/master/src/kernels/level3/xgemm_part1.opencl.",
    )
    parser.add_argument(
        "-b", "--backends", nargs="+", default=["CUDA", "CUPY", "NVCUDA"], required=True
    )
    parser.add_argument(
        "-s", "--size", type=int, nargs=1, default=[4096], required=False
    )
    args = parser.parse_args()
    backends = args.backends
    size = int(args.size[0])

    # start tuning process
    m = n = k = size
    tune([m,n,k], backends=backends, device=0)
