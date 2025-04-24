#!/usr/bin/env python

"""Tuning script to tune the CLTune GEMM CUDA kernel. Based on https://github.com/CNugteren/CLBlast/blob/master/src/kernels/level3/xgemm_part1.opencl."""

import os
import sys
import time

import kernel_tuner
import numpy as np
from kernel_tuner.file_utils import store_metadata_file, store_output_file


def get_metrics(total_flops):
    metrics = dict()
    metrics["GFLOP/s"] = lambda p: total_flops / (p["time"] / 1000.0)
    return metrics


def ops(m, n, k):
    return (m * n * k * 2 + 2 * m * k) / 1e9


def tune(
    device_name: str,
    inputs = [4096, 4096, 4096],
    strategy="brute_force",
    strategy_options=None,
    verbose=True,
    quiet=False,
    simulation_mode=False,
    lang="CUDA",
    searchspace_set=2,
    cachefile_path=None,
):
    path = os.path.dirname(os.path.realpath(__file__)) + "/gemm_milo/"

    # kernel string
    kernel_string = '#include "cl_to_cuda.h"\n' if lang != "OpenCL" else ""
    files = ["common.opencl", "xgemm_part1.opencl", "xgemm_part2.opencl", "xgemm_part3.opencl", "xgemm_part4.opencl"]
    for f in files:
        with open(path + f, "r") as fp:
            kernel_string += fp.read()

    # n = np.int32(32)
    # m = np.int32(16)
    # k = np.int32(32)
    m, n, k = [np.int32(i) for i in inputs]

    # // Matrices are accessed as follows:
    # // A: [k*M + m], with 'k' ranging from 0:K and 'm' from 0:M (m,k,m)
    # // B: [k*N + n], with 'k' ranging from 0:K and 'n' from 0:N (n,k,n)
    # // C: [n*M + m], with 'n' ranging from 0:N and 'm' from 0:M (m,n,m)

    # input / output array intialization
    print("array initialization (may take a while)")
    A = np.array(np.random.randn(m, k), order="F").astype(np.float32)
    B = np.array(np.random.randn(k, n), order="F").astype(np.float32)
    # C = np.array(np.random.randn(m, n), order='F').astype(np.float32)
    C = np.zeros((m, n), order="F").astype(np.float32)
    alpha, beta = np.random.randn(2).astype(np.float32)
    alpha, beta = np.array([1.0, 1.0]).astype(np.float32)

    # tunable parameters
    print("setting tunable parameters")
    tune_params = dict()

    if searchspace_set == 0:
        # original Kernel Tuner parameters
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
    elif searchspace_set == 2:
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
    restrict += ["(MWG != 128 or NWG != 128 or MDIMC != 8 or NDIMC != 8)"]  # for ATF parser compatibility
    # restrict += ["not (MWG == 128 and NWG == 128 and MDIMC == 8 and NDIMC == 8)"]

    # additional arguments
    args = [m, n, k, alpha, beta, A, B, C, np.int32(0), np.int32(0)]
    problem_size = (m, n)
    grid_div_x = ["MWG"]
    grid_div_y = ["NWG"]
    block_size_names = ["MDIMC", "NDIMC", "block_size_z"]
    total_flops = ops(*inputs)
    metrics = get_metrics(total_flops)

    # base_cachepath = f"../cachefiles/gemm_milo/{device_name.upper()}"

    # start tuning
    start = time.time()
    results, env = kernel_tuner.tune_kernel(
        "Xgemm",
        kernel_string,
        problem_size,
        args,
        tune_params,
        block_size_names=block_size_names,
        restrictions=restrict,
        compiler_options=["-I" + path],
        lang=lang,
        grid_div_x=grid_div_x,
        grid_div_y=grid_div_y,
        device=0,
        platform=0,
        iterations=32,
        metrics=metrics,
        cache=cachefile_path,
        verbose=verbose,
        quiet=quiet,
        strategy=strategy,
        strategy_options=strategy_options,
        simulation_mode=simulation_mode,
    )
    end = time.time()
    env["execution_time"] = end - start

    # store_output_file(f"{base_cachepath}-results.json", results, tune_params)
    # store_metadata_file(f"{base_cachepath}-metadata.json")
    return results, env


if __name__ == "__main__":
    language = sys.argv[1]
    device_name = sys.argv[2]

    if len(sys.argv) != 3:
        raise ValueError(f"Usage: python gemm_milo.py [language ('HIP', 'OpenCL' or 'CUDA')] [device name], given: {sys.argv}")

    if language not in ("HIP", "OpenCL", "CUDA"):
        raise ValueError(f"{language} not valid, specify HIP, OpenCL or CUDA")

    # start tuning process
    m = n = k = 4096
    tune([m, n, k], device_name=device_name, lang=language)
