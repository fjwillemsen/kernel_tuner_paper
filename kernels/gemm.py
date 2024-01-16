#!/usr/bin/env python

"""Tuning script to tune the GEMM kernel. Based on https://github.com/benvanwerkhoven/energy_experiments/blob/master/algorithm/gemm.py."""


import time
import os
import json

import numpy as np
import kernel_tuner

from common import get_metrics, get_device_name


def ops(m, n, k):
    return (m * n * k * 2 + 2 * m * k)/1e9


def tune(inputs, device=0):
    path = os.path.dirname(os.path.realpath(__file__)) + "/gemm/"
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
    A = np.array(np.random.randn(m, k), order='F').astype(np.float32)
    B = np.array(np.random.randn(k, n), order='F').astype(np.float32)
    #C = np.array(np.random.randn(m, n), order='F').astype(np.float32)
    C = np.zeros((m, n), order='F').astype(np.float32)
    alpha, beta = np.random.randn(2).astype(np.float32)
    alpha, beta = np.array([1.0, 1.0]).astype(np.float32)

    # tunable parameters
    tune_params = dict()
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

    # additional arguments
    args = [m, n, k, alpha, beta, A, B, C]
    problem_size = (m, n)
    grid_div_x = ["MWG"]
    grid_div_y = ["NWG"]
    block_size_names = ["MDIMC", "NDIMC", "block_size_z"]
    total_flops = ops(*inputs)
    metrics = get_metrics(total_flops)
    filename = f"GEMM_{device_name}"
    print(f"{filename=}")

    # start tuning
    start = time.time()
    results, env = kernel_tuner.tune_kernel("Xgemm", kernel_string, problem_size, args, tune_params, block_size_names=block_size_names,
                             lang="OpenCL", restrictions=restrict, verbose=False, compiler_options=["-I"+path],
                             grid_div_x=grid_div_x, grid_div_y=grid_div_y,
                             device=device, platform=0, iterations=30, metrics=metrics,
                             cache=filename + "_cache.json", simulation_mode=True)
    end = time.time()
    env['execution_time'] = end-start

    # write outputs
    with open(filename + "_output.json", 'w') as fh:
        json.dump(results, fh)
    with open(filename + "_env.json", 'w') as fh:
        json.dump(env, fh)
    return results, env


if __name__ == "__main__":
    m = n = k = 4096
    results, env = tune([m,n,k], device=0)
