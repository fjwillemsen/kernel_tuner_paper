#!/usr/bin/env python

"""Tuning script to tune the expdist kernel. Based on https://github.com/benvanwerkhoven/benchmark_kernels/blob/master/expdist/expdist.py."""


import argparse
import json
import numpy as np

from kernel_tuner import tune_kernel
from kernel_tuner.observers.nvml import NVMLObserver
from kernel_tuner.observers.register import RegisterObserver

from common import get_fallback, get_device_name

def tune_expdist(device=0, isize=1024):

    #setup tuning parameters
    tune_params = dict()
    tune_params["nvml_gr_clock"] = [1560]   # fix the core clock frequency at the A4000 boost clock
    tune_params["block_size_x"] = [2**i for i in range(5,11)][::-1] # 5
    tune_params["block_size_y"] = [2**i for i in range(6)]
    tune_params["tile_size_x"] = [1, 2, 3, 4, 5, 6, 7, 8]
    tune_params["tile_size_y"] = [1, 2, 3, 4, 5, 6, 7, 8]
    tune_params["use_shared_mem"] = [0, 1, 2]
    tune_params["loop_unroll_factor_x"] = [1, 2, 3, 4, 5, 6, 7, 8]
    tune_params["loop_unroll_factor_y"] = [1, 2, 3, 4, 5, 6, 7, 8]
    tune_params["use_column"] = [0, 1]
    tune_params["use_separate_acc"] = [0]
    tune_params["n_y_blocks"] = [2**i for i in range(11)]

    # use_shared_mem 2 and n_y_blocks are only relevant for the use_column == 1 kernel
    #tune_params["n_y_blocks"] = [2**i for i in range(11)]

    def config_valid(p):
        if p["use_column"] == 0 and p["n_y_blocks"] > 1:
            return False
        if p["use_column"] == 0 and p["use_shared_mem"] == 2:
            return False
        if p["loop_unroll_factor_x"] > p["tile_size_x"] or (p["loop_unroll_factor_x"] and p["tile_size_x"] % p["loop_unroll_factor_x"] != 0):
            return False #no need to test this loop unroll factor, as it is the same as not unrolling the loop
        if p["loop_unroll_factor_y"] > p["tile_size_y"] or (p["loop_unroll_factor_y"] and p["tile_size_y"] % p["loop_unroll_factor_y"] != 0):
            return False #no need to test this loop unroll factor, as it is the same as not unrolling the loop
        return True

    restrictions = config_valid

    # setup test input
    alloc_size = 32*isize
    size = np.int32(32*isize)
    max_blocks = np.int32( np.ceil(size / float(np.amin(tune_params["block_size_x"]))) *
                              np.ceil(size / float(np.amin(tune_params["block_size_y"]))) )
    ndim = np.int32(2)
    A = np.random.randn(alloc_size*ndim).astype(np.float64)
    B = A+0.00001*np.random.randn(alloc_size*ndim).astype(np.float64)
    scale_A = np.absolute(0.01*np.random.randn(alloc_size).astype(np.float64))
    scale_B = np.absolute(0.01*np.random.randn(alloc_size).astype(np.float64))
    cost = np.zeros((max_blocks)).astype(np.float64)

    # setup kernel
    kernel_name = "ExpDist"
    device_name = get_device_name(device)
    filename = f"outputdata/{kernel_name.lower()}/{kernel_name.lower()}_{device_name}_size-{isize}x{isize}"
    arguments = [A, B, size, size, scale_A, scale_B, cost]
    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

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

    observers = [nvmlobserver, RegisterObserver()]
    problem_size = lambda p: (size, size if p["use_column"] == 0 else p["n_y_blocks"]*p["block_size_y"]*p["tile_size_y"])

    metrics = dict()
    metrics["registers"] = lambda p: p["num_regs"]

    def FLOPs_in_partial_reduction(p):
        num_thread_blocks = np.ceil(size/(p["block_size_x"]*p["tile_size_x"])) * np.ceil(size/(p["block_size_y"]*p["tile_size_y"]))
        ops_per_thread_block = p["block_size_x"]*p["block_size_y"]/32*31+31 #minimal number of ops per warp times number of warps + #ops for 1 final warp
        return num_thread_blocks*ops_per_thread_block

    ops_per_iteration = 35 #from Nsight profiler
    metrics["GFLOP/s"] = lambda p: ((FLOPs_in_partial_reduction(p)+ops_per_iteration*size*size) /1e9) / (p["time"] / 1e3)

    cp = []

    results, env = tune_kernel(kernel_name, "expdist/expdist.cu", problem_size, arguments, tune_params,
                          grid_div_x=grid_div_x, grid_div_y=grid_div_y, metrics=metrics, iterations=32, compiler_options=cp,
                          restrictions=restrictions, objective="GFLOP/s", observers=observers, cache=filename + "_cache.json",
                          device=device, platform=0, verbose=True)
    
    # write outputs
    with open(filename + "_output.json", "w") as fh:
        json.dump(results, fh)
    with open(filename + "_env.json", "w") as fh:
        json.dump(env, fh)
    return results, env


if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser(
        prog="ExpDist Kernel tuning",
        description="Tuning script to tune the convolution kernel. Based on https://github.com/benvanwerkhoven/benchmark_kernels/blob/master/expdist/expdist.py.",
    )
    parser.add_argument(
        "-s", "--size", type=int, nargs=1, default=[1024], required=False
    )
    args = parser.parse_args()
    size = int(args.size[0])

    tune_expdist(isize=size)
