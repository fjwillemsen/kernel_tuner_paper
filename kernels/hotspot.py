#!/usr/bin/env python

"""Tuning script to tune the hotspot kernel. Based on https://github.com/benvanwerkhoven/benchmark_kernels/blob/master/hotspot/hotspot.py."""


import argparse
import json
import numpy as np

import kernel_tuner as kt
from kernel_tuner.observers import BenchmarkObserver
from kernel_tuner.observers.nvml import NVMLObserver
from common import get_fallback


def test_temporal_tiling():

    problem_size = (7, 7)

    tune_params, max_tfactor = get_tunable_parameters(problem_size)

    temp_src, power, temp_dst = get_input_data(problem_size, max_tfactor)

    test_input = np.array([i for i in range(np.prod(problem_size))]).reshape(*problem_size)
    temp_src[max_tfactor:-max_tfactor,max_tfactor:-max_tfactor] = test_input

    # setup arguments
    step_div_cap, Rx_1, Ry_1, Rz_1 = get_input_arguments(*problem_size)
    args = [power, temp_src, temp_dst, Rx_1, Ry_1, Rz_1, step_div_cap]

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    # call the kernel once with temporal tiling factor = 1
    params = dict(grid_width=problem_size[0], grid_height=problem_size[1], block_size_x=16, block_size_y=16,
                  tile_size_x=1, tile_size_y=1, temporal_tiling_factor=1, max_tfactor=max_tfactor)
    reference = kt.run_kernel("calculate_temp", "hotspot_new.cu", problem_size, args, params,
                               grid_div_x=grid_div_x, grid_div_y=grid_div_y)

    print(reference[2])

    # replace the input with the output of the first kernel
    temp2 = np.zeros_like(temp_src)
    temp2[max_tfactor:-max_tfactor,max_tfactor:-max_tfactor] = reference[2]

    # call the kernel again with temporal tiling factor = 1
    args2 = [power, temp2, temp_dst, Rx_1, Ry_1, Rz_1, step_div_cap]
    reference2 = kt.run_kernel("calculate_temp", "hotspot_new.cu", problem_size, args2, params,
                               grid_div_x=grid_div_x, grid_div_y=grid_div_y)
    answer = [None for _ in args]
    answer[2] = reference2[2]

    print(answer[2])

    # tune the kernel with temporal tiling factor = 2
    tune_params["block_size_y"] = [16, 32]
    tune_params["block_size_x"] = [16, 32]
    tune_params["tile_size_x"] = [1, 2, 4]
    tune_params["tile_size_y"] = [1, 2, 4]
    tune_params["temporal_tiling_factor"] = [2]

    results, env = kt.tune_kernel("calculate_temp", "hotspot_new.cu", problem_size, args, tune_params,
                                  grid_div_x=grid_div_x, grid_div_y=grid_div_y,
                                  verbose=True, answer=answer)



def get_input_arguments(grid_rows, grid_cols):

    # maximum power density possible (say 300W for a 10mm x 10mm chip)  */
    MAX_PD = 3.0e6

    # required precision in degrees */
    PRECISION = 0.001
    SPEC_HEAT_SI = 1.75e6
    K_SI = 100

    # capacitance fitting factor    */
    FACTOR_CHIP = 0.5

    # chip parameters   */
    t_chip = 0.0005
    chip_height = 0.016
    chip_width = 0.016

    grid_height = chip_height/grid_rows
    grid_width = chip_width/grid_cols
    cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height
    Rx = grid_width / (2.0 * K_SI * t_chip * grid_height)
    Ry = grid_height / (2.0 * K_SI * t_chip * grid_width)
    Rz = t_chip / (K_SI * grid_height * grid_width)
    max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI)
    step = PRECISION / max_slope
    step_div_cap = step/cap

    return [np.float32(i) for i in [step_div_cap, 1/Rx, 1/Ry, 1/Rz]]



def get_tunable_parameters(problem_size):

    tune_params = dict()
    tune_params["nvml_gr_clock"] = [1560]   # fix the core clock frequency at the A4000 boost clock

    # input sizes need at compile time
    tune_params["grid_width"] = [problem_size[0]]
    tune_params["grid_height"] = [problem_size[1]]

    # actual tunable parameters
    tune_params["block_size_x"] = [1, 2, 4, 8, 16] + [32*i for i in range(1,33)]
    tune_params["block_size_y"] = [2**i for i in range(6)]
    tune_params["tile_size_x"] = [i for i in range(1,11)]
    tune_params["tile_size_y"] = [i for i in range(1,11)]

    tune_params["temporal_tiling_factor"] = [i for i in range(1,11)]

    max_tfactor = max(tune_params["temporal_tiling_factor"])
    tune_params["max_tfactor"] = [max_tfactor]
    tune_params["loop_unroll_factor_t"] = [i for i in range(1,max_tfactor+1)]

    tune_params["sh_power"] = [0,1]
    tune_params["blocks_per_sm"] = [0,1,2,3,4]

    return tune_params, max_tfactor



def get_input_data(problem_size, max_tfactor):

    input_width = problem_size[0] + 2*max_tfactor
    input_height = problem_size[1] + 2*max_tfactor

    # setup main input/output data with a zero border around the input
    temp_src = np.zeros((input_height,input_width), dtype=np.float32)
    temp_src[max_tfactor:-max_tfactor,max_tfactor:-max_tfactor] = np.random.random(problem_size)+324
    power = np.zeros((input_height,input_width), dtype=np.float32)
    power[max_tfactor:-max_tfactor,max_tfactor:-max_tfactor] = np.random.random(problem_size)
    temp_dst = np.zeros(problem_size, dtype=np.float32)

    return temp_src, power, temp_dst


def get_device_info(device):
    """ Get device info using cupy """
    import cupy as cp
    result = dict()
    cupy_info = str(cp._cupyx.get_runtime_info()).split("\n")[:-1]
    info_dict = {s.split(":")[0].strip():s.split(":")[1].strip() for s in cupy_info}
    result["device_name"] = info_dict[f'Device {device} Name']

    with cp.cuda.Device(0) as dev:
        result['max_threads'] = dev.attributes['MaxThreadsPerBlock']
        result['max_shared_memory_per_block'] = dev.attributes['MaxSharedMemoryPerBlock']
        result['max_shared_memory'] = dev.attributes['MaxSharedMemoryPerMultiprocessor']

    return result



def tune(device=0, size=4096):
    dev = get_device_info(device)
    filename = f"outputdata/hotspot/hotspot_{dev['device_name']}_size-{size}x{size}"
    problem_size = (size, size)
    tune_params, max_tfactor = get_tunable_parameters(problem_size)
    temp_src, power, temp_dst = get_input_data(problem_size, max_tfactor)

    # setup arguments
    step_div_cap, Rx_1, Ry_1, Rz_1 = get_input_arguments(*problem_size)
    args = [power, temp_src, temp_dst, Rx_1, Ry_1, Rz_1, step_div_cap]

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]
    restrictions = ["block_size_x*block_size_y >= 32",
                    "temporal_tiling_factor % loop_unroll_factor_t == 0",
                    f"block_size_x*block_size_y <= {dev['max_threads']}",
                    f"(block_size_x*tile_size_x + temporal_tiling_factor * 2) * (block_size_y*tile_size_y + temporal_tiling_factor * 2) * (2+sh_power) * 4 <= {dev['max_shared_memory_per_block']}",
                    f"blocks_per_sm == 0 or (((block_size_x*tile_size_x + temporal_tiling_factor * 2) * (block_size_y*tile_size_y + temporal_tiling_factor * 2) * (2+sh_power) * 4) * blocks_per_sm <= {dev['max_shared_memory']})"]

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

    class RegisterObserver(BenchmarkObserver):
        def get_results(self):
            return {"num_regs": self.dev.current_module.get_function("calculate_temp").num_regs}

    observers = [nvmlobserver, RegisterObserver()]

    # setup metric
    metrics = dict()

    # Temporal tiling introduces redundant work but saves data transfers
    # needed when doing the same work in multiple kernels
    # in the GFLOP/s calculation we don't count this double work, but only the 'real' work counts
    metrics["GFLOP/s"] = lambda p : (p["temporal_tiling_factor"] * 15 * problem_size[0] * problem_size[1])/1e9 / (p["time"]/1e3)
    metrics["reg"] = lambda p : p["num_regs"]

    # call the tuner
    results, env = kt.tune_kernel("calculate_temp", "hotspot/hotspot.cu", problem_size, args, tune_params,
                                  metrics=metrics, grid_div_x=grid_div_x, grid_div_y=grid_div_y,
                                  restrictions=restrictions, iterations=32, verbose=True, lang="cupy", observers=observers,
                                  objective="GFLOP/s", cache=filename + "_cache.json", device=device, platform=0)
    
    # write outputs
    with open(filename + "_output.json", "w") as fh:
        json.dump(results, fh)
    with open(filename + "_env.json", "w") as fh:
        json.dump(env, fh)
    return results, env




if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser(
        prog="Hotspot Kernel tuning",
        description="Tuning script to tune the convolution kernel. Based on https://github.com/benvanwerkhoven/benchmark_kernels/blob/master/hotspot/hotspot.py.",
    )
    parser.add_argument(
        "-s", "--size", type=int, nargs=1, default=[4096], required=False
    )
    args = parser.parse_args()
    size = int(args.size[0])

    tune(size=size)
    #test_temporal_tiling()
