#!/usr/bin/env python

"""Tuning script to tune the convolution kernel. Based on https://github.com/benvanwerkhoven/energy_experiments/blob/master/algorithm/convolution.py."""


import json
import os
import time


def ops(w, h, fw, fh):
    return (w * h * fw * fh * 2) / 1e9


def tune(inputs, device=0):
    # do imports
    import kernel_tuner
    import numpy
    from common import get_device_name, get_metrics, get_fallback
    from kernel_tuner.observers.nvml import NVMLObserver

    # get inputs
    device_name = get_device_name(device)
    image_width, image_height, filter_width, filter_height = inputs

    # kernel string
    with open(
        os.path.dirname(os.path.realpath(__file__)) + "/convolution/convolution.cu", "r"
    ) as f:
        kernel_string = f.read()

    # tunable parameters
    tune_params = dict()
    tune_params["nvml_gr_clock"] = [1560]   # fix the core clock frequency at the A4000 boost clock
    tune_params["nvml_mem_clock"] = [7001]  # fix the memory clock frequency
    tune_params["block_size_x"] = [16 * i for i in range(1, 17)]
    tune_params["block_size_y"] = [2**i for i in range(5)]
    tune_params["tile_size_x"] = [i for i in range(1, 5)]
    tune_params["tile_size_y"] = [i for i in range(1, 5)]
    tune_params["read_only"] = [0, 1]  # toggle using the read-only cache
    tune_params["use_padding"] = [
        0,
        1,
    ]  # toggle the insertion of padding in shared memory

    # restrictions: limit the search to only use padding when its effective
    restrict = [
        "(use_padding==0 or (block_size_x % 32 != 0))",
        "((block_size_x*tile_size_x+4)*(block_size_y*tile_size_y+4) < 12*1024)",
    ]
    restrict.append(
        "(((block_size_x*tile_size_x+%d)*(block_size_y*tile_size_y+%d)) < 12*1024)"
        % (filter_width - 1, filter_height - 1)
    )

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
    problem_size = (image_width, image_height)
    size = numpy.prod(problem_size)
    largest_fh = filter_height
    largest_fw = filter_width
    input_size = (problem_size[0] + largest_fw - 1) * (problem_size[1] + largest_fh - 1)
    output_image = numpy.zeros(size).astype(numpy.float32)
    input_image = numpy.random.randn(input_size).astype(numpy.float32)
    filter_weights = numpy.random.randn(largest_fh * largest_fw).astype(numpy.float32)
    cmem_args = {"d_filter": filter_weights}
    input_args = [output_image, input_image, filter_weights]
    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]
    total_flops = ops(*inputs)
    metrics = get_metrics(total_flops)
    filename = f"outputdata/convolution_{device_name}"
    print(f"{filename=}")

    # start tuning
    start = time.time()
    results, env = kernel_tuner.tune_kernel(
        "convolution_kernel",
        kernel_string,
        problem_size,
        input_args,
        tune_params,
        simulation_mode=False,
        grid_div_y=grid_div_y,
        grid_div_x=grid_div_x,
        cmem_args=cmem_args,
        device=device,
        platform=0,
        lang="CUPY",
        verbose=True,
        metrics=metrics,
        restrictions=restrict,
        observers=observers,
        iterations=32,
        cache=filename + "_cache.json",
    )
    end = time.time()
    env["execution_time"] = end - start

    # write outputs
    with open(filename + "_output.json", "w") as fh:
        json.dump(results, fh)
    with open(filename + "_env.json", "w") as fh:
        json.dump(env, fh)
    return results, env


if __name__ == "__main__":
    w = h = 8192
    fw = fh = 15
    results, env = tune([w, h, fw, fh], device=0)
