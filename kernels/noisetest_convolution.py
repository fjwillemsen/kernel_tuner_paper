#!/usr/bin/env python

"""Tuning script to tune the convolution kernel. Based on https://github.com/benvanwerkhoven/energy_experiments/blob/master/algorithm/convolution.py."""


import json
import os
import time

import kernel_tuner
import numpy
from common import (
    check_pycuda_version_matches_cuda,
    get_device_name,
    get_metrics,
    get_nvcc_cuda_version_string,
)


def ops(w, h, fw, fh):
    return (w * h * fw * fh * 2) / 1e9


def tune(inputs, device=0):
    device_name = get_device_name(device)
    image_width, image_height, filter_width, filter_height = inputs

    # kernel string
    with open(
        os.path.dirname(os.path.realpath(__file__)) + "/convolution/convolution.cu", "r"
    ) as f:
        kernel_string = f.read()

    # tunable parameters
    tune_params = {
        "block_size_x": [64],
        "block_size_y": [2],
        "tile_size_x": [1],
        "tile_size_y": [4],
        "read_only": [1],
        "use_padding": [0],
    }
    tune_params["REPEAT"] = [i for i in range(1000)]

    # restrictions: limit the search to only use padding when its effective
    restrict = [
        "(use_padding==0 or (block_size_x % 32 != 0))",
        "((block_size_x*tile_size_x+4)*(block_size_y*tile_size_y+4) < 12*1024)",
    ]
    restrict.append(
        "(((block_size_x*tile_size_x+%d)*(block_size_y*tile_size_y+%d)) < 12*1024)"
        % (filter_width - 1, filter_height - 1)
    )

    # additional arguments
    problem_size = (image_width, image_height)
    size = numpy.prod(problem_size)
    largest_fh = filter_height
    largest_fw = filter_width
    input_size = (problem_size[0] + largest_fw - 1) * (problem_size[1] + largest_fh - 1)
    output_image = numpy.zeros(size).astype(numpy.float32)
    input_image = numpy.random.randn(input_size).astype(numpy.float32)
    filter_weights = numpy.random.randn(largest_fh * largest_fw).astype(numpy.float32)
    cmem_args = {"d_filter": filter_weights}
    args = [output_image, input_image, filter_weights]
    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]
    total_flops = ops(*inputs)
    metrics = get_metrics(total_flops)

    # backend selection
    # backends = ["CUDA", "CUPY", "NVCUDA"]
    backends = ["CUPY"]
    cuda_version = get_nvcc_cuda_version_string()
    for backend in backends:
        filename = f"outputdata/convolution_{device_name}_noisetest_backend-{backend}_CUDA-{cuda_version}"
        print(f"{filename=}")

        # start tuning
        start = time.time()
        results, env = kernel_tuner.tune_kernel(
            "convolution_kernel",
            kernel_string,
            problem_size,
            args,
            tune_params,
            grid_div_y=grid_div_y,
            grid_div_x=grid_div_x,
            cmem_args=cmem_args,
            device=device,
            platform=0,
            lang=backend,
            verbose=True,
            metrics=metrics,
            restrictions=restrict,
            iterations=30,
            cache=filename + "_cache.json",
        )
        end = time.time()
        env["execution_time"] = end - start

        # write outputs
        with open(filename + "_output.json", "w") as fh:
            json.dump(results, fh)
        with open(filename + "_env.json", "w") as fh:
            json.dump(env, fh)


if __name__ == "__main__":
    w = h = 4096
    fw = fh = 17
    assert (
        check_pycuda_version_matches_cuda()
    ), "PyCUDA was compiled against a different CUDA version than the current CUDA version"
    tune([w, h, fw, fh], device=0)
