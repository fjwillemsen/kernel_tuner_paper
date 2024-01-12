"""This script is intended to assess the consistency of Kernel Tuner's simulation mode."""

import time
from pathlib import Path

import kernel_tuner
import numpy

# get the path to the kernel files
path_script = Path(__file__).parent.resolve()
path_kernel_dir = path_script / ".." / "BAT" / "batbench" / "benchmarks" / "convolution"
assert path_kernel_dir.exists() and path_kernel_dir.is_dir()

unit = "GFLOP"


def ops(w, h, fw, fh):
    return (w * h * fw * fh * 2) / 1e9


def tune(inputs):
    image_width, image_height, filter_width, filter_height = inputs

    path_kernel = path_kernel_dir / "convolution.cu"
    assert path_kernel.exists() and path_kernel.is_file()
    kernel_string = path_kernel.read_text()

    # setup tunable parameters
    tune_params = dict()
    # tune_params["pwr_limit"] = get_pwr_limit(pwr_limit, 0)

    tune_params["block_size_x"] = [16 * i for i in range(1, 17)]
    tune_params["block_size_y"] = [2**i for i in range(5)]
    tune_params["tile_size_x"] = [i for i in range(1, 5)]
    tune_params["tile_size_y"] = [i for i in range(1, 5)]
    tune_params["read_only"] = [0, 1]  # toggle using the read-only cache

    # do dry run
    # tune_params["nvml_gr_clock"] = [2100]
    # tune_params["block_size_x"] = [16]
    # tune_params["block_size_y"] = [1]
    # tune_params["tile_size_x"] = [1, 2, 4]
    # tune_params["tile_size_y"] = [1]
    # tune_params["read_only"] = [1]    #toggle using the read-only cache

    tune_params["use_padding"] = [
        0,
        1,
    ]  # toggle the insertion of padding in shared memory

    # limit the search to only use padding when its effective
    restrict = [
        "(use_padding==0 or (block_size_x % 32 != 0))",
        "((block_size_x*tile_size_x+4)*(block_size_y*tile_size_y+4) < 12*1024)",
    ]
    restrict.append(
        "(((block_size_x*tile_size_x+%d)*(block_size_y*tile_size_y+%d)) < 12*1024)"
        % (filter_width - 1, filter_height - 1)
    )

    print(restrict)

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
    metrics = dict()
    metrics["GFLOP/s"] = lambda p: total_flops / (p["time"] / 1000.0)

    # start tuning
    start = time.time()

    r, e = kernel_tuner.tune_kernel(
        "convolution_kernel",
        kernel_string,
        problem_size,
        args,
        tune_params,
        grid_div_y=grid_div_y,
        grid_div_x=grid_div_x,
        cmem_args=cmem_args,
        verbose=True,
        metrics=metrics,
        restrictions=restrict,
        iterations=32,
        cache="convolution_cache.json",
        strategy="brute_force",
        strategy_options={"max_fevals": 100},
    )

    end = time.time()
    e["execution_time"] = end - start

    return r, e


if __name__ == "__main__":
    w = h = 4096
    fw = fh = 15
    # total_flops = ops(w, h, fw, fh)

    results, env = tune([w, h, fw, fh])
    print(results)
    print(env)
