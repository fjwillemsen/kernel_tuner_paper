import argparse
import math
import os
import re
from subprocess import PIPE, run

import kernel_tuner
import numpy as np
import pycuda.driver as drv

drv.init()

# from kernel_tuner.nvml import nvml


def get_device_name(device):
    return drv.Device(device).name().replace(" ", "_")


def get_pycuda_cuda_version() -> tuple:
    """Returns the CUDA version PyCUDA was installed against as a three-digit tuple (major, minor, fix)."""
    return drv.get_version()


def get_pycuda_cuda_version_string() -> str:
    """Returns the CUDA version PyCUDA was installed against as a string."""
    return ".".join(list(str(d) for d in get_pycuda_cuda_version()))


def get_nvcc_cuda_version_string() -> str:
    """Returns the CUDA version reported by NVCC as a string."""
    nvcc_output: str = run(["nvcc", "--version"], stdout=PIPE).stdout.decode("utf-8")
    nvcc_output = "".join(
        nvcc_output.splitlines()
    )  # convert to single string for easier REGEX
    cuda_version = (
        re.match(r"^.*release ([0-9]+.[0-9]+).*$", nvcc_output, flags=re.IGNORECASE)
        .group(1)
        .strip()
    )
    return cuda_version


def check_pycuda_version_matches_cuda() -> bool:
    """Checks whether the CUDA version PyCUDA was installed with matches the current CUDA version."""
    pycuda_version = get_pycuda_cuda_version_string()
    current_cuda_version = get_nvcc_cuda_version_string()
    shortest_string, longest_string = (
        (pycuda_version, current_cuda_version)
        if len(pycuda_version) < len(current_cuda_version)
        else (current_cuda_version, pycuda_version)
    )
    return longest_string[: len(shortest_string)] == shortest_string


def get_fallback():
    if os.uname()[1].startswith("node0"):
        return "/cm/shared/package/utils/bin/run-nvidia-smi"
    return "nvidia-smi"


def get_metrics(total_flops):
    metrics = dict()
    metrics["GFLOP/s"] = lambda p: total_flops / (p["time"] / 1000.0)
    # metrics["GFLOPS/W"] = lambda p: total_flops / p["nvml_energy"]
    return metrics


def get_pwr_limits(device, n=None):
    d = nvml(device)
    power_limits = d.pwr_constraints
    power_limit_min = power_limits[0]
    power_limit_max = power_limits[-1]
    power_limit_min *= 1e-3  # Convert to Watt
    power_limit_max *= 1e-3  # Convert to Watt
    power_limit_round = 5
    tune_params = dict()
    if n is None:
        n = int((power_limit_max - power_limit_min) / power_limit_round)

    # Rounded power limit values
    power_limits = power_limit_round * np.round(
        (np.linspace(power_limit_min, power_limit_max, n) / power_limit_round)
    )
    power_limits = list(set([int(power_limit) for power_limit in power_limits]))
    tune_params["nvml_pwr_limit"] = power_limits
    print("Using power limits:", tune_params["nvml_pwr_limit"])
    return tune_params


def get_supported_mem_clocks(device, n=None):
    d = nvml(device)
    mem_clocks = d.supported_mem_clocks

    if n and len(mem_clocks) > n:
        mem_clocks = mem_clocks[:: int(len(mem_clocks) / n)]

    tune_params = dict()
    tune_params["nvml_mem_clock"] = mem_clocks
    print("Using mem frequencies:", tune_params["nvml_mem_clock"])
    return tune_params


def get_gr_clocks(device, n=None):
    d = nvml(device)
    mem_clock = max(d.supported_mem_clocks)
    gr_clocks = d.supported_gr_clocks[mem_clock]

    if n and (len(gr_clocks) > n):
        gr_clocks = gr_clocks[:: math.ceil(len(gr_clocks) / n)]

    tune_params = dict()
    tune_params["nvml_gr_clock"] = gr_clocks[::-1]
    print("Using clock frequencies:", tune_params["nvml_gr_clock"])
    return tune_params


def get_default_parser():
    parser = argparse.ArgumentParser(description="Tune kernel")
    parser.add_argument("-d", dest="device", nargs="?", default=0, help="GPU ID to use")
    parser.add_argument(
        "-f",
        dest="overwrite",
        action="store_true",
        help="Overwrite any existing .json files",
    )
    parser.add_argument("--suffix", help="Suffix to append to output file names")
    parser.add_argument("--tune-power-limit", action="store_true")
    parser.add_argument("--power-limit-steps", nargs="?")
    parser.add_argument("--tune-gr-clock", action="store_true")
    parser.add_argument("--gr-clock-steps", nargs="?")
    return parser


def report_most_efficient(results, tune_params, metrics):
    best_config = min(results, key=lambda x: x["nvml_energy"])
    print("most efficient configuration:")
    kernel_tuner.util.print_config_output(
        tune_params, best_config, quiet=False, metrics=metrics, units=None
    )
