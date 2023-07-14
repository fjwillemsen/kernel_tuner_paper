"""Module for providing the tunable parameters and restrictions that make up searchspaces.

The following searchspaces are provided:
- the dedispersion kernel searchspace: dedispersion.
- the expdist kernel searchspace: expdist.
- a generated searchspace with parameters of choice: generate_searchspace.
- a function to generate a set of searchspace variants with parameters of choice: generate_searchspace_variants.
"""

from math import ceil, floor, prod
from typing import Any, Tuple

import numpy as np


def get_searchspace_tuple(name: str, tune_params: dict[str, Any], restrictions: list[str]) -> Tuple[dict[str, Any], list[str], int, int, int]:
    """Generate the info tuple for a searchspace.

    Args:
        name: the name of the searchspace to identify it in the cache.
        tune_params: _description_
        restrictions: _description_

    Returns:
        _description_
    """
    true_cartesian_size = prod(
        len(params) for params in tune_params.values()
    )
    num_dimensions = len(tune_params.keys())
    num_restrictions = len(restrictions) if isinstance(restrictions, (list, tuple)) else 0 if restrictions is None else 1
    return tuple(
        [
            tune_params,
            restrictions,
            num_dimensions,
            true_cartesian_size,
            num_restrictions,
            name
        ]
    )

def dedispersion() -> Tuple[dict[str, Any], list[str]]:
    """The Dedispersion kernel searchspace as per https://github.com/benvanwerkhoven/benchmark_kernels/blob/master/dedisp/dedispersion.py.

    Returns:
        Tuple[dict[str, Any], list[str]]: the tuneable parameters and restrictions.
    """
    tune_params = dict()
    tune_params["block_size_x"] = [1, 2, 4, 8] + [16*i for i in range(1,3)]
    tune_params["block_size_y"] = [8*i for i in range(4,33)]
    tune_params["block_size_z"] = [1]
    tune_params["tile_size_x"] = [i for i in range(1,5)]
    tune_params["tile_size_y"] = [i for i in range(1,9)]
    tune_params["tile_stride_x"] = [0, 1]
    tune_params["tile_stride_y"] = [0, 1]
    # tune_params["loop_unroll_factor_x"] = [0] #[i for i in range(1,max(tune_params["tile_size_x"]))]
    # tune_params["loop_unroll_factor_y"] = [0] #[i for i in range(1,max(tune_params["tile_size_y"]))]
    tune_params["loop_unroll_factor_channel"] = [0] #+ [i for i in range(1,nr_channels+1) if nr_channels % i == 0] #[i for i in range(nr_channels+1)]
    # tune_params["blocks_per_sm"] = [i for i in range(5)]


    check_block_size = "32 <= block_size_x * block_size_y <= 1024"

    check_tile_stride_x = "tile_size_x > 1 or tile_stride_x == 0"
    check_tile_stride_y = "tile_size_y > 1 or tile_stride_y == 0"

    config_valid = [check_block_size, check_tile_stride_x, check_tile_stride_y]

    restrictions = config_valid

    return get_searchspace_tuple("dedispersion", tune_params, restrictions)

def expdist(restrictions_type = "strings") -> Tuple[dict[str, Any], list[str]]:
    """The ExpDist kernel searchspace as per https://github.com/benvanwerkhoven/benchmark_kernels/blob/master/expdist/expdist.py.

    Args:
        restrictions_type (str, optional): the type of the restrictions used. Either 'function', 'strings', 'constraints'. Defaults to 'strings'.

    Returns:
        Tuple[dict[str, Any], list[str]]: the tuneable parameters and restrictions.
    """
    tune_params = dict()
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

    # multiple definitions of the restrictions
    if restrictions_type == 'function':
        def config_valid(p):
            if p["use_column"] == 0 and p["n_y_blocks"] > 1:
                return False
            if p["use_column"] == 0 and p["use_shared_mem"] == 2:
                return False
            if p["loop_unroll_factor_x"] > p["tile_size_x"] or (p["loop_unroll_factor_x"] and p["tile_size_x"] % p["loop_unroll_factor_x"] != 0):   # TODO what is the purpose of the second loop_unroll_factor_x here?
                return False #no need to test this loop unroll factor, as it is the same as not unrolling the loop
            if p["loop_unroll_factor_y"] > p["tile_size_y"] or (p["loop_unroll_factor_y"] and p["tile_size_y"] % p["loop_unroll_factor_y"] != 0):
                return False #no need to test this loop unroll factor, as it is the same as not unrolling the loop
            return True
        restrictions = config_valid
    elif restrictions_type == 'strings':
        restrictions = [
            "use_column != 0 or n_y_blocks <= 1",
            "use_column != 0 or use_shared_mem != 2",
            "loop_unroll_factor_x <= tile_size_x and (tile_size_x % loop_unroll_factor_x == 0)",
            "loop_unroll_factor_y <= tile_size_y and (tile_size_y % loop_unroll_factor_y == 0)"
        ]
    # TODO
    elif restrictions_type == 'constraints':
        pass
    else:
        raise ValueError(f"restrictions_type of undefined type {restrictions_type}")

    return get_searchspace_tuple("expdist", tune_params, restrictions)

def hotspot() -> Tuple[dict[str, Any], list[str]]:
    """The Hotspot kernel searchspace as per https://github.com/benvanwerkhoven/hotspot_kernel/blob/master/hotspot.py.

    Returns:
        Tuple[dict[str, Any], list[str]]: the tuneable parameters and restrictions.
    """
    problem_size = (4096, 4096)

    # setup the tunable parameters
    tune_params = dict()
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

    # setup device properties (for A4000 on DAS6)
    dev = {'max_threads': 1024, 'max_shared_memory_per_block': 49152, 'max_shared_memory': 102400}

    # setup the restrictions
    restrictions = ["block_size_x*block_size_y >= 32",
                    "temporal_tiling_factor % loop_unroll_factor_t == 0",
                    f"block_size_x*block_size_y <= {dev['max_threads']}",
                    f"(block_size_x*tile_size_x + temporal_tiling_factor * 2) * (block_size_y*tile_size_y + temporal_tiling_factor * 2) * (2+sh_power) * 4 <= {dev['max_shared_memory_per_block']}",
                    f"blocks_per_sm == 0 or (((block_size_x*tile_size_x + temporal_tiling_factor * 2) * (block_size_y*tile_size_y + temporal_tiling_factor * 2) * (2+sh_power) * 4) * blocks_per_sm <= {dev['max_shared_memory']})"]

    return get_searchspace_tuple("hotspot", tune_params, restrictions)

def generate_searchspace(
    num_dimensions=3, cartesian_size=100000, num_restrictions=9
) -> Tuple[dict[str, Any], list[str]]:
    """Function to generate a searchspace given some parameters.

    Args:
        num_dimensions: number of dimensions the searchspaces needs to consist of. Defaults to 3.
        cartesian_size: the (approximate) Cartesian size of the search space (before restrictions). Defaults to 100000.
        num_restrictions: the number of randomly chosen restrictions to apply. Defaults to 9.

    Returns:
        A tuple of the tunable parameters dictionary and the list of restrictions.
    """
    start_num = -500
    end_num = 500
    quarter_num = (end_num - start_num) * 0.25 + start_num

    # generate tunable parameters
    tune_params: dict[str, Any] = dict()
    params_per_dimension = cartesian_size ** (1 / num_dimensions)
    for dim in range(num_dimensions):
        # on the last dimension, do an adverserial rounding to make the true size closer to the Cartesian size parameter
        if dim == num_dimensions - 1:
            params_per_dimension = adverserial_rounding(params_per_dimension)
        tune_params[number_to_words(dim)] = np.linspace(
            start_num, end_num, num=max(1, round(params_per_dimension))
        )

    # generate restrictions
    restrictions = list()
    for dim1 in range(num_dimensions):
        for dim2 in range(num_dimensions):
            dim1_written = number_to_words(dim1)
            dim2_written = number_to_words(dim2)
            if dim1 == dim2:
                restrictions.append(f"{dim1_written} <= {quarter_num}")
            elif dim1 < dim2:
                restrictions.append(f"{dim1_written} * {dim2_written} >= {quarter_num}")
    restrictions_np = np.random.choice(
        restrictions, size=min(num_restrictions, len(restrictions)), replace=False
    )
    restrictions: list[str] = restrictions_np.tolist()

    return tune_params, restrictions

def generate_searchspace_variants(
    max_num_dimensions=5, max_cartesian_size=1000000
) -> list[Tuple[dict[str, Any], list[str], int, int, int]]:
    """Generates a set of various search spaces.

    Args:
        max_num_dimensions (int, optional): the maximum number of dimensions. Defaults to 5.
        max_cartesian_size (int, optional): the approximate Cartesian size of the largest search space. Defaults to 1,000,000.

    Returns:
        list[Tuple[dict[str, Any], list[str], int, int, int]]: list of tuples of the tuneable parameters, restrictions, number of dimensions, true cartesian size, number of restrictions.
    """
    cartesian_sizes = list(
        round(max_cartesian_size / div) for div in [1000, 100, 50, 10, 5, 2, 1]
    )

    # generate all searchspace variants
    searchspace_variants: list[Tuple[dict[str, Any], list[str], int, int, int]] = list()
    for num_dimensions in range(2, max_num_dimensions + 1):
        for cartesian_size in cartesian_sizes:
            for num_restrictions in range(max_num_dimensions):
                tune_params, restrictions = generate_searchspace(
                    num_dimensions=num_dimensions,
                    cartesian_size=cartesian_size,
                    num_restrictions=num_restrictions,
                )
                # print(f"Expected: {cartesian_size}, true: {true_cartesian_size}")
                info_tuple = get_searchspace_tuple("generated", tune_params, restrictions)
                assert num_dimensions == info_tuple[2]
                assert min(num_restrictions, len(restrictions)) == info_tuple[4]
                searchspace_variants.append(info_tuple)
    return searchspace_variants

def number_to_words(number: int) -> str:
    """Converts an integer number to the written-out number.

    Args:
        number (int): the input number.

    Returns:
        str: the written-out number.
    """
    words = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    ]
    return " ".join(words[int(i)] for i in str(number))

def adverserial_rounding(number) -> int:
    """Does an adverserial rounding, rounding to the second-nearest integer, except when it already is an integer.

    Args:
        number: the number to be rounded.

    Returns:
        The rounded number as an integer.
    """
    return floor(number) + round((ceil(number) - number))
