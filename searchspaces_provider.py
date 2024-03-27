"""Module for providing the tunable parameters and restrictions that make up searchspaces.

The following searchspaces are provided:
- the dedispersion kernel searchspace: dedispersion.
- the expdist kernel searchspace: expdist.
- a generated searchspace with parameters of choice: generate_searchspace.
- a function to generate a set of searchspace variants with parameters of choice: generate_searchspace_variants.
"""

from itertools import cycle
from math import ceil, floor, prod
from typing import Any, Tuple

import numpy as np


def get_searchspace_tuple(
    name: str, tune_params: dict[str, Any], restrictions: list[str]
) -> Tuple[dict[str, Any], list[str], int, int, int]:
    """Generate the info tuple for a searchspace.

    Args:
        name: the name of the searchspace to identify it in the cache.
        tune_params: _description_
        restrictions: _description_

    Returns:
        _description_
    """
    true_cartesian_size = prod(len(params) for params in tune_params.values())
    num_dimensions = len(tune_params.keys())
    num_restrictions = (
        len(restrictions)
        if isinstance(restrictions, (list, tuple))
        else 0 if restrictions is None else 1
    )
    return tuple(
        [
            tune_params,
            restrictions,
            num_dimensions,
            true_cartesian_size,
            num_restrictions,
            name,
        ]
    )


def dedispersion() -> Tuple[dict[str, Any], list[str]]:
    """The Dedispersion kernel searchspace as per https://github.com/benvanwerkhoven/benchmark_kernels/blob/master/dedisp/dedispersion.py.

    Returns:
        Tuple[dict[str, Any], list[str]]: the tuneable parameters and restrictions.
    """
    tune_params = dict()
    tune_params["block_size_x"] = [1, 2, 4, 8] + [16 * i for i in range(1, 3)]
    tune_params["block_size_y"] = [8 * i for i in range(4, 33)]
    tune_params["block_size_z"] = [1]
    tune_params["tile_size_x"] = [i for i in range(1, 5)]
    tune_params["tile_size_y"] = [i for i in range(1, 9)]
    tune_params["tile_stride_x"] = [0, 1]
    tune_params["tile_stride_y"] = [0, 1]
    tune_params["loop_unroll_factor_channel"] = [
        0
    ]  # + [i for i in range(1,nr_channels+1) if nr_channels % i == 0] #[i for i in range(nr_channels+1)]
    # tune_params["loop_unroll_factor_x"] = [0] #[i for i in range(1,max(tune_params["tile_size_x"]))]
    # tune_params["loop_unroll_factor_y"] = [0] #[i for i in range(1,max(tune_params["tile_size_y"]))]
    # tune_params["blocks_per_sm"] = [i for i in range(5)]

    check_block_size = "32 <= block_size_x * block_size_y <= 1024"

    check_tile_stride_x = "tile_size_x > 1 or tile_stride_x == 0"
    check_tile_stride_y = "tile_size_y > 1 or tile_stride_y == 0"

    config_valid = [check_block_size, check_tile_stride_x, check_tile_stride_y]

    restrictions = config_valid

    return get_searchspace_tuple("dedispersion", tune_params, restrictions)


def expdist(restrictions_type="strings") -> Tuple[dict[str, Any], list[str]]:
    """The ExpDist kernel searchspace as per https://github.com/benvanwerkhoven/benchmark_kernels/blob/master/expdist/expdist.py.

    Args:
        restrictions_type (str, optional): the type of the restrictions used. Either 'function', 'strings', 'constraints'. Defaults to 'strings'.

    Returns:
        Tuple[dict[str, Any], list[str]]: the tuneable parameters and restrictions.
    """
    tune_params = dict()
    tune_params["block_size_x"] = [2**i for i in range(5, 11)][::-1]  # 5
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
    # tune_params["n_y_blocks"] = [2**i for i in range(11)]

    # multiple definitions of the restrictions
    if restrictions_type == "function":

        def config_valid(p):
            if p["use_column"] == 0 and p["n_y_blocks"] > 1:
                return False
            if p["use_column"] == 0 and p["use_shared_mem"] == 2:
                return False
            if p["loop_unroll_factor_x"] > p["tile_size_x"] or (
                p["loop_unroll_factor_x"]
                and p["tile_size_x"] % p["loop_unroll_factor_x"] != 0
            ):  # TODO what is the purpose of the second loop_unroll_factor_x here?
                return False  # no need to test this loop unroll factor, as it is the same as not unrolling the loop
            if p["loop_unroll_factor_y"] > p["tile_size_y"] or (
                p["loop_unroll_factor_y"]
                and p["tile_size_y"] % p["loop_unroll_factor_y"] != 0
            ):
                return False  # no need to test this loop unroll factor, as it is the same as not unrolling the loop
            return True

        restrictions = config_valid
    elif restrictions_type == "strings":
        restrictions = [
            "use_column != 0 or n_y_blocks <= 1",
            "use_column != 0 or use_shared_mem != 2",
            "loop_unroll_factor_x <= tile_size_x and (tile_size_x % loop_unroll_factor_x == 0)",
            "loop_unroll_factor_y <= tile_size_y and (tile_size_y % loop_unroll_factor_y == 0)",
        ]
    else:
        raise ValueError(f"restrictions_type of undefined type {restrictions_type}")

    return get_searchspace_tuple("expdist", tune_params, restrictions)


def hotspot() -> Tuple[dict[str, Any], list[str]]:
    """The Hotspot kernel searchspace as per https://github.com/benvanwerkhoven/hotspot_kernel/blob/master/hotspot.py.

    Returns:
        Tuple[dict[str, Any], list[str]]: the tuneable parameters and restrictions.
    """
    # constants
    problem_size = (4096, 4096)

    # setup the tunable parameters
    tune_params = dict()
    # input sizes need at compile time
    tune_params["grid_width"] = [problem_size[0]]
    tune_params["grid_height"] = [problem_size[1]]
    # actual tunable parameters
    tune_params["block_size_x"] = [1, 2, 4, 8, 16] + [32 * i for i in range(1, 33)]
    tune_params["block_size_y"] = [2**i for i in range(6)]
    tune_params["tile_size_x"] = [i for i in range(1, 11)]
    tune_params["tile_size_y"] = [i for i in range(1, 11)]
    tune_params["temporal_tiling_factor"] = [i for i in range(1, 11)]
    max_tfactor = max(tune_params["temporal_tiling_factor"])
    tune_params["max_tfactor"] = [max_tfactor]
    tune_params["loop_unroll_factor_t"] = [i for i in range(1, max_tfactor + 1)]
    tune_params["sh_power"] = [0, 1]
    tune_params["blocks_per_sm"] = [0, 1, 2, 3, 4]

    # setup device properties (for A4000 on DAS6)
    dev = {
        "max_threads": 1024,
        "max_shared_memory_per_block": 49152,
        "max_shared_memory": 102400,
    }

    # setup the restrictions
    restrictions = [
        "block_size_x*block_size_y >= 32",
        "temporal_tiling_factor % loop_unroll_factor_t == 0",
        f"block_size_x*block_size_y <= {dev['max_threads']}",
        f"(block_size_x*tile_size_x + temporal_tiling_factor * 2) * (block_size_y*tile_size_y + temporal_tiling_factor * 2) * (2+sh_power) * 4 <= {dev['max_shared_memory_per_block']}",
        f"blocks_per_sm == 0 or (((block_size_x*tile_size_x + temporal_tiling_factor * 2) * (block_size_y*tile_size_y + temporal_tiling_factor * 2) * (2+sh_power) * 4) * blocks_per_sm <= {dev['max_shared_memory']})",
    ]

    return get_searchspace_tuple("hotspot", tune_params, restrictions)


def microhh(extra_tuning=True) -> Tuple[dict[str, Any], list[str]]:
    """The MicroHH kernel searchspace as per https://github.com/stijnh/microhh/blob/develop-stijn/kernel_tuner/helpers.py.

    Args:
        extra_tuning: whether to apply additional tuning parameters. Defaults to True.

    Returns:
        Tuple[dict[str, Any], list[str]]: the tuneable parameters and restrictions.
    """
    # constants
    cta_padding = 0  # default argument

    # setup the tunable parameters
    tune_params = dict()
    tune_params["BLOCK_SIZE_X"] = [1, 2, 4, 8, 16, 32, 128, 256]
    tune_params["BLOCK_SIZE_Y"] = [1, 2, 4, 8, 16, 32]
    tune_params["BLOCK_SIZE_Z"] = [1, 2]
    tune_params["STATIC_STRIDES"] = [0]
    tune_params["TILING_FACTOR_X"] = [1]
    tune_params["TILING_FACTOR_Y"] = [1]
    tune_params["TILING_FACTOR_Z"] = [1]
    tune_params["TILING_STRATEGY"] = [0]
    tune_params["REWRITE_INTERP"] = [0]
    tune_params["BLOCKS_PER_MP"] = [0]
    tune_params["LOOP_UNROLL_FACTOR_X"] = [1]
    tune_params["LOOP_UNROLL_FACTOR_Y"] = [1]
    tune_params["LOOP_UNROLL_FACTOR_Z"] = [1]

    # optionally add additional tuning parameters
    if extra_tuning:
        tune_params["BLOCK_SIZE_X"] = [1, 2, 4, 8, 16, 32, 128, 256, 512, 1024]
        tune_params["BLOCK_SIZE_Y"] = [1, 2, 4, 8, 16, 32]
        tune_params["BLOCK_SIZE_Z"] = [1, 2, 4]
        tune_params["TILING_FACTOR_X"] = [1, 2, 4, 8]
        tune_params["TILING_FACTOR_Y"] = [1, 2, 4]
        tune_params["TILING_FACTOR_Z"] = [1, 2, 4]
        tune_params["LOOP_UNROLL_FACTOR_X"] = tune_params["TILING_FACTOR_X"]  # [0, 1]
        tune_params["LOOP_UNROLL_FACTOR_Y"] = tune_params["TILING_FACTOR_Y"]  # [0, 1]
        tune_params["LOOP_UNROLL_FACTOR_Z"] = tune_params["TILING_FACTOR_Z"]  # [0, 1]
        tune_params["BLOCKS_PER_MP"] = [0, 1, 2, 3, 4]

    # setup device properties (for A4000 on DAS6)
    dev = {"max_threads_per_sm": 1024, "max_threads_per_block": 1536}

    # setup the restrictions
    restrictions = [
        f"BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * BLOCKS_PER_MP <= {dev['max_threads_per_sm']}",
        f"32 <= BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z <= {dev['max_threads_per_block']}",
        "LOOP_UNROLL_FACTOR_X == 0 or TILING_FACTOR_X % LOOP_UNROLL_FACTOR_X == 0",
        "LOOP_UNROLL_FACTOR_Y == 0 or TILING_FACTOR_Y % LOOP_UNROLL_FACTOR_Y == 0",
        "LOOP_UNROLL_FACTOR_Z == 0 or TILING_FACTOR_Z % LOOP_UNROLL_FACTOR_Z == 0",
        f"BLOCK_SIZE_X * TILING_FACTOR_X > {cta_padding}",
        f"BLOCK_SIZE_Y * TILING_FACTOR_Y > {cta_padding}",
        f"BLOCK_SIZE_Z * TILING_FACTOR_Z > {cta_padding}",
    ]

    return get_searchspace_tuple("microhh", tune_params, restrictions)


def atf_gaussian_convolution() -> Tuple[dict[str, Any], list[str]]:
    """The Gaussian Convolution kernel searchspace used in the ATF paper, as per https://gitlab.com/mdh-project/taco2020-atf/-/blob/master/evaluation/overall/ATF/gaussian.cpp?ref_type=heads.

    Returns:
        Tuple[dict[str, Any], list[str]]: the tuneable parameters and restrictions.
    """

    # constants
    H = 4096
    W = 4096

    # setup the tunable parameters
    tune_params = dict()
    tune_params["CACHE_L_CB"] = [0, 1]
    tune_params["CACHE_P_CB"] = [0, 1]
    tune_params["G_CB_RES_DEST_LEVEL"] = [2]
    tune_params["L_CB_RES_DEST_LEVEL"] = [2, 1, 0]
    tune_params["P_CB_RES_DEST_LEVEL"] = [2, 1, 0]

    tune_params["INPUT_SIZE_L_1"] = [H]
    tune_params["L_CB_SIZE_L_1"] = range(1, H + 1)
    tune_params["P_CB_SIZE_L_1"] = range(1, H + 1)
    tune_params["OCL_DIM_L_1"] = [0, 1]
    tune_params["NUM_WG_L_1"] = range(1, H + 1)
    tune_params["NUM_WI_L_1"] = range(1, H + 1)

    tune_params["INPUT_SIZE_L_2"] = [W]
    tune_params["L_CB_SIZE_L_2"] = range(1, W + 1)
    tune_params["P_CB_SIZE_L_2"] = range(1, W + 1)
    tune_params["OCL_DIM_L_2"] = [0, 1]
    tune_params["NUM_WG_L_2"] = range(1, W + 1)
    tune_params["NUM_WI_L_2"] = range(1, W + 1)

    tune_params["L_REDUCTION"] = [1]
    tune_params["P_WRITE_BACK"] = [0]
    tune_params["L_WRITE_BACK"] = [2]

    # setup the restrictions
    restrictions = [
        "L_CB_RES_DEST_LEVEL <= G_CB_RES_DEST_LEVEL",
        "P_CB_RES_DEST_LEVEL <= L_CB_RES_DEST_LEVEL",
        "INPUT_SIZE_L_1 % L_CB_SIZE_L_1 == 0",
        "L_CB_SIZE_L_1 % P_CB_SIZE_L_1 == 0",
        "(INPUT_SIZE_L_1 / L_CB_SIZE_L_1) % NUM_WG_L_1 == 0",
        "(L_CB_SIZE_L_1 / P_CB_SIZE_L_1) % NUM_WI_L_1 == 0",
        "NUM_WI_L_1 <= (INPUT_SIZE_L_1 + NUM_WG_L_1 - 1 / NUM_WG_L_1)",
        "INPUT_SIZE_L_2 % L_CB_SIZE_L_2 == 0",
        "L_CB_SIZE_L_2 % P_CB_SIZE_L_2 == 0",
        "OCL_DIM_L_2 != OCL_DIM_L_1",
        "(INPUT_SIZE_L_2 / L_CB_SIZE_L_2) % NUM_WG_L_2 == 0",
        "(L_CB_SIZE_L_2 / P_CB_SIZE_L_2) % NUM_WI_L_2 == 0",
        "NUM_WI_L_2 <= (INPUT_SIZE_L_2 + NUM_WG_L_2 - 1 / NUM_WG_L_2)",
    ]

    return get_searchspace_tuple("atf_gaussian_convolution", tune_params, restrictions)


def generate_searchspace(
    num_dimensions=3, cartesian_size=100000, num_restrictions=3, random_state=np.random
) -> Tuple[dict[str, Any], list[str]]:
    """Function to generate a searchspace given some parameters.

    Args:
        num_dimensions: number of dimensions the searchspaces needs to consist of. Defaults to 3.
        cartesian_size: the (approximate) Cartesian size of the search space (before restrictions). Defaults to 100000.
        num_restrictions: the number of randomly chosen restrictions to apply. Defaults to 3.
        random_state: a random state optionally passed to provide a fixed seed. Defaults to np.random.

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
            elif dim1 > dim2:
                restrictions.append(f"{dim1_written} > {dim2_written} / 2.0")
    restrictions_np = random_state.choice(
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
    random_seeds = cycle(
        [
            7301,
            1581,
            2517,
            5875,
            9494,
            6633,
            4385,
            2019,
            7114,
            1775,
            8227,
            9159,
            8252,
            9793,
            9867,
            9616,
            4698,
            6927,
            3986,
            9535,
        ]
    )  # generated with `random.randint(0, 10000, 20)`
    cartesian_sizes = list(
        round(max_cartesian_size / div) for div in [100, 50, 20, 10, 5, 2, 1]
    )

    # generate all searchspace variants
    searchspace_variants: list[Tuple[dict[str, Any], list[str], int, int, int]] = list()
    for num_dimensions in range(2, max_num_dimensions + 1):
        for cartesian_size in cartesian_sizes:
            for num_restrictions in range(1, max_num_dimensions):
                num_restrictions = floor(num_restrictions * 1.5)
                random_state = np.random.RandomState(next(random_seeds))
                tune_params, restrictions = generate_searchspace(
                    num_dimensions=num_dimensions,
                    cartesian_size=cartesian_size,
                    num_restrictions=num_restrictions,
                    random_state=random_state,
                )
                # print(f"Expected: {cartesian_size}, true: {true_cartesian_size}")
                info_tuple = get_searchspace_tuple(
                    "generated", tune_params, restrictions
                )
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
    return "_".join(words[int(i)] for i in str(number))


def adverserial_rounding(number) -> int:
    """Does an adverserial rounding, rounding to the second-nearest integer, except when it already is an integer.

    Args:
        number: the number to be rounded.

    Returns:
        The rounded number as an integer.
    """
    return floor(number) + round((ceil(number) - number))
