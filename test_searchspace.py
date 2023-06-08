"""A script to test Searchspace initialization times using various search spaces."""

from time import perf_counter
from typing import Tuple, Any
from math import prod, floor, ceil
from platform import node as get_hostname
import pickle

import matplotlib.pyplot as plt
import numpy as np
import progressbar

from kernel_tuner.searchspace import Searchspace
from kernel_tuner.util import compile_restrictions


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


def get_cache_filename() -> str:
    """Gets the filename of the cachefile using the hostname to identify devices.

    Raises:
        ValueError: if no hostname is found.

    Returns:
        str: the filename of the cache to use.
    """
    machinename = get_hostname()
    if len(machinename) <= 0:
        raise ValueError("No hostname found")
    # cut off the '.local' part at the end
    if machinename[-6:] == ".local":
        machinename = machinename[:-6]
    # replace spaces and dashes with underscores
    machinename = machinename.replace(" ", "_")
    machinename = machinename.replace("-", "_")
    return f"searchspaces_results_cache_{machinename}.pkl"


def write_to_cache(dictionary: dict[str, Any]):
    """Write the given dictionary to a pickled file.

    Args:
        dictionary (dict[str, Any]): the dictionary to write.
    """
    with open(get_cache_filename(), "wb") as f:
        pickle.dump(dictionary, f)


def read_from_cache() -> dict[str, Any]:
    """Read a dictionary object from the pickled cache file.

    Returns:
        dict[str, Any]: _description_
    """
    with open(get_cache_filename(), "rb") as f:
        return pickle.load(f)


def searchspace_variant_to_key(searchspace_variant: tuple, index: int) -> str:
    """Generate a unique key for a searchspace variant to use as a hash.

    Args:
        searchspace_variant (tuple): uses the `num_dimensions`, `cartesian_size` and `num_restrictions` as part of the unique key.
        index (int): uses the index as part of the unique key.

    Returns:
        str: the unique key.
    """
    (
        _,
        _,
        num_dimensions,
        cartesian_size,
        num_restrictions,
    ) = searchspace_variant
    key = ",".join(
        str(x)
        for x in [
            index,
            num_dimensions,
            cartesian_size,
            num_restrictions,
        ]
    )
    return key


def run_searchspace_initialization(
    tune_params, restrictions, kwargs=dict()
) -> Searchspace:
    # if there are strings in the restrictions, parse them to functions (increases restrictions check performance significantly)
    if (
        isinstance(restrictions, list)
        and len(restrictions) > 0
        and any(isinstance(restriction, str) for restriction in restrictions)
    ):
        restrictions = compile_restrictions(restrictions, tune_params)
    ss = Searchspace(
        tune_params=tune_params, restrictions=restrictions, max_threads=1, **kwargs
    )
    return ss


def searchspace_initialization(
    tune_params, restrictions, method: str
) -> Tuple[float, int]:
    """Tests the duration of the search space object initialization for a given set of parameters and restrictions and a method.

    Args:
        tune_params: a dictionary of tunable parameters.
        restrictions: restrictions to apply to the tunable parameters.
        method (str): the method with which to initialize the searchspace.

    Returns:
        A tuple of the total time taken by the search space initialization and the true size of the search space.
    """
    # get the keyword arguments
    if method == "default":
        kwargs = {}
    else:
        kwargs = {}
        for kwarg in method.split(","):
            keyword, argument = tuple(kwarg.split("="))
            kwargs[keyword] = argument

    # initialize and track the performance
    start_time = perf_counter()
    ss = run_searchspace_initialization(tune_params, restrictions, kwargs)
    time_taken = perf_counter() - start_time
    return time_taken, ss.size


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
        list[Tuple[dict[str, Any], list[str], int, int, int]]: _description_
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
                true_cartesian_size = prod(
                    len(params) for params in tune_params.values()
                )
                # print(f"Expected: {cartesian_size}, true: {true_cartesian_size}")
                tup = tuple(
                    [
                        tune_params,
                        restrictions,
                        num_dimensions,
                        true_cartesian_size,
                        num_restrictions,
                    ]
                )
                searchspace_variants.append(tup)
    return searchspace_variants


def run(num_repeats=3) -> dict[str, Any]:
    """Run the search space variants or retrieve them from cache.

    Args:
        num_repeats (int, optional): the number of times each search space variant is repeated. Defaults to 3.

    Returns:
        dict[str, Any]: the search space variants results.
    """

    # get cached results if available
    try:
        searchspaces_results = read_from_cache()
    except FileNotFoundError:
        print(f"Cachefile '{get_cache_filename()}' not found, creating...")
        searchspaces_results = dict()

    # run or retrieve from cache all searchspace variants
    for searchspace_variant_index in progressbar.progressbar(
        range(len(searchspace_variants)),
        redirect_stdout=True,
        prefix=" | - |-> running: ",
        widgets=[
            progressbar.PercentageLabelBar(),
            " [",
            progressbar.SimpleProgress(format="%(value_s)s/%(max_value_s)s"),
            ", ",
            progressbar.Timer(format="Elapsed: %(elapsed)s"),
            ", ",
            progressbar.ETA(),
            "]",
        ],
    ):
        # get the searchspace variant details
        searchspace_variant = searchspace_variants[searchspace_variant_index]
        (
            tune_params,
            restrictions,
            num_dimensions,
            cartesian_size,
            num_restrictions,
        ) = searchspace_variant

        # check if the searchspace variant is in the cache, if not, run it
        key = searchspace_variant_to_key(
            searchspace_variant, index=searchspace_variant_index
        )
        if key not in searchspaces_results:
            # run the variant
            results = dict()
            for method in searchspace_methods:
                times_in_seconds = list()
                true_sizes = list()
                for _ in range(num_repeats):
                    time_in_seconds, true_size = searchspace_initialization(
                        tune_params=tune_params,
                        restrictions=restrictions,
                        method=method,
                    )
                    times_in_seconds.append(time_in_seconds)
                    true_sizes.append(true_size)
                results[method] = dict(
                    {"time_in_seconds": times_in_seconds, "true_size": true_sizes}
                )

            # write the results to the cache
            searchspaces_results[key] = dict(
                {
                    "tune_params": tune_params,
                    "restrictions": restrictions,
                    "num_dimensions": num_dimensions,
                    "cartesian_size": cartesian_size,
                    "num_restrictions": num_restrictions,
                    "results": results,
                }
            )

    # write the results to the cache
    write_to_cache(searchspaces_results)

    return searchspaces_results


def visualize(searchspaces_results: dict[str, Any], project_3d=False):
    """Visualize the results of search spaces in a plot.

    Args:
        searchspaces_results (dict[str, Any]): the cached results dictionary.
    """
    # setup visualization
    if project_3d:
        plt.style.use("_mpl-gallery")
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 14))
    else:
        fig, ax = plt.subplots(nrows=2, figsize=(8, 14))

    # loop over each method
    for method in searchspace_methods:
        # setup arrays
        x = list()  # cartesian size
        y = list()  # true size after restrictions
        z = list()  # time taken in seconds

        # retrieve the data from the results dictionary
        for searchspace_variant_index, searchspace_variant in enumerate(
            searchspace_variants
        ):
            cartesian_size = searchspace_variant[3]
            key = searchspace_variant_to_key(
                searchspace_variant, searchspace_variant_index
            )
            searchspace_result = searchspaces_results[key]
            results = searchspace_result["results"][method]

            # get the results
            time_in_seconds = np.median(results["time_in_seconds"])
            true_size = round(np.mean(results["true_size"]))

            # write to the arrays
            x.append(cartesian_size)
            y.append(1 - (true_size / cartesian_size))
            # y.append(num_restrictions)
            z.append(time_in_seconds)

        # clean up data
        X = np.array(x)
        Y = np.array(y)
        Z = np.array(z)

        # plot
        if project_3d:
            ax.scatter(X, Y, Z)
        else:
            ax[0].scatter(X, Z)
            ax[1].scatter(Y, Z)

    # set labels and axis
    if project_3d:
        # ax.set_xscale("log")
        ax.set_xlabel("Cartesian size (approx. number of configs before restrictions)")
        ax.set_ylabel("Percentage of search space restricted")
        # ax.set_ylabel("Number of restrictions")
        ax.set_zlabel("Time in seconds")
        # ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        # ax.set_xticks(np.arange(np.min(X), np.max(X), np.max(X) / 10))
    else:
        ax[0].set_xlabel(
            "Cartesian size (approx. number of configs before restrictions)"
        )
        ax[1].set_xlabel("Percentage of search space restricted")
        ax[0].set_ylabel("Time in seconds")
        ax[1].set_ylabel("Time in seconds")

    # finish plot setup
    fig.tight_layout()
    plt.show()


searchspace_variants = generate_searchspace_variants(max_cartesian_size=1000000)
searchspace_methods = [
    "default"
]  # must be either 'default' or a kwargs-string passed to Searchspace (e.g. "build_neighbors_index=5,neighbor_method='adjacent'")


def main():
    """Entry point for execution."""
    searchspaces_results = run()
    visualize(searchspaces_results)


if __name__ == "__main__":
    main()
