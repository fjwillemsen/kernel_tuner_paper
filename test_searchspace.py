"""A script to test Searchspace initialization times using various search spaces."""

import pickle
from inspect import signature
from itertools import product
from os import execv
from platform import machine, system
from subprocess import DEVNULL, STDOUT, check_call
from sys import argv, executable
from time import perf_counter
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import progressbar
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.util import check_restrictions, compile_restrictions, default_block_size_names
from psutil import cpu_count, virtual_memory

from searchspaces_provider import dedispersion, expdist, generate_searchspace_variants, hotspot, microhh


def test_package_version_is_old() -> bool:
    """Tests whether the old, unoptimized packages or the new, optimized packages are installed.

    Returns:
        whether the old packages are installed.
    """
    try:
        from constraint import check_if_compiled
        check_if_compiled()
        from kernel_tuner.searchspace import Searchspace
        if "solver_method" not in str(signature(Searchspace)):
            raise ImportError()
        return False
    except ImportError:
        return True

default_max_threads = 1024
installed_unoptimized = test_package_version_is_old()


def switch_packages_to(old=True) -> bool:
    """Function to switch between the old and the optimized packages. Reloads imports by restarting the script, so be careful of loops.

    Args:
        old: Whether to load the old packages (if True) or the optimized ones (if False). Defaults to True.

    Returns:
        whether the old packages are installed (True) or the new packages are installed (False).
    """
    # check whether switching is necessary, if not return immediately
    old_installed = test_package_version_is_old()
    if old_installed and old:
        return True
    if not old_installed and not old:
        return False

    # install the new packages
    print("")
    if old:
        print("Switching from new to old packages")
        check_call(['sh', 'switch_packages_old.sh'], stdout=DEVNULL, stderr=STDOUT)
    else:
        print("Switching from old to new packages")
        check_call(['sh', 'switch_packages_optimized.sh'], stdout=DEVNULL, stderr=STDOUT)

    print(f"Restarting after installing {'old' if old else 'optimized'} packages")

    # restart this script entirely to reload the imports correctly
    execv(executable, ['python'] + argv)


def get_machine_info() -> str:
    """Generates a string of device information.

    Returns:
        str: the device information, formatted as {architecture}_{system}_{core count}_{RAM size in GB}.
    """
    arch = ("Arch", machine())
    sys = ("Sys", system())
    cpus = ("CPUs", str(cpu_count()))
    ram = ("RAM", str(round(virtual_memory().total / 1024 / 1024 / 1024)))  # in GB
    return "_".join(
        f"{k}={v}"
        for k, v in filter(
            lambda s: len(str(s)) > 0,
            [arch, sys, cpus, ram],
        )
    )


def get_cache_filename() -> str:
    """Gets the filename of the cachefile using the hostname to identify devices.

    Raises:
        ValueError: if no hostname is found.

    Returns:
        str: the filename of the cache to use.
    """
    machinename = get_machine_info()
    if len(machinename) <= 0:
        raise ValueError("No system info found")
    # replace spaces and dashes with underscores
    machinename = machinename.replace(" ", "_").replace("-", "_")
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
    """Generate a unique key for a searchspace variant to use as a hash for the cache.

    Args:
        searchspace_variant (tuple): uses the `num_dimensions`, `cartesian_size` and `num_restrictions` for the key.
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
        name
    ) = searchspace_variant
    key = ",".join(
        str(x)
        for x in [
            name,
            index,
            num_dimensions,
            cartesian_size,
            num_restrictions,
        ]
    )
    return key


def run_searchspace_initialization(
    tune_params, restrictions, framework: str, kwargs={}
) -> Searchspace:
    # initialize the searchspace
    ss = Searchspace(
        tune_params=tune_params,
        restrictions=restrictions,
        max_threads=default_max_threads,
        **kwargs,
    )
    return ss

def bruteforce_searchspace(tune_params: dict, restrictions: list, max_threads = default_max_threads) -> list[tuple]:
    """Bruteforce solving a searchspace (can take a long time depending on input!).

    Args:
        tune_params: a dictionary of tunable parameters.
        restrictions: restrictions to apply to the tunable parameters.

    Returns:
        The resulting list of configurations.
    """
    # compute cartesian product of all tunable parameters
    parameter_space = product(*tune_params.values())

    # check if there are block sizes in the parameters, if so add default restrictions
    used_block_size_names = list(
        block_size_name
        for block_size_name in default_block_size_names
        if block_size_name in tune_params
    )
    if len(used_block_size_names) > 0:
        if not isinstance(restrictions, list):
            restrictions = [restrictions]
        restrictions.append(f"{' * '.join(used_block_size_names)} <= {max_threads}")

    # check for search space restrictions
    if restrictions is not None:
        parameter_space = filter(lambda p: check_restrictions(restrictions, dict(zip(tune_params.keys(), p)), False), parameter_space)
    return list(parameter_space)

def assert_searchspace_validity(bruteforced: list[tuple], searchspace: Searchspace):
    """Asserts that the given searchspace has the same outcome as the bruteforced list of configurations."""
    assert searchspace.size == len(bruteforced), f"Lengths differ: {searchspace.size} != {len(bruteforced)}"
    for config in bruteforced:
        assert searchspace.is_param_config_valid(config), f"Config '{config}' is in the bruteforced searchspace but not in the evaluated searchspace."

def restrictions_strings_to_function(restrictions: list, tune_params: dict):
    """Parses a list of strings to a monolithic function.

    Args:
        restrictions: a list of string restrictions.
        tune_params: dictionary of tunable parameters.

    Raises:
        ValueError: if not a list of strings.

    Returns:
        the restriction function.
    """
    # check whether the correct types of restrictions have been passed
    if not isinstance(restrictions, list):
        raise ValueError(f"Not a list of restrictions: {type(restrictions)}; {restrictions}")
    for r in restrictions:
        if not isinstance(r, str):
            raise ValueError(f"Non-string restriction {type(r)}; {r}")

    return compile_restrictions(restrictions, tune_params)

def searchspace_initialization(
    tune_params, restrictions, method: str
) -> Tuple[float, int, Searchspace]:
    """Tests the duration of the search space object initialization for a given set of parameters and restrictions and a method.

    Args:
        tune_params: a dictionary of tunable parameters.
        restrictions: restrictions to apply to the tunable parameters.
        method (str): the method with which to initialize the searchspace.

    Returns:
        A tuple of the total time taken by the search space initialization, the true size of the search space, and the Searchspace object.
    """
    if callable(restrictions) or ((isinstance(restrictions, list) and len(restrictions) > 0 and callable(restrictions[0]))):
        raise ValueError("Function restrictions can't be pickled")

    # get the keyword arguments
    unoptimized = False
    if method == "default":
        kwargs = {}
        framework = 'PythonConstraint'
    else:
        kwargs = {}
        for kwarg in method.split(","):
            keyword, argument = tuple(kwarg.split("="))
            if argument.lower() in ['true', 'false']:
                argument = True if argument.lower() == 'true' else False
            if keyword.lower() == 'unoptimized':
                unoptimized = True
                continue
            kwargs[keyword] = argument
        if 'framework' in kwargs:
            framework = kwargs["framework"]

    # install the old (unoptimized) packages if necessary
    global installed_unoptimized
    if unoptimized:
        if not installed_unoptimized:
            installed_unoptimized = switch_packages_to(old=True)
        # kwargs are dropped for old KernelTuner & PythonConstraint packages
        kwargs = {}
        framework = 'Old'
        # convert restrictions from list of string to function
        if isinstance(restrictions, list) and all(isinstance(r, str) for r in restrictions):
            restrictions = restrictions_strings_to_function(restrictions, tune_params)
    elif installed_unoptimized:
        # re-install the new (optimized) packages if we previously installed the old packages
        installed_unoptimized = switch_packages_to(old=False)

    # initialize and track the performance
    start_time = perf_counter()
    ss = run_searchspace_initialization(
        tune_params, restrictions, framework=framework, kwargs=kwargs
    )
    time_taken = perf_counter() - start_time

    # return the time taken in seconds, the searchspace size, and the Searchspace object.
    return time_taken, ss.size, ss


def run(num_repeats=3, validate_results=True) -> dict[str, Any]:
    """Run the search space variants or retrieve them from cache.

    Args:
        num_repeats (int, optional): the number of times each search space variant is repeated. Defaults to 3.

    Returns:
        dict[str, Any]: the search space variants results.
    """
    global searchspaces_ignore_cache, searchspace_methods_ignore_cache

    # run each searchspace method
    for method_index, method in enumerate(searchspace_methods):

        # get cached results if available
        try:
            searchspaces_results = read_from_cache()
        except FileNotFoundError:
            print(f"Cachefile '{get_cache_filename()}' not found, creating...")
            searchspaces_results = dict()

        # run or retrieve from cache all searchspace variants
        for searchspace_variant_index in progressbar.progressbar(
            range(len(searchspaces)),
            redirect_stdout=True,
            prefix=" |-> running: ",
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
            searchspace_variant = searchspaces[searchspace_variant_index]
            (
                tune_params,
                restrictions,
                num_dimensions,
                cartesian_size,
                num_restrictions,
                searchspace_name
            ) = searchspace_variant

            # run the bruteforce to validate the results against
            if validate_results:
                bruteforced = bruteforce_searchspace(tune_params, restrictions)
                # TODO can be made more efficient by saving the bruteforced to a separate cache

            # check if the searchspace variant is in the cache, if not, run it
            key = searchspace_variant_to_key(searchspace_variant, index=searchspace_variant_index)
            if (
                key not in searchspaces_results
                or len(searchspaces_ignore_cache) > 0
                or len(searchspace_methods_ignore_cache) > 0
                or not all(
                    method in searchspaces_results[key]["results"]
                    for method in searchspace_methods
                )
            ):
                # run the variant
                results = (
                    searchspaces_results[key]["results"]
                    if key in searchspaces_results
                    else dict()
                )
                if (
                    method in results
                    and method_index not in searchspace_methods_ignore_cache
                    and searchspace_variant_index not in searchspaces_ignore_cache
                ):
                    continue
                times_in_seconds = list()
                true_sizes = list()
                for _ in range(num_repeats):
                    time_in_seconds, true_size, searchspace = searchspace_initialization(
                        tune_params=tune_params,
                        restrictions=restrictions,
                        method=method,
                    )
                    times_in_seconds.append(time_in_seconds)
                    true_sizes.append(true_size)
                results[method] = dict(
                    {"time_in_seconds": times_in_seconds, "true_size": true_sizes}
                )
                if validate_results:
                    assert_searchspace_validity(bruteforced, searchspace)

                # write the results to the cache
                searchspaces_results[key] = dict(
                    {
                        "name": searchspace_name,
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
        # reset the ignore_caches to avoid infinite loops
        searchspaces_ignore_cache = []
        searchspace_methods_ignore_cache = []

    return searchspaces_results


def visualize(
    searchspaces_results: dict[str, Any], project_3d=False, show_overall=True, log_scale=True,
):
    """Visualize the results of search spaces in a plot.

    Args:
        searchspaces_results (dict[str, Any]): the cached results dictionary.
        project_3d (bool, optional): whether to visualize as one 3D or two 2D plots. Defaults to False.
        show_overall (bool, optional): whether to also plot overall performance between methods. Defaults to True.
        log_scale (bool, optional): whether to plot time on a logarithmic scale instead of default. Defaults to True.
    """
    # setup visualization
    if project_3d:
        plt.style.use("_mpl-gallery")
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 14))
    else:
        fig, ax = plt.subplots(nrows=2, figsize=(8, 14))

    # loop over each method
    sums = list()
    means = list()
    medians = list()
    stds = list()
    last_y = list()
    speedup_per_searchspace_median = list()
    speedup_per_searchspace_std = list()
    speedup_baseline_data = None
    for method_index, method in enumerate(searchspace_methods):
        # setup arrays
        x = list()  # cartesian size
        y = list()  # fraction of cartesian size after restrictions
        y_1 = list()  # true size after restrictions
        z = list()  # time taken in seconds

        # retrieve the data from the results dictionary
        for searchspace_variant_index, searchspace_variant in enumerate(
            searchspaces
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
            y_1.append(true_size)
            z.append(time_in_seconds)

        # clean up data
        X = np.array(x)
        Y = np.array(y)
        np.array(y_1)
        Z = np.array(z)

        # add statistical data for reporting
        data = Z
        sums.append(np.sum(data))
        means.append(np.mean(data))
        medians.append(np.median(data))
        stds.append(np.std(data))
        last_y.append(data[-1])

        # calculate speedups relative to baseline
        if speedup_baseline_data is None:
            speedup_baseline_data = data.copy()
        else:
            speedup_per_searchspace = speedup_baseline_data / data
            speedup_per_searchspace_median.append(np.median(speedup_per_searchspace))
            speedup_per_searchspace_std.append(np.std(speedup_per_searchspace))

        # plot
        if project_3d:
            ax.scatter(X, Y, Z, label=searchspace_methods_displayname[method_index])
        else:
            ax[0].scatter(X, Z, label=searchspace_methods_displayname[method_index])
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
        if log_scale:
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')

    # finish plot setup
    fig.tight_layout()
    fig.legend()
    plt.show()

    # plot overall information if applicable
    if show_overall:
        fig, ax = plt.subplots(nrows=2, figsize=(8, 14))
        labels = searchspace_methods_displayname
        ax1, ax2 = ax

        # # setup overall plot
        # ax1.set_xticks(range(len(medians)), labels)
        # ax1.set_xlabel("Method")
        # ax1.set_ylabel("Average time per configuration in seconds")
        # ax1.bar(range(len(medians)), medians, yerr=stds)
        # if log_scale:
        #     ax1.set_yscale('log')

        # setup overall plot
        ax1.set_xticks(range(len(speedup_per_searchspace_median)), labels[1:])
        ax1.set_xlabel("Method")
        ax1.set_ylabel("Median speedup per searchspace")
        ax1.bar(range(len(speedup_per_searchspace_median)), speedup_per_searchspace_median, yerr=speedup_per_searchspace_std)

        # setup plot total searchspaces
        ax2.set_xticks(range(len(medians)), labels)
        ax2.set_xlabel("Method")
        ax2.set_ylabel("Total time in seconds")
        ax2.bar(range(len(medians)), sums)
        if log_scale:
            ax2.set_yscale('log')

        # finish plot setup
        fig.tight_layout()
        plt.show()

        # print speedup
        if len(sums) > 1:
            for method_index in range(1, len(sums)):
                speedup = round(sums[0] / sums[method_index])
                print(f"Total speedup of method '{searchspace_methods_displayname[method_index]}' ({round(sums[method_index], 2)} seconds) over '{searchspace_methods_displayname[0]}' ({round(sums[0], 2)} seconds): {speedup}x")



####
#### User Inputs
####

# searchspaces = [hotspot()]
# searchspaces = [expdist()]
# searchspaces = [dedispersion()]
# searchspaces = [microhh()]
searchspaces = generate_searchspace_variants(max_cartesian_size=100000)
searchspaces = [dedispersion(), expdist(), hotspot(), microhh()]

searchspace_methods = [
    "unoptimized=True",
    # "framework=PythonConstraint,solver_method=PC_BacktrackingSolver",
    "framework=PythonConstraint,solver_method=PC_OptimizedBacktrackingSolver",
    # "framework=PySMT",
]  # must be either 'default' or a kwargs-string passed to Searchspace (e.g. "build_neighbors_index=5,neighbor_method='adjacent'")
searchspace_methods_displayname = [
    "KT 0.4.5",
    # "KT optimized",
    "Optimized",
    # "PySMT",
]

searchspaces_ignore_cache = []      # the indices of the searchspaces to always run, even if they are in cache
searchspace_methods_ignore_cache = []   # the indices of the methods to always run, even if they are in cache


def main():
    """Entry point for execution."""
    searchspaces_results = run(validate_results=False)
    visualize(searchspaces_results)


if __name__ == "__main__":
    main()
