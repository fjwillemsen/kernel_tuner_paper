"""A script to test Searchspace initialization times using various search spaces."""

import pickle
import warnings
from inspect import signature
from itertools import product
from math import fabs
from os import execv
from pathlib import Path
from platform import machine, system
from subprocess import DEVNULL, STDOUT, check_call
from sys import argv, executable
from time import perf_counter
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import progressbar
import seaborn as sns
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.util import (
    check_restrictions,
    compile_restrictions,
    default_block_size_names,
)
from matplotlib.ticker import MaxNLocator

from searchspaces_provider import (
    atf_gaussian_convolution,
    atf_PRL,
    dedispersion,
    expdist,
    generate_searchspace_variants,
    hotspot,
    microhh,
)

# optional imports
psutil_available = True
try:
    from psutil import cpu_count, virtual_memory
except ModuleNotFoundError:
    psutil_available = False

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
progressbar_widgets = [
    progressbar.PercentageLabelBar(),
    " [",
    progressbar.SimpleProgress(format="%(value_s)s/%(max_value_s)s"),
    ", ",
    progressbar.Timer(format="Elapsed: %(elapsed)s"),
    ", ",
    progressbar.ETA(),
    "]",
]


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


def switch_packages_to(old=True, method_index=0) -> bool:
    """Function to switch between the old and the optimized packages. Reloads imports by restarting the script, so be careful of loops.

    Args:
        old: Whether to load the old packages (if True) or the optimized ones (if False). Defaults to True.
        method_index: the method index as an argument to restart with.

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
        check_call(["sh", "switch_packages_old.sh"], stdout=DEVNULL, stderr=STDOUT)
    else:
        print("Switching from old to new packages")
        check_call(
            ["sh", "switch_packages_optimized.sh"], stdout=DEVNULL, stderr=STDOUT
        )

    print(f"Restarting after installing {'old' if old else 'optimized'} packages")

    # restart this script entirely to reload the imports correctly
    execv(executable, ["python"] + [argv[0], str(method_index)])


def get_machine_info() -> str:
    """Generates a string of device information.

    Returns:
        str: the device information, formatted as {architecture}_{system}_{core count}_{RAM size in GB}.
    """
    if not psutil_available:
        raise ImportError("PSUtil not installed")
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
    return (
        "searchspaces_results_cache_Arch=x86_64_Sys=Linux_CPUs=48_RAM=126_new_prl.pkl"
    )
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
    (_, _, num_dimensions, cartesian_size, num_restrictions, name) = searchspace_variant
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


def run_searchspace_initialization(tune_params, restrictions, kwargs={}) -> Searchspace:
    # initialize the searchspace
    ss = Searchspace(
        tune_params=tune_params,
        restrictions=restrictions,
        max_threads=default_max_threads,
        **kwargs,
    )
    return ss


def bruteforce_searchspace(
    tune_params: dict, restrictions: list, max_threads=default_max_threads
) -> list[tuple]:
    """Bruteforce solving a searchspace (can take a long time depending on input!).

    Args:
        tune_params: a dictionary of tunable parameters.
        restrictions: restrictions to apply to the tunable parameters.

    Returns:
        The resulting list of configurations.
    """
    # compute cartesian product of all tunable parameters
    parameter_space = product(*tune_params.values())
    # size = prod([len(v) for v in tune_params.values()])

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
        parameter_space = filter(
            lambda p: check_restrictions(
                restrictions, dict(zip(tune_params.keys(), p)), False
            ),
            parameter_space,
        )
    return list(parameter_space)


def assert_searchspace_validity(
    bruteforced: list[tuple], searchspace: Searchspace, float_tolerance=None
):
    """Asserts that the given searchspace has the same outcome as the bruteforced list of configurations."""
    assert searchspace.size == len(
        bruteforced
    ), f"Lengths differ: {searchspace.size} != {len(bruteforced)}"

    def find_nearest(array, value):
        """Function to find a value in a sorted array that is closest to value, as per https://stackoverflow.com/a/26026189/7009556."""
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (
            idx == len(array) or fabs(value - array[idx - 1]) < fabs(value - array[idx])
        ):
            return array[idx - 1]
        else:
            return array[idx]

    # get the indices of parameter values in the searchspace that are floating-points
    if float_tolerance is not None and searchspace.size > 0:
        list_numpy = searchspace.get_list_numpy()
        tune_params_keys = list(searchspace.tune_params.keys())
        float_indices = list(
            i for i, v in enumerate(searchspace.list[0]) if isinstance(v, float)
        )
        # for each float value, compare the actual searchspace and tune_param values to find the closest match, map this
        float_replacement_value: list[dict[float, float]] = list(
            dict() for _ in searchspace.list[0]
        )
        for float_index in float_indices:
            tune_param_key = tune_params_keys[float_index]
            tune_param_values = np.array(searchspace.tune_params[tune_param_key])
            searchspace_values = np.unique(list_numpy[:, float_index])
            for tune_param_value in tune_param_values:
                # find the searchspace value that is closest to the tune_param value
                searchspace_value_nearest = find_nearest(
                    searchspace_values, tune_param_value
                )
                # if the difference is within tolerance, add it to the mapping
                if np.isclose(
                    tune_param_value,
                    searchspace_value_nearest,
                    atol=float_tolerance,
                    rtol=1e-10,
                ):
                    float_replacement_value[float_index][
                        tune_param_value
                    ] = searchspace_value_nearest

    # iterate over the bruteforce, checking if each configuration is in the searchspace
    for config in bruteforced:
        if not searchspace.is_param_config_valid(config):
            if float_tolerance is not None:
                # if the config was not found, replace the values with alternative values that certainly occur in the searchspace if within tolerance
                config_replaced = tuple(
                    float_replacement_value[i].get(v, v) if i in float_indices else v
                    for i, v in enumerate(config)
                )
                if searchspace.is_param_config_valid(config_replaced):
                    continue

            raise AssertionError(
                f"Config '{config}' is in the bruteforced searchspace but not in the evaluated searchspace ({float_tolerance=})."
            )


def restrictions_strings_to_function(restrictions: list, tune_params: dict):
    """Parses a list of strings and callables to a monolithic function.

    Args:
        restrictions: a list of string or callable restrictions (can be mixed).
        tune_params: dictionary of tunable parameters.

    Raises:
        ValueError: if not a list of strings.

    Returns:
        the restriction function.
    """
    # check whether the correct types of restrictions have been passed
    if not isinstance(restrictions, list):
        raise ValueError(
            f"Not a list of restrictions: {type(restrictions)}; {restrictions}"
        )
    multiple_callables = []
    string_restrictions = []
    for r in restrictions:
        if isinstance(r, str):
            string_restrictions.append(r)
        elif callable(r):
            multiple_callables.append((r, False))
        elif isinstance(r, tuple) and callable(r[0]):
            multiple_callables.append((r[0], False))
        else:
            raise ValueError(f"Non-string or callable restriction {type(r)}; {r}")

    # add the string restrictions as a function
    if len(multiple_callables) < len(restrictions):
        string_restrictions_function = compile_restrictions(
            string_restrictions, tune_params
        )
        multiple_callables.append((string_restrictions_function, True))

    # return a monolithic function
    if len(multiple_callables) == 1:
        f, b = multiple_callables[0]
        return f if b else lambda p: f(**p)

    # return a wrapper function that calls all other restriction functions
    return lambda p: all(f(p) if b else f(**p) for f, b in multiple_callables)


def searchspace_initialization(
    tune_params, restrictions, method: str, method_index: int, ATF_recompile=True
) -> Tuple[float, int, Searchspace]:
    """Tests the duration of the search space object initialization for a given set of parameters and restrictions and a method.

    Args:
        tune_params: a dictionary of tunable parameters.
        restrictions: restrictions to apply to the tunable parameters.
        method (str): the method with which to initialize the searchspace.
        method_index (int): the current index of the method, used for restarting the script.
        ATF_recompile (bool): whether to recompile the ATF code, must be done after changing tune_params or restrictions. Defaults to True.

    Returns:
        A tuple of the total time taken by the search space initialization, the true size of the search space, and the Searchspace object.
    """
    if callable(restrictions) or (
        (
            isinstance(restrictions, list)
            and len(restrictions) > 0
            and callable(restrictions[0])
        )
    ):
        raise ValueError("Function restrictions can't be pickled")

    # get the keyword arguments
    unoptimized = False
    if method == "default":
        kwargs = {}
        framework = "PythonConstraint"
    else:
        kwargs = {}
        for kwarg in method.split(","):
            keyword, argument = tuple(kwarg.split("="))
            if argument.lower() in ["true", "false"]:
                argument = True if argument.lower() == "true" else False
            if keyword.lower() == "unoptimized":
                unoptimized = True
                continue
            kwargs[keyword] = argument
        framework = kwargs["framework"] if "framework" in kwargs else ""

    # select the appropriate framework
    global installed_unoptimized
    if framework == "ATF":
        assert not installed_unoptimized
        from ATF.ATF import (
            ATF_compile,
            ATF_result_searchspace,
            ATF_run,
            ATF_specify_searchspace_in_source,
        )

        logfilename = "ATF_tuning_log.csv"
        if ATF_recompile:
            # add the tune_params and restrictions to the ATF source file
            ATF_specify_searchspace_in_source(
                tune_params, restrictions, logfilename=logfilename
            )

            # compile the ATF source file
            ATF_compile()

        # run ATF via a spawned subprocess (because Python C-extensions can not be reloaded with importlib)
        results = ATF_run()
        ss = ATF_result_searchspace(tune_params, restrictions, logfilename=logfilename)
        assert results["V"] == ss.size
        return results["T"], ss.size, ss
    elif framework == "PySMT":
        assert not installed_unoptimized
        # initialize and track the performance
        start_time = perf_counter()
        ss = run_searchspace_initialization(tune_params, restrictions, kwargs=kwargs)
        time_taken = perf_counter() - start_time

        # return the time taken in seconds, the searchspace size, and the Searchspace object.
        return time_taken, ss.size, ss
    else:
        # install the old (unoptimized) packages if necessary
        if unoptimized:
            if not installed_unoptimized:
                installed_unoptimized = switch_packages_to(
                    old=True, method_index=method_index
                )
            # kwargs are dropped for old KernelTuner & PythonConstraint packages
            kwargs = {}
            framework = "Old"
            # convert restrictions from list of string to function
            if (
                isinstance(restrictions, list)
                and len(restrictions) > 0
                and any(isinstance(r, str) for r in restrictions)
            ):
                restrictions = restrictions_strings_to_function(
                    restrictions, tune_params
                )
        elif installed_unoptimized:
            # re-install the new (optimized) packages if we previously installed the old packages
            installed_unoptimized = switch_packages_to(
                old=False, method_index=method_index
            )

        # initialize and track the performance
        start_time = perf_counter()
        ss = run_searchspace_initialization(tune_params, restrictions, kwargs=kwargs)
        time_taken = perf_counter() - start_time

        # return the time taken in seconds, the searchspace size, and the Searchspace object.
        return time_taken, ss.size, ss


def get_cached_results() -> dict:
    """Get the dictionary of results.

    Returns:
        the results dictionary.
    """
    try:
        return read_from_cache()
    except FileNotFoundError:
        print(f"Cachefile '{get_cache_filename()}' not found, creating...")
        return dict()


def get_searchspace_result_dict(searchspace_variant: tuple, results: dict) -> dict:
    (
        tune_params,
        restrictions,
        num_dimensions,
        cartesian_size,
        num_restrictions,
        searchspace_name,
    ) = searchspace_variant
    return dict(
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


def run(
    num_repeats=3, validate_results=True, start_from_method_index=0
) -> dict[str, Any]:
    """Run the search space variants or retrieve them from cache.

    Args:
        num_repeats (int, optional): the number of times each search space variant is repeated. Defaults to 3.

    Returns:
        dict[str, Any]: the search space variants results.
    """
    global searchspaces_ignore_cache, searchspace_methods_ignore_cache
    bruteforced_key = "bruteforce"

    # calculate or retrieve the bruteforced results for each variant
    if validate_results:
        bruteforced_searchspaces = list()
        already_bruteforced_indices = list()
        searchspaces_results = get_cached_results()
        try:
            method_index = searchspace_methods.index(bruteforced_key)
        except ValueError:
            method_index = -1

        # check if all searchspaces have been bruteforced
        for searchspace_variant_index, searchspace_variant in enumerate(searchspaces):
            # check if the bruteforce method is in `searchspace_methods_ignore_cache`
            if method_index in searchspace_methods_ignore_cache:
                break
            # check if this searchspace variant is in `searchspaces_ignore_cache``
            if searchspace_variant_index in searchspaces_ignore_cache:
                continue
            key = searchspace_variant_to_key(
                searchspace_variant, index=searchspace_variant_index
            )
            results = (
                searchspaces_results[key]["results"]
                if key in searchspaces_results
                else dict()
            )
            if bruteforced_key in results:
                print(f"Bruteforced {key}")
                bruteforced_searchspaces.append(results[bruteforced_key]["configs"])
                already_bruteforced_indices.append(searchspace_variant_index)
            else:
                print(f"Non bruteforced {key}")

        # if not, bruteforce the searchspaces
        if len(bruteforced_searchspaces) < len(searchspaces):
            for searchspace_variant_index in progressbar.progressbar(
                range(len(searchspaces)),
                redirect_stdout=True,
                prefix=" |-> bruteforcing: ",
                widgets=progressbar_widgets,
            ):
                if searchspace_variant_index in already_bruteforced_indices:
                    continue

                searchspace_variant = searchspaces[searchspace_variant_index]
                key = searchspace_variant_to_key(
                    searchspace_variant, index=searchspace_variant_index
                )
                results = (
                    searchspaces_results[key]["results"]
                    if key in searchspaces_results
                    else dict()
                )
                if (
                    method_index not in searchspace_methods_ignore_cache
                    and searchspace_variant_index not in searchspaces_ignore_cache
                    and bruteforced_key in results
                ):
                    # if in cache, retrieve the results from there
                    bruteforced = results[bruteforced_key]["configs"]
                else:
                    # if not in cache, brute-force the searchspace
                    tune_params, restrictions, _, _, _, _ = searchspace_variant
                    start_time = perf_counter()
                    bruteforced = bruteforce_searchspace(tune_params, restrictions)
                    time_in_seconds = perf_counter() - start_time

                    # set the results
                    results[bruteforced_key] = dict(
                        {
                            "time_in_seconds": [time_in_seconds],
                            "true_size": [len(bruteforced)],
                            "configs": bruteforced,
                        }
                    )
                    searchspaces_results[key] = get_searchspace_result_dict(
                        searchspace_variant, results
                    )

                # add to the list for later usage
                bruteforced_searchspaces.append(bruteforced)

            # write the results to the cache
            write_to_cache(searchspaces_results)
            print("All searchspaces have been bruteforced for validation.")

    # run each searchspace method
    for method_index in range(start_from_method_index, len(searchspace_methods)):
        method = searchspace_methods[method_index]
        if method == bruteforced_key:
            continue

        # get cached results if available
        searchspaces_results = get_cached_results()

        # run or retrieve from cache all searchspace variants
        dirty = False
        for searchspace_variant_index in progressbar.progressbar(
            range(len(searchspaces)),
            redirect_stdout=True,
            prefix=f" |-> running '{searchspace_methods_displayname[method_index]}': ",
            widgets=progressbar_widgets,
        ):
            # get the searchspace variant details
            searchspace_variant = searchspaces[searchspace_variant_index]
            tune_params, restrictions, _, _, _, _ = searchspace_variant
            key = searchspace_variant_to_key(
                searchspace_variant, index=searchspace_variant_index
            )

            # check if the searchspace variant is in the cache
            if (
                not validate_results
                and key in searchspaces_results
                and searchspace_variant_index not in searchspaces_ignore_cache
                and method_index not in searchspace_methods_ignore_cache
                and method in searchspaces_results[key]["results"]
            ):
                continue

            # run the variant
            results = (
                searchspaces_results[key]["results"]
                if key in searchspaces_results
                else dict()
            )
            if (
                method not in results
                or searchspace_variant_index in searchspaces_ignore_cache
                or method_index in searchspace_methods_ignore_cache
                or (
                    validate_results
                    and (
                        "validated" not in results[method]
                        or results[method]["validated"] is False
                    )
                )
            ):
                times_in_seconds = list()
                true_sizes = list()
                for i in range(num_repeats):
                    time_in_seconds, true_size, searchspace = (
                        searchspace_initialization(
                            tune_params=tune_params,
                            restrictions=restrictions,
                            method=method,
                            method_index=method_index,
                            ATF_recompile=i == 0,
                        )
                    )
                    times_in_seconds.append(time_in_seconds)
                    true_sizes.append(true_size)
                    # validate the results if enabled (only on the first repeat)
                    if validate_results and i == 0:
                        bruteforced = bruteforced_searchspaces[
                            searchspace_variant_index
                        ]
                        assert (
                            len(bruteforced) == true_size
                        ), f"{len(bruteforced)} != {true_size}"
                        if searchspace is not None:
                            float_tolerance = (
                                1e-9 if "framework=ATF" in method else None
                            )  # with ATF, configurations may be imprecisely rounded due to C++/Python conversion
                            assert_searchspace_validity(
                                bruteforced,
                                searchspace,
                                float_tolerance=float_tolerance,
                            )
                # set the results
                dirty = True
                results[method] = dict(
                    {
                        "time_in_seconds": times_in_seconds,
                        "true_size": true_sizes,
                        "validated": validate_results,
                    }
                )
                searchspaces_results[key] = get_searchspace_result_dict(
                    searchspace_variant, results
                )

                # write the results to the cache if they took a while to obtain
                if np.mean(times_in_seconds) > 10:
                    write_to_cache(searchspaces_results)

        # write the results to the cache
        if dirty:
            write_to_cache(searchspaces_results)

    return searchspaces_results


def visualize(
    searchspaces_results: dict[str, Any],
    project_3d=False,
    show_overall=True,
    log_scale=True,
    time_scale=True,
    show_figs=True,
    save_figs=False,
    save_folder="figures/MacBook",
    save_filename_prefix="",
    dpi=200,
    legend_outside=False,
):
    """Visualize the results of search spaces in a plot.

    Args:
        searchspaces_results (dict[str, Any]): the cached results dictionary.
        project_3d (bool, optional): whether to visualize as one 3D or two 2D plots. Defaults to False.
        show_overall (bool, optional): whether to also plot overall performance between methods. Defaults to True.
        log_scale (bool, optional): whether to plot time on a logarithmic scale instead of default. Defaults to True.
        time_scale (bool, optional): whether to show an additional time scale for context. Defaults to False.
    """
    # setup characteristics
    characteristics_info = {
        "size_true": {
            "log_scale": True,
            "label": "Number of valid configurations (constrained size)",
        },
        "size_cartesian": {
            "log_scale": True,
            "label": "Cartesian size (non-constrained size)",
        },
        "fraction_restricted": {
            "log_scale": False,
            "label": "Fraction of search space constrained",
        },
        "num_dimensions": {
            "log_scale": False,
            "label": "Number of dimensions (tunable parameters)",
        },
    }
    selected_characteristics = [
        "size_true",
        "size_cartesian",
        "fraction_restricted",
        "num_dimensions",
    ]  # possible values: 'size_true', 'size_cartesian', 'percentage_restrictions', 'num_dimensions'
    if len(selected_characteristics) < 1:
        raise ValueError("At least one characteristic must be selected")

    # setup visualization
    figsize_baseheight = 4
    figsize_basewidth = 3.5
    if project_3d:
        if len(selected_characteristics) > 2:
            raise ValueError("Number of characteristics may be at most 2 for 3D view")
        plt.style.use("_mpl-gallery")
        fig, ax = plt.subplots(
            subplot_kw={"projection": "3d"},
            figsize=(figsize_baseheight, figsize_basewidth),
        )
    else:
        if len(selected_characteristics) % 2 == 0:
            ncolumns = 2
            nrows = int(len(selected_characteristics) / 2)
        else:
            ncolumns = 1
            nrows = len(selected_characteristics)
        fig, ax = plt.subplots(
            ncols=ncolumns,
            nrows=nrows,
            figsize=(figsize_baseheight * ncolumns, figsize_basewidth * nrows),
            dpi=dpi,
        )
        if isinstance(ax, (list, np.ndarray)):
            ax = np.array(ax).flatten()
        else:
            ax = [ax]

    # setup saving
    if save_figs:
        save_path = Path(save_folder)
        assert save_path.exists(), f"Path {save_path} does not exist"
        if save_filename_prefix == "":
            warnings.warn(
                f"Unused figure filename prefix ({save_filename_prefix=})", UserWarning
            )

    # loop over each method
    sums = list()
    means = list()
    medians = list()
    stds = list()
    last_y = list()
    times = list()
    speedup_per_searchspace_median = list()
    speedup_per_searchspace_std = list()
    speedup_baseline_data = None
    for method_index, method in enumerate(searchspace_methods):
        # setup arrays
        cartesian_sizes = list()  # cartesian size
        nums_dimensions = list()  # number of dimensions
        fraction_restricteds = list()  # fraction of cartesian size after restrictions
        true_sizes = list()  # true size after restrictions
        times_in_seconds = list()  # time taken in seconds

        # retrieve the data from the results dictionary
        for searchspace_variant_index, searchspace_variant in enumerate(searchspaces):
            num_dimensions = searchspace_variant[2]
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
            cartesian_sizes.append(cartesian_size)
            nums_dimensions.append(num_dimensions)
            fraction_restricteds.append(1 - (true_size / cartesian_size))
            true_sizes.append(true_size)
            times_in_seconds.append(time_in_seconds)

        # clean up data
        cartesian_sizes = np.array(cartesian_sizes)
        fraction_restricteds = np.array(fraction_restricteds)
        true_sizes = np.array(true_sizes)
        nums_dimensions = np.array(nums_dimensions)
        times_in_seconds = np.array(times_in_seconds)

        def get_data(key: str) -> np.ndarray:
            if key == "size_cartesian":
                return cartesian_sizes
            elif key == "fraction_restricted":
                return fraction_restricteds
            elif key == "size_true":
                return true_sizes
            elif key == "num_dimensions":
                return nums_dimensions
            else:
                raise ValueError(f"Unkown data {key}")

        # add statistical data for reporting
        performance_data = times_in_seconds
        sums.append(np.sum(performance_data))
        means.append(np.mean(performance_data))
        medians.append(np.median(performance_data))
        stds.append(np.std(performance_data))
        last_y.append(performance_data[-1])
        times.append(times_in_seconds)

        # calculate speedups relative to baseline
        if speedup_baseline_data is None:
            speedup_baseline_data = performance_data.copy()
        else:
            speedup_per_searchspace = speedup_baseline_data / performance_data
            speedup_per_searchspace_median.append(np.median(speedup_per_searchspace))
            speedup_per_searchspace_std.append(np.std(speedup_per_searchspace))

        # plot
        if project_3d:
            X = get_data(selected_characteristics[0])
            if len(selected_characteristics) == 1:
                ax.scatter(
                    X,
                    performance_data,
                    label=searchspace_methods_displayname[method_index],
                    c=searchspace_methods_colors[method_index],
                )
            else:
                Y = get_data(selected_characteristics[1])
                ax.scatter(
                    X,
                    Y,
                    performance_data,
                    label=searchspace_methods_displayname[method_index],
                    c=searchspace_methods_colors[method_index],
                )
        else:
            for index, characteristic in enumerate(selected_characteristics):
                if index == 0:
                    ax[index].scatter(
                        get_data(characteristic),
                        performance_data,
                        label=searchspace_methods_displayname[method_index],
                        c=searchspace_methods_colors[method_index],
                    )
                else:
                    ax[index].scatter(
                        get_data(characteristic),
                        performance_data,
                        c=searchspace_methods_colors[method_index],
                    )
                if characteristic == "num_dimensions":
                    ax[index].xaxis.set_major_locator(MaxNLocator(integer=True))

    # set labels and axis
    if project_3d:
        info = characteristics_info[selected_characteristics[0]]
        ax.set_xlabel(info["label"])
        if info["log_scale"]:
            pass
            # ax.set_xscale('log')
        if len(selected_characteristics) > 1:
            info = characteristics_info[selected_characteristics[1]]
            ax.set_ylabel(info["label"])
            if info["log_scale"]:
                pass
                # ax.set_yscale('log')
            ax.set_zlabel("Time in seconds")
        else:
            ax.set_ylabel("Time in seconds")
            if log_scale:
                pass
                # ax.set_zscale('log')
        # ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        # ax.set_xticks(np.arange(np.min(X), np.max(X), np.max(X) / 10))
    else:
        for index, characteristic in enumerate(selected_characteristics):
            info = characteristics_info[characteristic]
            ax[index].set_xlabel(info["label"])
            # ax[index].set_ylabel("Time in seconds")
            if info["log_scale"] is True:
                ax[index].set_xscale("log")
            if log_scale:
                ax[index].set_yscale("log")
        fig.supylabel("Time per search space in seconds")

    # finish plot setup
    fig.tight_layout()
    if legend_outside:
        fig.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    else:
        # fig.legend(loc="upper left")
        # fig.legend()
        pass
    if save_figs:
        filename = f"results_{save_filename_prefix}_characteristics"
        plt.savefig(Path(save_path, filename), dpi=dpi)
    if show_figs:
        plt.show()

    # plot overall information if applicable
    if show_overall:
        fig, ax = plt.subplots(nrows=2, figsize=(4.3, 7.5), dpi=dpi)
        labels = searchspace_methods_displayname
        ax1, ax2 = ax

        # setup overall plot
        # ax1.set_xticks(range(len(medians)), labels)
        # ax1.set_xlabel("Method")
        # ax1.set_ylabel("Average time per configuration in seconds")
        # # bars = ax1.bar(range(len(medians)), medians, yerr=stds)
        # bars = ax1.bar(range(len(medians)), medians)
        # for i, bar in enumerate(bars):
        #     bar.set_color(searchspace_methods_colors[i])
        # if log_scale:
        #     ax1.set_yscale("log")

        # # setup overall plot
        # ax1.set_xticks(range(len(speedup_per_searchspace_median)), labels[1:])
        # ax1.set_xlabel("Method")
        # ax1.set_ylabel("Median speedup per searchspace")
        # ax1.bar(
        #     range(len(speedup_per_searchspace_median)),
        #     speedup_per_searchspace_median,
        #     yerr=speedup_per_searchspace_std,
        # )

        # setup overall plot with distribution
        sns.set_style("whitegrid")
        for i, times_ in enumerate(times):
            sns.kdeplot(
                y=times_,
                ax=ax1,
                color=searchspace_methods_colors[i],
                log_scale=log_scale,
                fill=True,
            )
        ax1.set_ylabel("Time in seconds")
        if log_scale:
            ax1.set_yscale("log")

        # setup plot total searchspaces
        ax2.set_xticks(range(len(medians)), labels)
        ax2.set_xlabel("Method")
        ax2.set_ylabel("Total time in seconds")
        bars = ax2.bar(range(len(medians)), sums)
        for i, bar in enumerate(bars):
            bar.set_color(searchspace_methods_colors[i])
        if log_scale:
            ax2.set_yscale("log")

        # set additional time scale
        if time_scale:
            time_dict = {
                10**-9: "ns",
                10**-6: "µs",
                10**-3: "ms",
                1: "s",
                60: "min",
                60 * 60: "hr",
                60 * 60 * 24: "d",
                60 * 60 * 24 * 365: "y",
                60 * 60 * 24 * 365 * 100: "c",
            }
            ax1t = ax1.secondary_yaxis(location=1)
            if log_scale:
                ax1t.set_yscale("log")
            ax1t.set_yticks(list(time_dict.keys()), labels=list(time_dict.values()))
            ax2t = ax2.secondary_yaxis(location=1)
            if log_scale:
                ax2t.set_yscale("log")
            ax2t.set_yticks(list(time_dict.keys()), labels=list(time_dict.values()))

        # finish plot setup
        fig.tight_layout()
        if save_figs:
            filename = f"results_{save_filename_prefix}_overall"
            plt.savefig(Path(save_path, filename), dpi=dpi)
        if show_figs:
            plt.show()

        # print speedup
        if len(sums) > 1:
            for method_index in range(1, len(sums)):
                speedup = round(sums[0] / sums[method_index], 1)
                print(
                    f"Total speedup of method '{searchspace_methods_displayname[method_index]}' ({round(sums[method_index], 2)} seconds) over '{searchspace_methods_displayname[0]}' ({round(sums[0], 2)} seconds): {speedup}x"
                )


def get_searchspaces_info_latex(searchspaces: list[tuple], use_cache_info=True):
    """Function to automatically generate a LaTeX table of the searchspace information."""
    if use_cache_info:
        print("\\begin{tabularx}{\\linewidth}{l|X|X|X|X}")
    else:
        print("\\begin{tabularx}{\\linewidth}{l|X|X|X}")
    print("    \\hline")
    if use_cache_info:
        print(
            "    \\textbf{Name} & \\textbf{Cartesian size} & \\textbf{Constraint size} & \\textbf{Dimensions} & \\textbf{Restrictions} \\\\"
        )
    else:
        print(
            "    \\textbf{Name} & \\textbf{Cartesian size} & \\textbf{Dimensions} & \\textbf{Restrictions} \\\\"
        )
    print("    \\hline")
    if use_cache_info:
        searchspaces_results = get_cached_results()
        assert len(searchspaces) == len()
    for searchspace_index, searchspace in enumerate(searchspaces):
        (
            tune_params,
            restrictions,
            num_dimensions,
            true_cartesian_size,
            num_restrictions,
            name,
        ) = searchspace
        if not use_cache_info:
            print(
                f"    {str(name).capitalize()} & {true_cartesian_size} & {num_dimensions} & {num_restrictions} \\\\\\hline"
            )
        else:
            key = searchspace_variant_to_key(searchspace, index=searchspace_index)
            if key in searchspaces_results:
                method = list(searchspaces_results.keys())[0]
                reported_sizes = searchspaces_results[key]["results"][method][
                    "true_size"
                ]
                assert all(reported_sizes[0] == s for s in reported_sizes)
                print(
                    f"    {str(name).capitalize()} & {true_cartesian_size} & {reported_sizes[0]} & {num_dimensions} & {num_restrictions} \\\\\\hline"
                )
    print("\end{tabularx}")


####
#### User Inputs
####

searchspaces = generate_searchspace_variants(max_cartesian_size=1000000)
searchspaces = [hotspot()]
searchspaces = [expdist()]
searchspaces = [dedispersion()]
searchspaces = [microhh()]
searchspaces = [atf_gaussian_convolution()]
searchspaces = [atf_PRL()]
searchspaces = [
    atf_PRL(input_size=8),
    dedispersion(),
    expdist(),
    hotspot(),
    microhh(),
    atf_PRL(input_size=4),
    atf_PRL(input_size=2),
]
searchspaces_name = "synthetic"
searchspaces_name = "realworld"

searchspace_methods = [
    "bruteforce",
    # "unoptimized=True",
    # "framework=PythonConstraint,solver_method=PC_BacktrackingSolver",
    "framework=PythonConstraint,solver_method=PC_OptimizedBacktrackingSolver",
    "framework=ATF",
    # "framework=PySMT",
]  # must be either 'default' or a kwargs-string passed to Searchspace (e.g. "build_neighbors_index=5,neighbor_method='adjacent'")
searchspace_methods_displayname = [
    "Bruteforce",
    # "Kernel Tuner\n(current)",
    # "KT optimized",
    "Kernel Tuner\n(optimized)",
    "ATF",
    # "PySMT",
]
# searchspace_methods = [
#     "unoptimized=True",
#     "framework=PythonConstraint,solver_method=PC_OptimizedBacktrackingSolver",
# ]
# searchspace_methods_displayname = [
#     "python-constraint (current)",
#     # "KT optimized",
#     "python-constraint (optimized)",
# ]
searchspace_methods_colors = [
    colors[i] for i in range(len(searchspace_methods_displayname))
]

searchspaces_ignore_cache = (
    []
)  # the indices of the searchspaces to always run again, even if they are in cache
# searchspaces_ignore_cache = list(range(len(searchspaces)))
searchspace_methods_ignore_cache = (
    []
)  # the indices of the methods to always run again, even if they are in cache


def main():
    """Entry point for execution."""
    # print("")
    start_from_method_index = 0
    if len(argv) > 1:
        # if the program has been restarted to switch packages, restart from that method
        try:
            start_from_method_index = int(argv[1])
        except ValueError:
            pass
    searchspaces_results = run(
        validate_results=True, start_from_method_index=start_from_method_index
    )
    visualize(
        searchspaces_results,
        show_figs=False,
        save_figs=True,
        save_folder="figures/searchspace_generation/DAS6",
        save_filename_prefix=searchspaces_name,
    )


if __name__ == "__main__":
    main()
