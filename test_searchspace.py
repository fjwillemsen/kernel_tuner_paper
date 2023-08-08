"""A script to test Searchspace initialization times using various search spaces."""

import pickle
import subprocess as sp
from inspect import signature
from itertools import product
from os import execv
from pathlib import Path
from platform import machine, python_version, system
from subprocess import DEVNULL, STDOUT, check_call
from sys import argv, executable, platform
from time import perf_counter
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import progressbar
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.util import check_restrictions, compile_restrictions, default_block_size_names
from psutil import cpu_count, virtual_memory

from searchspaces_provider import dedispersion, expdist, generate_searchspace_variants, hotspot, microhh

progressbar_widgets = [progressbar.PercentageLabelBar(), " [",
                       progressbar.SimpleProgress(format="%(value_s)s/%(max_value_s)s"), ", ",
                       progressbar.Timer(format="Elapsed: %(elapsed)s"), ", ",
                       progressbar.ETA(), "]", ]


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
        check_call(['sh', 'switch_packages_old.sh'], stdout=DEVNULL, stderr=STDOUT)
    else:
        print("Switching from old to new packages")
        check_call(['sh', 'switch_packages_optimized.sh'], stdout=DEVNULL, stderr=STDOUT)

    print(f"Restarting after installing {'old' if old else 'optimized'} packages")

    # restart this script entirely to reload the imports correctly
    execv(executable, ['python'] + [argv[0], str(method_index)])


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
    # return "searchspaces_results_cache_Arch=x86_64_Sys=Linux_CPUs=48_RAM=126.pkl"
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
    tune_params, restrictions, method: str, method_index: int
) -> Tuple[float, int, Searchspace]:
    """Tests the duration of the search space object initialization for a given set of parameters and restrictions and a method.

    Args:
        tune_params: a dictionary of tunable parameters.
        restrictions: restrictions to apply to the tunable parameters.
        method (str): the method with which to initialize the searchspace.
        method_index (int): the current index of the method, used for restarting the script.

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

    # select the appropriate framework
    if framework == "ATF":

        # add the tune_params and restrictions to the ATF source file
        ATF_specify_searchspace_in_source()

        # compile the ATF source file
        ATF_compile()

        # run ATF via a spawned subprocess (because Python C-extensions can not be reloaded with importlib)
        cmd = ['python', "./ATF/run_ATF.py"]
        # input_obj = pickle.dumps(tuple([tune_params, restrictions]))  # can be used with `input=input_obj` in sp.run
        result = sp.run(cmd, shell=False, capture_output=True, text=False, check=True)
        results = pickle.loads(result.stdout)
        print(results)
        exit(0)
    else:
        # install the old (unoptimized) packages if necessary
        global installed_unoptimized
        if unoptimized:
            if not installed_unoptimized:
                installed_unoptimized = switch_packages_to(old=True, method_index=method_index)
            # kwargs are dropped for old KernelTuner & PythonConstraint packages
            kwargs = {}
            framework = 'Old'
            # convert restrictions from list of string to function
            if isinstance(restrictions, list) and len(restrictions) > 0 and all(isinstance(r, str) for r in restrictions):
                restrictions = restrictions_strings_to_function(restrictions, tune_params)
        elif installed_unoptimized:
            # re-install the new (optimized) packages if we previously installed the old packages
            installed_unoptimized = switch_packages_to(old=False, method_index=method_index)

        # initialize and track the performance
        start_time = perf_counter()
        ss = run_searchspace_initialization(
            tune_params, restrictions, framework=framework, kwargs=kwargs
        )
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
    (   tune_params,
        restrictions,
        num_dimensions,
        cartesian_size,
        num_restrictions,
        searchspace_name
    ) = searchspace_variant
    return dict({
            "name": searchspace_name,
            "tune_params": tune_params,
            "restrictions": restrictions,
            "num_dimensions": num_dimensions,
            "cartesian_size": cartesian_size,
            "num_restrictions": num_restrictions,
            "results": results,
        })

def ATF_specify_searchspace_in_source(path_prefix='ATF', sourcename='ATFPython_searchspacespec.cpp'):
    """Replace the contents of the ATF source input file.

    Args:
        path_prefix: the path to the source. Defaults to 'ATF'.
        sourcename: the name of the source input file. Defaults to 'ATFPython_searchspacespec.cpp'.
    """
    source = Path(path_prefix, sourcename)
    assert source.exists() and source.is_file()
    original = source.read_text()
    source.unlink(missing_ok=True)
    source.touch()
    new = "return i - j;" if original == "return i + j;" else "return i + j;"
    source.write_text(new)
    assert source.exists() and source.is_file()

def ATF_compile(std='c++14', path_prefix='ATF'):
    """Compile the ATF source file.

    Args:
        std: the C++ standard to use. Defaults to 'c++14'.
        path_prefix: the path to the source. Defaults to 'ATF'.

    Raises:
        ValueError: in case of unsupported platform.
    """
    # set up environment specifics
    pyversion = '.'.join(python_version().split('.')[:-1]) # python version (major.minor)
    platform_specific = ""
    if platform == "linux":
        platform_specific = "-fPIC"
    elif platform == "darwin":
        platform_specific = "-undefined dynamic_lookup"
    else:
        raise ValueError(f"Platform {platform} not supported.")

    # resolve paths
    pybind11_path = Path(path_prefix, "extern/pybind11/include")
    source_path = Path(path_prefix, "ATFPython.cpp")
    assert pybind11_path.exists()
    assert source_path.exists()

    # define the full command
    command = f"c++ -O2 -shared -std={std} {platform_specific} $(python{pyversion}-config --includes) -I {pybind11_path} {source_path} -o {path_prefix}/ATFPython$(python{pyversion}-config --extension-suffix)"

    # compile by running the command
    sp.run(command, shell=True, text=True, check=True, capture_output=True)


def run(num_repeats=3, validate_results=True, start_from_method_index=0) -> dict[str, Any]:
    """Run the search space variants or retrieve them from cache.

    Args:
        num_repeats (int, optional): the number of times each search space variant is repeated. Defaults to 3.

    Returns:
        dict[str, Any]: the search space variants results.
    """
    global searchspaces_ignore_cache, searchspace_methods_ignore_cache
    bruteforced_key = 'bruteforce'

    # calculate or retrieve the bruteforced results for each variant
    if validate_results:
        bruteforced_searchspaces = list()
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
            key = searchspace_variant_to_key(searchspace_variant, index=searchspace_variant_index)
            results = (
                searchspaces_results[key]["results"]
                if key in searchspaces_results
                else dict()
            )
            if bruteforced_key in results:
                bruteforced_searchspaces.append(results[bruteforced_key]['configs'])

        # if not, bruteforce the searchspaces
        if len(bruteforced_searchspaces) < len(searchspaces):
            for searchspace_variant_index in progressbar.progressbar(
                range(len(searchspaces)),
                redirect_stdout=True,
                prefix=" |-> bruteforcing: ",
                widgets=progressbar_widgets,
            ):
                searchspace_variant = searchspaces[searchspace_variant_index]
                key = searchspace_variant_to_key(searchspace_variant, index=searchspace_variant_index)
                results = (
                    searchspaces_results[key]["results"]
                    if key in searchspaces_results
                    else dict()
                )
                if (method_index not in searchspace_methods_ignore_cache
                    and searchspace_variant_index not in searchspaces_ignore_cache
                    and bruteforced_key in results):
                    # if in cache, retrieve the results from there
                    bruteforced = results[bruteforced_key]['configs']
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
                            "configs": bruteforced
                        }
                    )
                    searchspaces_results[key] = get_searchspace_result_dict(searchspace_variant, results)

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
            key = searchspace_variant_to_key(searchspace_variant, index=searchspace_variant_index)

            # check if the searchspace variant is in the cache
            if (not validate_results
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
            if (method not in results
                or searchspace_variant_index in searchspaces_ignore_cache
                or method_index in searchspace_methods_ignore_cache
                or (validate_results and ('validated' not in results[method] or results[method]["validated"] is False))):
                times_in_seconds = list()
                true_sizes = list()
                for _ in range(num_repeats):
                    time_in_seconds, true_size, searchspace = searchspace_initialization(
                        tune_params=tune_params,
                        restrictions=restrictions,
                        method=method,
                        method_index=method_index
                    )
                    times_in_seconds.append(time_in_seconds)
                    true_sizes.append(true_size)
                    if validate_results:
                        assert_searchspace_validity(bruteforced_searchspaces[searchspace_variant_index], searchspace)
                # set the results
                dirty = True
                results[method] = dict(
                    {
                        "time_in_seconds": times_in_seconds,
                        "true_size": true_sizes,
                        "validated": validate_results
                    }
                )
                searchspaces_results[key] = get_searchspace_result_dict(searchspace_variant, results)

        # write the results to the cache
        if dirty:
            write_to_cache(searchspaces_results)

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
    characteristics_info = {
        'size_true': {
            'log_scale': True,
            'label': 'Number of valid configurations (constrained size)'
        },
        'size_cartesian': {
            'log_scale': True,
            'label': 'Cartesian size (non-constrained size)',
        },
        'fraction_restricted': {
            'log_scale': False,
            'label': 'Fraction of search space constrained',
        },
        'num_dimensions': {
            'log_scale': False,
            'label': 'Number of dimensions (tunable parameters)',
        }

    }
    selected_characteristics = ['size_true', 'size_cartesian', 'fraction_restricted', 'num_dimensions'] # possible values: 'size_true', 'size_cartesian', 'percentage_restrictions', 'num_dimensions'
    if len(selected_characteristics) < 1:
        raise ValueError("At least one characteristic must be selected")

    # setup visualization
    figsize_baseheight = 4
    figsize_basewidth = 3.5
    if project_3d:
        if len(selected_characteristics) > 2:
            raise ValueError("Number of characteristics may be at most 2 for 3D view")
        plt.style.use("_mpl-gallery")
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(figsize_baseheight, figsize_basewidth))
    else:
        if len(selected_characteristics) % 2 == 0:
            ncolumns = 2
            nrows = int(len(selected_characteristics) / 2)
        else:
            ncolumns = 1
            nrows = len(selected_characteristics)
        fig, ax = plt.subplots(ncols=ncolumns, nrows=nrows, figsize=(figsize_baseheight*ncolumns, figsize_basewidth*nrows))
        if isinstance(ax, (list, np.ndarray)):
            ax = np.array(ax).flatten()
        else:
            ax = [ax]

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
        cartesian_sizes = list()  # cartesian size
        nums_dimensions = list()    # number of dimensions
        fraction_restricteds = list()  # fraction of cartesian size after restrictions
        true_sizes = list()  # true size after restrictions
        times_in_seconds = list()  # time taken in seconds

        # retrieve the data from the results dictionary
        for searchspace_variant_index, searchspace_variant in enumerate(
            searchspaces
        ):
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
            if key == 'size_cartesian':
                return cartesian_sizes
            elif key == 'fraction_restricted':
                return fraction_restricteds
            elif key == 'size_true':
                return true_sizes
            elif key == 'num_dimensions':
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
                ax.scatter(X, performance_data, label=searchspace_methods_displayname[method_index])
            else:
                Y = get_data(selected_characteristics[1])
                ax.scatter(X, Y, performance_data, label=searchspace_methods_displayname[method_index])
        else:
            for index, characteristic in enumerate(selected_characteristics):
                if index == 0:
                    ax[index].scatter(get_data(characteristic), performance_data, label=searchspace_methods_displayname[method_index])
                else:
                    ax[index].scatter(get_data(characteristic), performance_data)

    # set labels and axis
    if project_3d:
        info = characteristics_info[selected_characteristics[0]]
        ax.set_xlabel(info['label'])
        if info['log_scale']:
            pass
            # ax.set_xscale('log')
        if len(selected_characteristics) > 1:
            info = characteristics_info[selected_characteristics[1]]
            ax.set_ylabel(info['label'])
            if info['log_scale']:
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
            ax[index].set_xlabel(info['label'])
            # ax[index].set_ylabel("Time in seconds")
            if info['log_scale'] is True:
                ax[index].set_xscale('log')
            if log_scale:
                ax[index].set_yscale('log')
        fig.supylabel("Time per search space in seconds")

    # finish plot setup
    fig.tight_layout()
    fig.legend()
    plt.show()

    # plot overall information if applicable
    if show_overall:
        fig, ax = plt.subplots(nrows=2, figsize=(8, 14))
        labels = searchspace_methods_displayname
        ax1, ax2 = ax

        # setup overall plot
        ax1.set_xticks(range(len(medians)), labels)
        ax1.set_xlabel("Method")
        ax1.set_ylabel("Average time per configuration in seconds")
        ax1.bar(range(len(medians)), medians, yerr=stds)
        if log_scale:
            ax1.set_yscale('log')

        # # setup overall plot
        # ax1.set_xticks(range(len(speedup_per_searchspace_median)), labels[1:])
        # ax1.set_xlabel("Method")
        # ax1.set_ylabel("Median speedup per searchspace")
        # ax1.bar(range(len(speedup_per_searchspace_median)), speedup_per_searchspace_median, yerr=speedup_per_searchspace_std)

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
                speedup = round(sums[0] / sums[method_index], 1)
                print(f"Total speedup of method '{searchspace_methods_displayname[method_index]}' ({round(sums[method_index], 2)} seconds) over '{searchspace_methods_displayname[0]}' ({round(sums[0], 2)} seconds): {speedup}x")


def get_searchspaces_info_latex(searchspaces: list[tuple]):
    print("\\begin{tabularx}{\\linewidth}{l|X|X|X}")
    print("    \\hline")
    print("    \\textbf{Name} & \\textbf{Cartesian size} & \\textbf{Dimensions} & \\textbf{Restrictions} \\\\")
    print("    \\hline")
    for searchspace in searchspaces:
        (tune_params,
        restrictions,
        num_dimensions,
        true_cartesian_size,
        num_restrictions,
        name
        ) = searchspace
        print(f"    {str(name).capitalize()} & {true_cartesian_size} & {num_dimensions} & {num_restrictions} \\\\\\hline")
    print("\end{tabularx}")


####
#### User Inputs
####

# searchspaces = [hotspot()]
# searchspaces = [expdist()]
# searchspaces = [dedispersion()]
# searchspaces = [microhh()]
searchspaces = generate_searchspace_variants(max_cartesian_size=1000000)
searchspaces = [dedispersion(), expdist(), hotspot(), microhh()]

searchspace_methods = [
    # "bruteforce",
    # "unoptimized=True",
    # # "framework=PythonConstraint,solver_method=PC_BacktrackingSolver",
    # "framework=PythonConstraint,solver_method=PC_OptimizedBacktrackingSolver",
    # # "framework=PySMT",
    "framework=ATF"
]  # must be either 'default' or a kwargs-string passed to Searchspace (e.g. "build_neighbors_index=5,neighbor_method='adjacent'")
searchspace_methods_displayname = [
    # "Bruteforce",
    # "Python-Constraint",
    # # "KT optimized",
    # "Optimized",
    # # "PySMT",
    "ATF",
]

searchspaces_ignore_cache = []      # the indices of the searchspaces to always run again, even if they are in cache
# searchspaces_ignore_cache = list(range(len(searchspaces)))
searchspace_methods_ignore_cache = []   # the indices of the methods to always run again, even if they are in cache


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
    searchspaces_results = run(validate_results=True, start_from_method_index=start_from_method_index)
    visualize(searchspaces_results)


if __name__ == "__main__":
    main()
