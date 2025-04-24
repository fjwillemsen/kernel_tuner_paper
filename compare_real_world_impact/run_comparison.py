""""Run the tuning of each combination of kernel and platform for each searchspace constructor."""

from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kernels.hotspot.hotspot import tune as tune_hotspot
from kernels.gemm.gemm import tune as tune_gemm
# from kernels.expdist.expdist import tune as tune_expdist

# beware this code currently has some assumptions that we use a single searchspace (kernel+device+inputs combination)!
performance_objective = 'GFLOP/s'  # the key to use for the performance metric
kernels = ["gemm"]              # names of the kernel and folder in the kernels folder (must be the same)
platforms = [("CUDA", "A100")]  # tuple of language and device, for language choose from CUDA, HIP and OpenCL
iterations = 10                 # number of times to repeat each tuning run
num_minutes = 10                # time limit for each tuning run in minutes
minimize = False                # whether to minimize the objective function (time) or maximize it (performance)
strategy = "random_sample"      # the strategy to use for tuning
searchspace_constructors = [    # the searchspace construction frameworks to use
    "bruteforce",
    "pythonconstraint",
    "pyatf",
]

# for gemm plots
performance_objective = "GFLOP/s"
kernels = ["gemm"]              
num_minutes = 10

# # for hotspot plots
# performance_objective = "gridpoints/s"
# kernels = ["hotspot"]              
# num_minutes = 30

minutes_line = np.linspace(0, num_minutes, num_minutes*60)  # time line for the plot

# map the searchspace constructor names to their display names
searchspace_methods_displaynames = {
    "pythonconstraint": "Optimized",
    "pyatf": "pyATF",
    "bruteforce": "Bruteforce", 
    "original": "Original",
}

objective_displaynames = {
    'time': 'time in seconds',
    'GFLOP/s': 'GFLOP/s',
    'gridpoints/s': 'gridpoints per second',
}

# set the tune_func
assert len(kernels) == 1 # currently we can do one kernel at a time
if kernels[0] == "hotspot":
    tune_func = tune_hotspot
elif kernels[0] == "gemm":
    tune_func = tune_gemm
elif kernels[0] == "expdist":
    tune_func = tune_expdist
else:
    raise ValueError(f"Kernel {kernels} not a valid selection")

# execute the tuning for each combination
print("  Starting tuning runs ")
print("-------------------------")
for iteration in range(iterations):
    for searchspace_constructor in searchspace_constructors:
        for kernel in kernels:
            if len(kernels) > 1:
                print(f"running kernel {kernel}:")
            for language, device in platforms:
                if len(platforms) > 1:
                    print(f"  on {device=} with {language=}:")
                # create the cache path and skip this combination if it already exists
                # cachefile_path = Path(f"results/hotspot/{device.upper()}_f={searchspace_constructor}_s={strategy}_i={iteration}.json")
                cachefile_path = Path(f"results/{kernel}/{device.upper()}_f={searchspace_constructor}_i={iteration}.json")
                assert cachefile_path.parent.exists(), f"Cachefile path {cachefile_path.parent} does not exist"
                if cachefile_path.exists():
                    print(f"    skipping {searchspace_constructor} (iter. {iteration}) as it already exists")
                    # with cachefile_path.open("r") as f:
                    #     file_creation_time = datetime.fromtimestamp(cachefile_path.stat().st_mtime)
                    #     raise ValueError(file_creation_time)
                    #     data = json.load(f)
                    #     if creation_timestamp_key not in data:
                    #         data[creation_timestamp_key] = file_creation_time.isoformat()
                    #         json.write(data, f, indent=4)
                    continue

                # set the tuning parameters
                strategy_options = {
                    'time_limit': num_minutes*60,     # time limit in seconds
                    'searchspace_construction_options': {
                        'framework': searchspace_constructor
                    }
                }

                # tune
                try:
                    res, env = tune_func(
                        device_name=device, 
                        strategy=strategy, 
                        strategy_options=strategy_options,
                        simulation_mode=False,
                        lang=language,
                        verbose=False,
                        cachefile_path=str(cachefile_path),
                        # objective=performance_objective,
                        # objective_higher_is_better=not minimize,
                    )
                except RuntimeError as e:
                    # if it's one time-out, try again
                    print(f"Runtime error encountered: {e}, trying once more...")
                    cachefile_path.unlink(missing_ok=True)
                    res, env = tune_func(
                        device_name=device, 
                        strategy=strategy, 
                        strategy_options=strategy_options,
                        simulation_mode=False,
                        lang=language,
                        verbose=False,
                        cachefile_path=str(cachefile_path),
                        # objective=performance_objective,
                        # objective_higher_is_better=not minimize,
                    )
                print(f"{searchspace_constructor} (iter. {iteration}) evaluated {len(res)} configs in ~{num_minutes} minutes")

                # assert the cache file path now exists
                assert cachefile_path.exists(), f"Cachefile does not exist at {cachefile_path}"

    print("|------------------------")
print("  Finished tuning runs ")
print("")

# # add additional information to the cachefile if it is missing
# for searchspace_constructor in searchspace_constructors:
#     for kernel in kernels:
#         for language, device in platforms:
#             for iteration in range(iterations):
#                 # create the cache path and skip this combination if it already exists
#                 cachefile_path = Path(f"results/hotspot/{device.upper()}_f={searchspace_constructor}_s={strategy}_i={iteration}.json")
#                 cachefile_path = Path(f"results/{kernel}/{device.upper()}_f={searchspace_constructor}_i={iteration}.json")
#                 # for each searchspace constructor, aggregate the results
#                 if cachefile_path.exists():
#                     # get the file created time as a date
#                     # file_creation_time = datetime.fromtimestamp(cachefile_path.stat().st_mtime)
#                     print(f"Overwriting modified cachefile at {cachefile_path}")
#                     modified = False
#                     with cachefile_path.open("r") as f:
#                         data = json.load(f)
#                         cache = data['cache']
#                         for k, v in cache.items():
#                             if 'gridpoints/s' not in v and 'GFLOP/s' in v and isinstance(v['GFLOP/s'], (int, float)):
#                                 v['gridpoints/s'] = (v['GFLOP/s'] * 1e9) / 15
#                                 cache[k] = v
#                                 modified = True
#                         data['cache'] = cache
#                     if modified:
#                         with cachefile_path.open("w") as f:
#                             json.dump(data, f)

# aggregate the results
results = {
    'num_configs': {},
    'num_configs_time': {},
    'configs_performance': {},
    'configs_performance_time': {},
    'best_relative_performance': {},
    'tuning_start_time': {}
}
for searchspace_constructor in searchspace_constructors:
    results['num_configs'][searchspace_constructor] = []
    results['num_configs_time'][searchspace_constructor] = []
    results['configs_performance'][searchspace_constructor] = []
    results['configs_performance_time'][searchspace_constructor] = []
    for kernel in kernels:
        for language, device in platforms:
            for iteration in range(iterations):
                # create the cache path and skip this combination if it already exists
                # cachefile_path = Path(f"results/hotspot/{device.upper()}_f={searchspace_constructor}_s={strategy}_i={iteration}.json")
                cachefile_path = Path(f"results/{kernel}/{device.upper()}_f={searchspace_constructor}_i={iteration}.json")

                # for each searchspace constructor, aggregate the results
                assert cachefile_path.exists(), f"Cachefile at {cachefile_path} must exist"
                # get the file created time as a date
                # file_creation_time = datetime.fromtimestamp(cachefile_path.stat().st_mtime)
                with cachefile_path.open("r") as f:
                    data = json.load(f)
                    cache = data['cache']
                    num_configs = len(cache)
                    tuning_start_time = datetime.fromisoformat(cache[list(cache.keys())[-1]]['timestamp']) - timedelta(minutes=num_minutes)
                    # check if the tuning start time is before the first config
                    assert tuning_start_time < datetime.fromisoformat(next(iter(cache.values()))['timestamp']), f"{cachefile_path} tuning start time {tuning_start_time} is not before the first config time"
                    results['tuning_start_time'][searchspace_constructor] = tuning_start_time
                    results['num_configs'][searchspace_constructor].append(num_configs)
                    configs_performance = [c[performance_objective] for c in cache.values() if performance_objective in c and isinstance(c[performance_objective], (int, float))]
                    results['configs_performance'][searchspace_constructor].append(configs_performance)

                    # write the best performance so far at fixed time intervals of minutes_line (e.g. [0.1, 0.2, 0.3, ...] minutes)
                    num_configs_so_far = 0
                    best_performance_so_far = np.inf if minimize else -np.inf
                    first_time_to_config_index = None
                    last_time_to_config_index = None
                    num_configs_time_local = []
                    configs_performance_time_local = []
                    for k, v in cache.items():
                        num_configs_so_far += 1
                        config_timestamp = datetime.fromisoformat(v['timestamp'])
                        time_to_config = (config_timestamp - tuning_start_time).total_seconds() / 60
                        # if we've passed the next point on the minutes line, add the best performance so far
                        if last_time_to_config_index is None or (last_time_to_config_index+1 < len(minutes_line) and time_to_config >= minutes_line[last_time_to_config_index+1]):
                            # get the new last_time_to_config_index it should be set to
                            new_time_to_config_index = np.where(minutes_line >= time_to_config)[0][0]
                            if last_time_to_config_index is None:
                                # if this is the first time we are adding elements, set the performance so far to NaN
                                configs_performance_time_local += [np.nan] * min(new_time_to_config_index, len(minutes_line))
                                num_configs_time_local += [np.nan] * min(new_time_to_config_index, len(minutes_line))
                                first_time_to_config_index = new_time_to_config_index
                            else:
                                # calculate the number of elements that should be added in between
                                elems_to_add = new_time_to_config_index - last_time_to_config_index
                                assert elems_to_add >= 1 and elems_to_add + len(configs_performance_time_local) <= len(minutes_line), f"{elems_to_add=} + {len(configs_performance_time_local)=} not <= {len(minutes_line)=}"
                                # add the best performance so far for the missing elements
                                configs_performance_time_local += [best_performance_so_far] * elems_to_add
                                num_configs_time_local += [num_configs_so_far] * elems_to_add
                            # update the best performance so far from now on
                            best_performance_so_far = min(best_performance_so_far, v[performance_objective]) if minimize else max(best_performance_so_far, v[performance_objective])
                            last_time_to_config_index = new_time_to_config_index
                        # print(f"File time for {cachefile_path}: {tuning_start_time}, config: {config_timestamp}, time taken: {time_to_config}")
                    # fill the remaining elements with the best performance so far, if we never added any elements, set the elements to NaN
                    remaining_elems_to_add = len(minutes_line) - len(configs_performance_time_local)
                    configs_performance_time_local += [np.nan if last_time_to_config_index is None else best_performance_so_far] * remaining_elems_to_add
                    num_configs_time_local += [np.nan if last_time_to_config_index is None else num_configs_so_far] * remaining_elems_to_add
                    assert len(configs_performance_time_local) == len(minutes_line), f"{len(configs_performance_time_local)=} != {len(minutes_line)=}"
                    assert len(num_configs_time_local) == len(minutes_line), f"{len(num_configs_time_local)=} != {len(minutes_line)=}"
                    # check if the performance is monotonically increasing or decreasing (depending on minimize)
                    assert all(x >= y if minimize else x <= y for x, y in zip(configs_performance_time_local[first_time_to_config_index:], configs_performance_time_local[first_time_to_config_index+1:]))
                    assert all(x <= y for x, y in zip(num_configs_time_local[first_time_to_config_index:], num_configs_time_local[first_time_to_config_index+1:]))
                    results['configs_performance_time'][searchspace_constructor].append(configs_performance_time_local)
                    results['num_configs_time'][searchspace_constructor].append(num_configs_time_local)


# get the average performance over all searchspace constructors
avg_performance = np.mean(np.array([c for s in searchspace_constructors for i in results['configs_performance'][s] for c in i]).flatten())
# set the best performance for each searchspace constructor relative to the average
for s in searchspace_constructors:
    # for each iteration get the best performance relative to the overall average
    results['best_relative_performance'][s] = []
    for i in range(iterations):
        # get the performance of the configurations for this iteration
        configs_performance = np.array(results['configs_performance'][s][i])
        # calculate the speedup of the best performance over the average performance
        if minimize:
            results['best_relative_performance'][s].append(avg_performance / min(configs_performance))
        else:
            results['best_relative_performance'][s].append(max(configs_performance) / avg_performance)


# plot the results

# generate the colors (same as in test_searchspace.py)
searchspace_methods_colors_dict = {
    "Bruteforce": "#1f77b4",
    "Original": "#ff7f0e",
    "Optimized": "#2ca02c",
    "ATF": "#d62728",
    "pyATF": "#9467bd",
    "PySMT": "#8c564b",
    "parallel": "#e377c2",
    "optimized2": "#7f7f7f",
    "non_method": "#17becf",    # reserve a color for non-method plots
    # currently unused: bcbd22
}

# # bar plot of number of configurations obtained by each searchspace constructor within the time limit
# df = pd.DataFrame.from_dict(results['num_configs'], orient='index')
# df = df.rename(index = searchspace_methods_displaynames)
# df.mean(axis=1).plot(kind='bar', yerr=df.std(axis=1), capsize=5, color=[searchspace_methods_colors_dict[s] for s in df.index])
# plt.xlabel('Searchspace construction method')
# plt.ylabel('Number of configurations evaluated (higher is better)')
# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.savefig('compare_real_world_number_of_evaluations.png', dpi=300)
# plt.show()

# # plot the performance of the configurations obtained by each searchspace constructor
# df = pd.DataFrame.from_dict(results['best_relative_performance'], orient='index')
# df = df.rename(index = searchspace_methods_displaynames)
# df.mean(axis=1).plot(kind='bar', yerr=df.std(axis=1), capsize=5, color=[searchspace_methods_colors_dict[s] for s in df.index])
# plt.xlabel('Searchspace construction method')
# plt.ylabel('Speedup found over the average performance (higher is better)')
# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.savefig('compare_real_world_speedup.png', dpi=300)
# plt.show()


# plot the performance over time by each searchspace constructor

# calculate the mean and standard deviation for each method over time
mean_performances = []
std_performances = []
first_non_nan_indices = []
first_non_nan_all = False    # if true, all iterations must be non-nan at the first index, otherwise only one iteration must be non-nan
for s in searchspace_constructors:
    r = np.array(results['configs_performance_time'][s])
    if first_non_nan_all:
        # get the first index where all iterations are not NaN
        first_non_nan_index = np.where(~np.isnan(r).any(axis=0))[0][0]
        assert np.all(~np.isnan(r[:, first_non_nan_index])), f"Not all iterations are non-nan at index {first_non_nan_index} for {s}"
    else:
        # get the first index where any iterations are not NaN
        first_non_nan_index = np.where(~np.isnan(r).all(axis=0))[0][0]
        assert np.any(~np.isnan(r[:, first_non_nan_index])), f"Not any iterations are non-nan at index {first_non_nan_index} for {s}"
    assert first_non_nan_index < len(minutes_line)
    # cut r to the first index where the iterations are not NaN
    r2 = r[:, first_non_nan_index:]
    # calculate the mean and std over r
    print(s, r2[:, -1])
    if first_non_nan_all:
        mean_performance = np.mean(r2, axis=0)
        std_performance = np.std(r2, axis=0)
    else:
        mean_performance = np.nanmean(r2, axis=0)
        std_performance = np.nanstd(r2, axis=0)
    # add back the NaNs removed earlier
    mean_performance = np.concatenate((np.full(first_non_nan_index, np.nan), mean_performance)) 
    std_performance = np.concatenate((np.full(first_non_nan_index, np.nan), std_performance)) 
    assert len(mean_performance) == len(minutes_line)
    assert len(std_performance) == len(minutes_line)
    mean_performances.append(mean_performance)
    std_performances.append(std_performance)
    first_non_nan_indices.append(first_non_nan_index)
# raise ValueError("stop")

# plot the performance over time
plt.figure(figsize=(7, 3.5))
# plot the performance over time for each searchspace constructor
for i, method in enumerate([searchspace_methods_displaynames[s] for s in searchspace_constructors]):
    plt.plot(minutes_line, mean_performances[i], label=method, color=searchspace_methods_colors_dict[method])
    first_non_nan_index = first_non_nan_indices[i]
    # fill the area between the mean and std
    mean_no_nan = mean_performances[i][first_non_nan_index:]
    std_no_nan = std_performances[i][first_non_nan_index:]
    plt.fill_between(
        minutes_line[first_non_nan_index:],
        mean_no_nan - std_no_nan,
        mean_no_nan + std_no_nan,
        color=searchspace_methods_colors_dict[method],
        alpha=0.2
    )

plt.xlabel('Tuning time in minutes')
plt.ylabel(f"Performance in {objective_displaynames[performance_objective]}")
# plt.ylabel(f"Performance in {objective_displaynames[performance_objective]} {'(higher is better)' if not minimize else '(lower is better)'}")
# plt.title('Performance over time by searchspace construction method')
if performance_objective == 'gridpoints/s':
    plt.ticklabel_format(axis='y', style='sci', scilimits=(9,9))
plt.legend(title='Method', loc='lower center')
plt.xlim(0, num_minutes)
plt.grid(True)
plt.tight_layout()
plt.savefig(f'compare_real_world_performance_over_time_{kernel}.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
# plt.show()
