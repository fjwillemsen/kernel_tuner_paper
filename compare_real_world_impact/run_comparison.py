""""Run the tuning of each combination of kernel and platform for each searchspace constructor."""

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kernels.hotspot.hotspot import tune as tune_hotspot

# beware this code currently has some assumptions that we use a single searchspace (kernel+device+inputs combination)!
kernels = ["hotspot"]           # names of the kernel and folder in the kernels folder (must be the same)
platforms = [("CUDA", "A4000")] # tuple of language and device, for language choose from CUDA, HIP and OpenCL
iterations = 10                 # number of times to repeat each tuning run
num_minutes = 20                # time limit for each tuning run in minutes
minimize = True                 # whether to minimize the objective function (time) or maximize it (performance)
searchspace_constructors = [    # the searchspace construction frameworks to use
    "pythonconstraint",
    "pyatf",
    "bruteforce",
]

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
                cachefile_path = Path(f"results/hotspot/{device.upper()}_f={searchspace_constructor}_i={iteration}.json")
                if cachefile_path.exists():
                    print(f"    skipping {searchspace_constructor} (iter. {iteration}) as it already exists")
                    continue

                # set the tuning parameters
                strategy_options = {
                    'time_limit': num_minutes*60,     # time limit in seconds
                    'searchspace_construction_options': {
                        'framework': searchspace_constructor
                    }
                }

                # tune
                res, env = tune_hotspot(
                    device_name=device, 
                    strategy="random_sample", 
                    strategy_options=strategy_options,
                    simulation_mode=False,
                    lang=language,
                    verbose=False,
                    cachefile_path=str(cachefile_path)
                )
                print(f"{searchspace_constructor} (iter. {iteration}) evaluated {len(res)} configs in ~20 minutes")

                # assert the cache file path now exists
                assert cachefile_path.exists(), f"Cachefile does not exist at {cachefile_path}"

    print("|------------------------")
print("  Finished tuning runs ")
print("")

# aggregate the results
results = {
    'num_configs': {},
    'configs_performance': {},
    'best_relative_performance': {}
}
for searchspace_constructor in searchspace_constructors:
    results['num_configs'][searchspace_constructor] = []
    results['configs_performance'][searchspace_constructor] = []
    for kernel in kernels:
        for language, device in platforms:
            for iteration in range(iterations):
                # create the cache path and skip this combination if it already exists
                cachefile_path = Path(f"results/hotspot/{device.upper()}_f={searchspace_constructor}_i={iteration}.json")

                # for each searchspace constructor, aggregate the results
                if cachefile_path.exists():
                    # get the file created time as a date
                    file_creation_time = datetime.fromtimestamp(cachefile_path.stat().st_mtime)
                    with cachefile_path.open("r") as f:
                        data = json.load(f)
                        cache = data['cache']
                        num_configs = len(cache)
                        results['num_configs'][searchspace_constructor].append(num_configs)
                        configs_performance = [c['time'] for c in cache.values() if 'time' in c and isinstance(c['time'], (int, float))]
                        results['configs_performance'][searchspace_constructor].append(configs_performance)
                        for k, v in cache.items():
                            config_timestamp = datetime.fromisoformat(v['timestamp'])
                            # raise ValueError(f"File time for {cachefile_path}: {file_creation_time}, config: {config_timestamp}")

# get the average performance over all searchspace constructors
avg_performance = np.mean(np.array([c for s in searchspace_constructors for i in results['configs_performance'][s] for c in i]).flatten())
# set the best performance for each searchspace constructor relative to the average
for s in searchspace_constructors:
    # for each iteration get the best performance relative to the overall average
    results['best_relative_performance'][s] = []
    for i in range(iterations):
        # get the performance of the configurations for this iteration
        configs_performance = np.array(results['configs_performance'][s][i])
        # find the best performance obtained
        best_performance = min(configs_performance) if minimize else max(configs_performance)
        # calculate the speedup of the best performance over the average performance
        assert minimize, "This code assumes we are minimizing the performance metric"
        results['best_relative_performance'][s].append(avg_performance / best_performance)

# plot the results

# bar plot of number of configurations obtained by each searchspace constructor within the time limit
df = pd.DataFrame.from_dict(results['num_configs'], orient='index')
df.mean(axis=1).plot(kind='bar', yerr=df.std(axis=1), capsize=4)
plt.xlabel('Searchspace construction method')
plt.ylabel('Number of configurations')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# plot the performance of the configurations obtained by each searchspace constructor
df = pd.DataFrame.from_dict(results['best_relative_performance'], orient='index')
df.mean(axis=1).plot(kind='bar', yerr=df.std(axis=1), capsize=4)
plt.xlabel('Searchspace construction method')
plt.ylabel('Speedup found over the average performance')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
