from datetime import datetime

# time between datestrings
dt_start = datetime.fromisoformat("2025-03-12 13:50:43.910064+00:00")
dt_end = datetime.fromisoformat("2025-03-12 19:38:17.721189+00:00")
print(dt_end-dt_start)

# calculated time without simulation mode
time_per_searchspace = {    # time budget in seconds per search space
    'convolution': {
        'MI250X': 85,
        'A100': 2550,
        'A4000': 800,
    },
    'dedispersion': {
        'MI250X': 1700,
        'A100': 2550,
        'A4000': 1700,
    },
    'hotspot': {
        'MI250X': 32,
        'A100': 75,
        'A4000': 21,
    },
    'gemm': {
        'MI250X': 850,
        'A100': 350,
        'A4000': 850,
    },
}

hyperparam_configs_per_algorithm = {
    'DA': 8,
    'GA': 108,
}

repeats = 25
for algorithm, num_configs in hyperparam_configs_per_algorithm.items():
    total_time = 0
    for kernel_name, gpu in time_per_searchspace.items():
        for gpu_name, time_budget in gpu.items():
            total_time += num_configs * time_budget
        # print(f"Time budget for {algorithm} with {kernel_name} on {gpu_name} is {num_configs*time_budget} seconds.")
    print(f"Total time for {algorithm} is {total_time / (60*60)} hours.")
