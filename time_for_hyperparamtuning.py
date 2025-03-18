from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# settings
rounding = 1
repeats = 25

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

hyperparam_tuning_per_algorithm = {
    # 'BH': {
    #     'configs': 56,
    #     'start_time': "2025-03-12 13:29:59.750036+00:00",
    #     'end_time': "2025-03-13 22:11:20.786336+00:00",
    # },
    # 'DE': {
    #     'configs': 90,
    #     'start_time': "2025-03-12 13:26:15.797900+00:00",
    #     'end_time': "2025-03-14 07:53:47.424864+00:00",
    # },
    'DA': {
        'configs': 8,
        'start_time': "2025-03-12 13:35:52.423869+00:00",
        'end_time': "2025-03-12 18:06:16.839432+00:00",
    },
    'GA': {
        'configs': 108,
        'start_time': "2025-03-12 13:23:00.128063+00:00",
        'end_time': "2025-03-14 20:05:00.711049+00:00",
    },
    # 'GILS': {
    #     'configs': 80,
    #     'start_time': "2025-03-12 13:25:58.183128+00:00",
    #     'end_time': "2025-03-14 06:45:05.872338+00:00",
    # },
    # 'MLS': {
    #     'configs': 8,
    #     'start_time': "2025-03-12 13:50:43.910064+00:00",
    #     'end_time': "2025-03-12 19:38:17.721189+00:00",
    # },
    'PSO': {
        'configs': 81,
        'start_time': "2025-03-12 13:27:57.184038+00:00",
        'end_time': "2025-03-14 08:56:05.257832+00:00",
    },
    'SA': {
        'configs': 81,
        'start_time': "2025-03-12 13:20:22.494310+00:00",
        'end_time': "2025-03-14 04:12:07.103166+00:00",
    },
}

displaynames = {
    "BH": "Basinhopping",
    "DE": "Differential Evolution",
    "DA": "Dual Annealing",
    "GA": "Genetic Algorithm",
    "GILS": "Greedy ILS",
    "MLS": "MLS",
    "PSO": "Particle Swarm Optimization",
    "SA": "Simulated Annealing",
}

# Data storage for visualization
algorithms = []
live_times = []
simulated_times = []

# calculate per algorithm
sum_simulated_time = 0
sum_live_time = 0
for algorithm, data in hyperparam_tuning_per_algorithm.items():

    # calculate the time needed for simulation mode
    start_time = datetime.fromisoformat(data['start_time'])
    end_time = datetime.fromisoformat(data['end_time'])
    simulated_time = (end_time - start_time).total_seconds()
    sum_simulated_time += simulated_time

    # calculate the time needed for live tuning
    num_configs = data['configs']
    live_time = 0
    for kernel_name, gpu in time_per_searchspace.items():
        for gpu_name, time_budget in gpu.items():
            live_time += num_configs * time_budget * repeats
        # print(f"Time budget for {algorithm} with {kernel_name} on {gpu_name} is {num_configs*time_budget} seconds.")
    sum_live_time += live_time

    # Store data for visualization
    algorithms.append(displaynames[algorithm])
    live_times.append(live_time / 3600)  # Convert to hours
    simulated_times.append(simulated_time / 3600)  # Convert to hours

    # report the difference
    print(f"| Live-tuning time for {algorithm} is {round(live_time / (60*60), rounding)} hours, against {round(simulated_time / (60*60), rounding)} hours using simulation")

# report the total and average difference
average_live_time = sum_live_time / len(hyperparam_tuning_per_algorithm)
average_simulated_time = sum_simulated_time / len(hyperparam_tuning_per_algorithm)
print(f"Average live-tuning time is {round(average_live_time / (60*60), rounding)} hours, against {round(average_simulated_time / (60*60), rounding)} hours using simulation (speedup: {round(average_live_time / average_simulated_time, rounding)}x)")
print(f"Total live-tuning time is {round(sum_live_time / (60*60), rounding)} hours, against {round(sum_simulated_time / (60*60), rounding)} hours using simulation (speedup: {round(sum_live_time / sum_simulated_time, rounding)}x)")

# Create a bar chart
plt.figure(figsize=(8, 4))
df = {
    'Algorithm': algorithms * 2,
    'Time (hours)': simulated_times + live_times,
    'Mode': ['Simulation'] * len(algorithms) + ['Live'] * len(algorithms)
}
sns.barplot(x='Algorithm', y='Time (hours)', hue='Mode', data=df)
# plt.title('Comparison of Live and Simulated Tuning Times per Algorithm')
plt.xlabel('Algorithm')
plt.ylabel('Time (hours)')
plt.yscale('log')
# plt.xticks(rotation=45)
plt.legend(title='Mode', loc='upper left')
plt.tight_layout()
plt.savefig("tuning_time_comparison.png", dpi=300)
plt.show()
