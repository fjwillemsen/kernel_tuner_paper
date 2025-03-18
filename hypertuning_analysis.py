import json

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

# file_prefix = "/Users/fjwillemsen/Downloads/new_0.95_25x/hyperparamtuning_paper_bruteforce_"
file_prefix = "/Users/fjwillemsen/Downloads/new_0.95_10x50x/hyperparamtuning_paper_bruteforce_"
file_suffix = ".json"

displaynames = {
    "basinhopping": "Basinhopping",
    "diff_evo": "Differential Evolution",
    "dual_annealing": "Dual Annealing",
    "genetic_algorithm": "Genetic Algorithm",
    "greedy_ils": "Greedy ILS",
    "mls": "MLS",
    "pso": "Particle Swarm Optimization",
    "simulated_annealing": "Simulated Annealing",
}

# consistent with aggregate plot colors
colors = ["Blues", "Greens", "Reds", "Purples", "Greys"]
color_palette = [sns.color_palette(c, 1)[0] for c in colors]

# score after re-execution on training data
training_scores = {
    "dual_annealing": {
        "best": -0.135,
        "worst": -0.974,
    },
    "genetic_algorithm": {
        "best": 0.108,
        "worst": -0.309,
    },
    "pso": {
        "best": 0.079,
        "worst": -0.549,
    },
    "simulated_annealing": {
        "best": -0.161,
        "worst": -0.321,
    },
}

# score after execution on test data
test_scores = {
    "dual_annealing": {
        "best": -0.245,
        "worst": -0.437,
    },
    "genetic_algorithm": {
        "best": -0.05,
        "worst": -0.345,
    },
    "pso": {
        "best": -0.051,
        "worst": -0.431,
    },
    "simulated_annealing": {
        "best": -0.175,
        "worst": -0.392,
    },
}


def load_data(json_files):
    """Load tuning results from JSON files into a dictionary of Pandas DataFrames."""
    dataframes = {}
    for file in json_files:
        with open(file, 'r') as f:
            filename = file.replace(file_prefix, "").replace(file_suffix, "")
            displayname = displaynames[filename]
            print(displayname)
            content = json.load(f)
            cache = content.get("cache", {})
            data = []
            for key, entry in cache.items():
                data.append(entry)
            dataframes[displayname] = pd.DataFrame(data)
    return dataframes

def plot_violin(dataframes):
    """Plot violin plots of the score distributions for multiple dataframes."""
    plt.figure(figsize=(8, 3.5), dpi=100)
    sns.set_style("whitegrid")
    combined_df = pd.concat([df.assign(file=file) for file, df in dataframes.items()])
    sns.violinplot(x="file", y="score", data=combined_df, inner="box", palette=color_palette)
    # plt.xticks(rotation=30, ha="right")
    plt.xlabel("Optimization Algorithm")
    plt.ylabel("Performance score")
    plt.ylim(None, 1.0)
    # plt.title("Score Distributions per Optimization Algorithm")
    plt.tight_layout()
    plt.savefig("tuning_violin_plot.png", dpi=300)
    plt.show()

def plot_dumbbell_chart(dataframes, training_scores, test_scores):
    """Creates a dumbbell chart using Seaborn, showing the score range for each phase with a unique color per algorithm, with a small x-axis offset to avoid overlap."""
    data = []
    phase_offset = -((len(dataframes) / 2) * 0.025)
    phase_positions = {"Tuning": 0, "Training": 1, "Test": 2}  # Base positions for each phase
    phase_offsets = {"Tuning": phase_offset, "Training": phase_offset, "Test": phase_offset}  # Small horizontal offsets
    
    for i, (algorithm_name, df) in enumerate(dataframes.items()):
        tuning_best = df["score"].max()
        tuning_worst = df["score"].min()
        algorithm = list(displaynames.keys())[list(displaynames.values()).index(algorithm_name)]
        training_best = training_scores.get(algorithm, {}).get("best", None)
        training_worst = training_scores.get(algorithm, {}).get("worst", None)
        test_best = test_scores.get(algorithm, {}).get("best", None)
        test_worst = test_scores.get(algorithm, {}).get("worst", None)
        
        if None not in [training_best, training_worst, test_best, test_worst]:
            data.append((displaynames.get(algorithm, algorithm), "Tuning", tuning_worst, tuning_best, phase_positions["Tuning"] + phase_offsets["Tuning"] + i * 0.05))
            data.append((displaynames.get(algorithm, algorithm), "Training", training_worst, training_best, phase_positions["Training"] + phase_offsets["Training"] + i * 0.05))
            data.append((displaynames.get(algorithm, algorithm), "Test", test_worst, test_best, phase_positions["Test"] + phase_offsets["Test"] + i * 0.05))
    
    df_plot = pd.DataFrame(data, columns=["Algorithm", "Phase", "Worst", "Best", "Offset"])
    
    plt.figure(figsize=(8, 4.5))
    sns.set_style("whitegrid")
    
    for i, algo in enumerate(df_plot["Algorithm"].unique()):
        df_algo = df_plot[df_plot["Algorithm"] == algo]
        color = color_palette[i]
        sns.scatterplot(x=df_algo["Offset"], y=df_algo["Worst"], color=color, label=algo, s=100)
        sns.scatterplot(x=df_algo["Offset"], y=df_algo["Best"], color=color, s=100)
        for _, row in df_algo.iterrows():
            plt.plot([row["Offset"], row["Offset"]], [row["Worst"], row["Best"]], color=color, linewidth=2)
    
    plt.xlabel("Phase")
    plt.ylabel("Performance score")
    plt.ylim(None, 1.0)
    # plt.title("Best and worst scores across tuning, training, and test phases")
    plt.xticks(ticks=list(phase_positions.values()), labels=list(phase_positions.keys()))
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.savefig("tuning_training_test_dumbbell_chart.png", dpi=300)
    plt.show()

def score_difference(dataframes):
    """Report the score and magnitude difference between the best and worst configs."""
    score_diff_sum = 0
    for file, df in dataframes.items():
        best_score = df["score"].max()
        worst_score = df["score"].min()
        score_diff = best_score - worst_score
        score_diff_sum += score_diff
        print(f"Score difference for {file}: {score_diff:.4f} (best={best_score:.4f}, worst={worst_score:.4f})")
    print(f"Average score difference: {score_diff_sum/len(dataframes):.4f}")

def analyze_hyperparameter_influence(dataframes):
    """Perform ANOVA to analyze the influence of categorical hyperparameters on score for each file."""
    for file, df in dataframes.items():
        print(f"Analysis for {file}")
        param_keys = [col for col in df.columns if col not in {"score", "scores", "timestamp", "compile_time", "verification_time", "benchmark_time", "strategy_time", "framework_time"}]
        for param in param_keys:
            if param in df.columns:
                groups = [df[df[param] == val]["score"] for val in df[param].unique()]
                f_stat, p_value = stats.f_oneway(*groups)
                print(f"  ANOVA for {param}: F={f_stat:.4f}, p={p_value:.4e}")
                if p_value < 0.20:
                    print(f"    {param} has a significant effect on score.")
                # else:
                #     print(f"    {param} does not have a significant effect on score.")
        print("\n")

def analyze_hyperparameter_influence_non_parametric(dataframes):
    """Perform Kruskal-Wallis test to analyze the influence of categorical hyperparameters on score for each file."""
    for file, df in dataframes.items():
        print(f"Analysis for {file}")
        param_keys = [col for col in df.columns if col not in {"score", "scores", "timestamp", "compile_time", "verification_time", "benchmark_time", "strategy_time", "framework_time"}]
        for param in param_keys:
            if param in df.columns:
                groups = [df[df[param] == val]["score"] for val in df[param].unique()]
                h_stat, p_value = stats.kruskal(*groups)
                print(f"  Kruskal-Wallis for {param}: H={h_stat:.4f}, p={p_value:.4e}")
                if p_value < 0.20:
                    print(f"    {param} has a significant effect on score.")
                if h_stat < 1.0:
                    print(f"    {param} has low H-value.")
                # else:
                    # print(f"    {param} does not have a significant effect on score.")
        print("\n")

def analyze_hyperparameter_mutual_info(dataframes):
    """Compute mutual information scores for categorical hyperparameters against the score."""
    for file, df in dataframes.items():
        print(f"Mutual Information Analysis for {file}")
        param_keys = [col for col in df.columns if col not in {"score", "scores", "timestamp", "compile_time", "verification_time", "benchmark_time", "strategy_time", "framework_time"}]
        df_encoded = df.copy()
        for param in param_keys:
            if df_encoded[param].dtype == 'object':
                df_encoded[param] = df_encoded[param].astype("category").cat.codes
        
        mi_scores = mutual_info_regression(df_encoded[param_keys], df_encoded["score"], discrete_features=True)
        
        for param, score in zip(param_keys, mi_scores):
            print(f"  Mutual Information for {param}: {score:.4f}")
            if score > 0.1:
                print(f"    {param} has a strong influence on score.")
            elif score > 0.02:
                print(f"    {param} has a moderate influence on score.")
            else:
                print(f"    {param} has little to no influence on score.")
        print("\n")

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python script.py file1.json file2.json ...")
    #     sys.exit(1)
    # json_files = sys.argv[1:]

    json_files = [
        # "basinhopping",
        # "diff_evo", 
        "dual_annealing", 
        "genetic_algorithm", 
        # "greedy_ils",
        # "mls", 
        "pso",
        "simulated_annealing",
    ]
    for i in range(len(json_files)):
        json_files[i] = file_prefix + json_files[i] + file_suffix

    dataframes = load_data(json_files)
    assert len(dataframes) == len(json_files), "Error: Not all JSON files were loaded."
    
    for file, df in dataframes.items():
        if "score" not in df.columns:
            raise ValueError(f"Error: No 'score' column found in {file}.")

    plot_violin(dataframes)
    # plot_grouped_bar_chart(dataframes, training_scores, test_scores)
    plot_dumbbell_chart(dataframes, training_scores, test_scores)
    score_difference(dataframes)
    # # analyze_hyperparameter_influence(dataframes)
    # analyze_hyperparameter_influence_non_parametric(dataframes)
    # analyze_hyperparameter_mutual_info(dataframes)
