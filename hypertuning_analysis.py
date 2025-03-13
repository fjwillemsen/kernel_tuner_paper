import json

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

file_prefix = "/Users/fjwillemsen/Downloads/new_0.95_20x/hyperparamtuning_paper_bruteforce_"
file_suffix = ".json"


def load_data(json_files):
    """Load tuning results from JSON files into a dictionary of Pandas DataFrames."""
    dataframes = {}
    for file in json_files:
        with open(file, 'r') as f:
            filename = file.replace(file_prefix, "").replace(file_suffix, "")
            print(filename)
            content = json.load(f)
            cache = content.get("cache", {})
            data = []
            for key, entry in cache.items():
                data.append(entry)
            dataframes[filename] = pd.DataFrame(data)
    return dataframes

def plot_violin(dataframes):
    """Plot violin plots of the score distributions for multiple dataframes."""
    plt.figure(figsize=(10, 6))
    combined_df = pd.concat([df.assign(file=file) for file, df in dataframes.items()])
    sns.violinplot(x="file", y="score", data=combined_df, inner="point", scale="width")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Optimization Algorithm")
    plt.ylabel("Score")
    plt.title("Score Distributions per Optimization Algorithm")
    plt.tight_layout()
    plt.show()

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
        "basinhopping", 
        # "diff_evo", 
        # "dual_annealing", 
        "genetic_algorithm", 
        "greedy_ils",
        "mls", 
        "pso"
    ]
    for i in range(len(json_files)):
        json_files[i] = file_prefix + json_files[i] + file_suffix

    dataframes = load_data(json_files)
    
    for file, df in dataframes.items():
        if "score" not in df.columns:
            raise ValueError(f"Error: No 'score' column found in {file}.")

    plot_violin(dataframes)
    # analyze_hyperparameter_influence(dataframes)
    analyze_hyperparameter_influence_non_parametric(dataframes)
    analyze_hyperparameter_mutual_info(dataframes)
