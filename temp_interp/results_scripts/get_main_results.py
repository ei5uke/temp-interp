import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.weight'] = 'bold'

sns.set_theme(style="dark", context="paper", font_scale=1.2)

SMOOTHING_WEIGHT = 0.0

def smooth(scalars, weight):
    """
    Exponential moving average implementation similar to Tensorboard/WandB.
    """
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def read_and_process_wandb_csv(filepath, value_column_name='Value'):
    """
    Reads a CSV, sorts by step, and smooths the value column.
    Assumes standard WandB export format (Step, <MetricName>, MIN, MAX).
    """
    try:
        df = pd.read_csv(filepath)
        
        # W&B CSVs often have a 'Step' column. 
        # If your step column is named differently (e.g., 'global_step'), change it here.
        if 'global_step' not in df.columns:
            # Fallback: try to find a column that looks like a step or index
            df['global_step'] = df.index
            
        # Sort just in case
        df = df.sort_values('global_step')
        
        # We need to find the column with the actual values. 
        # Sometimes W&B exports complex names like "run-id - reward".
        # This logic finds the column that isn't "Step", "MIN", or "MAX".
        # target_col = [c for c in df.columns if c not in ['Step', 'MIN', 'MAX']][0]
        target_col = [c for c in df.columns if ('MIN' not in c) and ('MAX' not in c) and ('step' not in c) and ('Step' not in c)][0]
        
        # Extract and smooth
        values = df[target_col].values
        # Handle NaNs if steps align differently across runs
        mask = ~np.isnan(values)
        steps = df['global_step'].values[mask]
        values = values[mask]
        
        smoothed_values = smooth(values, SMOOTHING_WEIGHT)
        
        return steps, smoothed_values
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None

def plot_experiment(experiment_name, csv_files, ax, color):
    """
    Aggregates data from multiple CSVs (seeds) and plots mean +/- std dev.
    """
    all_values = []
    common_steps = None

    # Load data from all seeds
    for f in csv_files:
        steps, vals = read_and_process_wandb_csv(f)
        if steps is not None:
            # For simplicity, we assume all CSVs have roughly the same steps.
            # If lengths differ slightly, we trim to the shortest run.
            if common_steps is None or len(steps) < len(common_steps):
                common_steps = steps
            all_values.append(vals)

    # Trim all runs to the length of the shortest run to align them
    min_len = len(common_steps)
    all_values = [v[:min_len] for v in all_values]
    
    # Convert to numpy array for easy math (rows=seeds, cols=steps)
    data_array = np.array(all_values)
    
    # Calculate Mean and Se Dev
    mean_curve = np.mean(data_array, axis=0)
    se_curve = np.std(data_array, axis=0) / np.sqrt(3) # b/c we use 3 seeds
    
    # Plot Mean
    ax.plot(common_steps, mean_curve, label=experiment_name, color=color, linewidth=2)
    
    # Plot Shaded Se Dev
    ax.fill_between(common_steps, 
                    mean_curve - se_curve, 
                    mean_curve + se_curve, 
                    color=color, alpha=0.2)
    
def get_results(experiment_name, csv_files):
    """
    Aggregates data from multiple CSVs (seeds) and finds the max result (mean +/- s.e.).
    """
    all_values = []
    common_steps = None

    # Load data from all seeds
    for f in csv_files:
        steps, vals = read_and_process_wandb_csv(f)
        if steps is not None:
            # For simplicity, we assume all CSVs have roughly the same steps.
            # If lengths differ slightly, we trim to the shortest run.
            if common_steps is None or len(steps) < len(common_steps):
                common_steps = steps
            all_values.append(vals)

    # Trim all runs to the length of the shortest run to align them
    min_len = len(common_steps)
    all_values = [v[:min_len] for v in all_values]
    
    # Convert to numpy array for easy math (rows=seeds, cols=steps)
    data_array = np.array(all_values)
    max_reward = data_array.max(axis=-1)
    mean_reward = max_reward.mean()
    se_reward = max_reward.std() / np.sqrt(max_reward.shape[0])
    print(f"Mean: {mean_reward}; SE: {se_reward}")

# --- Main Execution ---

idx = 3
envs = ['ip', 'lk', 'll', 'll-h']
env = envs[idx]
fullenvs = ['Inverted Pendulum v5', 'Lane Keeping', 'Lunar Lander v3', 'Lunar Lander v3 Hard']
fullenv = fullenvs[idx]
experiments = {
    "DDT": [f"main_results_NoAC/ddt_{env}_seed0.csv", f"main_results_NoAC/ddt_{env}_seed1.csv", f"main_results_NoAC/ddt_{env}_seed2.csv"],
    # "DDT-Ensemble": [f"main_results_temp-ensemble/ddt_{env}_seed0.csv", f"main_results_temp-ensemble/ddt_{env}_seed1.csv", f"main_results_temp-ensemble/ddt_{env}_seed2.csv"],
    # "DDT-Prediction": [f"main_results_temp-pred/ddt_{env}_seed0.csv", f"main_results_temp-pred/ddt_{env}_seed1.csv", f"main_results_temp-pred/ddt_{env}_seed2.csv"],
    # "MLP": [f"main_results_NoAC/mlp_{env}_seed0.csv", f"main_results_NoAC/mlp_{env}_seed1.csv", f"main_results_NoAC/mlp_{env}_seed2.csv"],
    # "MLP-Ensemble": [f"main_results_temp-ensemble/mlp_{env}_seed0.csv", f"main_results_temp-ensemble/mlp_{env}_seed1.csv", f"main_results_temp-ensemble/mlp_{env}_seed2.csv"],
    # "MLP-Prediction": [f"main_results_temp-pred/mlp_{env}_seed0.csv", f"main_results_temp-pred/mlp_{env}_seed1.csv", f"main_results_temp-pred/mlp_{env}_seed2.csv"],
}

for i, (name, files) in enumerate(experiments.items()):
    get_results(name, files)