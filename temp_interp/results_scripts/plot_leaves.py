import argparse
from pathlib import Path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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

def read_tensorboard_leaves(run_dir: Path):
    """
    Reads ``train/num_leaves`` from the tensorboard logs of a local run (see temp_interp/run/run_local.py).
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    for events in run_dir.rglob('events.out.tfevents.*'):
        if 'wandb' in events.parts:
            continue
        acc = EventAccumulator(str(events), size_guidance={'scalars': 0})
        acc.Reload()
        if 'train/num_leaves' in acc.Tags()['scalars']:
            scalars = acc.Scalars('train/num_leaves')
            return np.array([e.step for e in scalars]), smooth(np.array([e.value for e in scalars]), SMOOTHING_WEIGHT)
    print(f"No train/num_leaves found in {run_dir}")
    return None, None

def plot_experiment(experiment_name, runs, ax, color, linestyle='-'):
    """
    Aggregates the (steps, values) of multiple seeds and plots mean +/- s.e.
    """
    all_values = []
    common_steps = None

    # Load data from all seeds
    for steps, vals in runs:
        if steps is not None:
            # For simplicity, we assume all runs have roughly the same steps.
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
    se_curve = np.std(data_array, axis=0) / np.sqrt(len(all_values))

    # Plot Mean
    ax.plot(common_steps, mean_curve, label=experiment_name, color=color, linewidth=2, linestyle=linestyle)

    # Plot Shaded Se Dev
    ax.fill_between(common_steps,
                    mean_curve - se_curve,
                    mean_curve + se_curve,
                    color=color, alpha=0.2)

def plot_figure(title, experiments, filename, colors, linestyles=None):
    """
    One panel of Figure 2: a curve (mean +/- s.e. over seeds) per experiment.

    :param experiments: dict of curve name -> list of (steps, values), one per seed
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, runs) in enumerate(experiments.items()):
        plot_experiment(name, runs, ax, colors[i], linestyles[i] if linestyles else '-')

    ax.set_ylim(1.9, 4.1)  # same range on every panel
    ax.set_title(title, fontsize=28, pad=15, fontweight='bold')
    ax.set_xlabel("Time Steps", fontsize=28, fontweight='bold')
    ax.set_ylabel("Number of leaves", fontsize=28, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.xaxis.get_offset_text().set_fontsize(22)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, frameon=True, ncol=4, fontsize=22)

    plt.tight_layout()
    plt.savefig(f"{filename}.pdf")
    plt.savefig(f"{filename}.png")
    plt.close(fig)

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the number of leaves during training (Figure 2)')
    parser.add_argument('--root', type=Path, default=Path('results'), help='output directory of temp_interp/run/run_local.py')
    parser.add_argument('--out', type=Path, default=Path('.'), help='directory to save the plots in')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    envs = {"LK": "lk", "IP": "ip", "LL": "ll", "LL-H": "ll_hard"}
    seeds = [0, 1, 2]
    colors = sns.color_palette("deep", n_colors=len(envs))

    def runs(env, run_name):
        return [read_tensorboard_leaves(args.root / env / run_name / f"seed{seed}") for seed in seeds]

    # (a), (b): DDTs trained from scratch in every domain
    for algo, title in [("temp-ensemble", "DDT-Ensemble"), ("temp-pred", "DDT-Prediction")]:
        plot_figure(title, {name: runs(env, f"ddt_{algo}") for name, env in envs.items()}, args.out / f"leaves_plots_{algo}", colors)

    # (c): warm-started DDTs in LL-H, the only domain where ITTR restructures them
    plot_figure("Warm (LL-H)", {"Warm-Ensemble": runs("ll_hard", "warmcompact_temp-ensemble"),
                                "Warm-Prediction": runs("ll_hard", "warmcompact_temp-pred")},
                args.out / "leaves_plots_warm", colors, linestyles=['-', '--'])
