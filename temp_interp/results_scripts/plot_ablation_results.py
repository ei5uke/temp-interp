import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.weight'] = 'bold'

sns.set_theme(style="dark", context="paper", font_scale=1.2)

def plot_experiment(decays, rewards, ax, colors):
    """
    Aggregates data from multiple CSVs (seeds) and plots mean +/- std dev.
    """
    markers = ['o', 'x', '*']
    # Plot scatter points
    for i in range(len(decays)):
        ax.scatter(decays[i], rewards[i], label=f"Seed {i}", marker=markers[i], s=100)

    x = np.concatenate(decays)
    y = np.concatenate(rewards)
    coeffs = np.polyfit(x, y, 2)
    poly_function = np.poly1d(coeffs)
    x_smooth = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_smooth, poly_function(x_smooth), color='black', linewidth=2)

def get_results(root: Path, env: str, seeds, etas):
    """
    Best evaluation return of each DDT-Ensemble run per seed and EMA coefficient.
    eta = 0.75 is the main DDT-Ensemble run; the others come from the R9 ablation of temp_interp/run/run_local.py.
    """
    from temp_interp.results_scripts.summarize_local import best_eval_return

    decays, rewards = [], []
    for seed in seeds:
        seed_decays, seed_rewards = [], []
        for eta in etas:
            run_name = 'ddt_temp-ensemble' if eta == 0.75 else f'ddt_temp-ensemble_ema{eta}'
            run_dir = root / env / run_name / f'seed{seed}'
            if not (run_dir / 'DONE').exists():
                continue
            seed_decays.append(eta)
            seed_rewards.append(best_eval_return(run_dir))
        decays.append(np.array(seed_decays))
        rewards.append(np.array(seed_rewards))
    return decays, rewards

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the EMA coefficient ablation (Figure 5)')
    parser.add_argument('--root', type=Path, default=Path('results'), help='output directory of temp_interp/run/run_local.py')
    parser.add_argument('--out', type=Path, default=Path('ablation_plots'), help='directory to save the plots in')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    etas = [0.5, 0.625, 0.75, 0.875, 1.0]
    envs = {'ip': ('ip', 'Inverted Pendulum v5'), 'lk': ('lk', 'Lane Keeping'),
            'll': ('ll', 'Lunar Lander v3'), 'll-h': ('ll_hard', 'Lunar Lander v3 Hard')}
    for name, (env, fullenv) in envs.items():
        decays, rewards = get_results(args.root, env, [0, 1, 2], etas)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = sns.color_palette("deep", n_colors=len(decays))
        plot_experiment(decays, rewards, ax, colors)

        ax.set_title(f"{fullenv} Ablation", fontsize=28, pad=15, fontweight='bold')
        ax.set_xlabel("EMA coefficient", fontsize=28, fontweight='bold')
        ax.set_ylabel("Average Return", fontsize=28, fontweight='bold')
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, frameon=True, ncol=3, fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.xaxis.get_offset_text().set_fontsize(22)

        plt.savefig(args.out / f"{name}.pdf", bbox_inches='tight')
        plt.savefig(args.out / f"{name}.png", bbox_inches='tight')
        plt.close(fig)
