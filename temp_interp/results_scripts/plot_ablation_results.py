import pandas as pd
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

    x = np.array(decays).flatten()
    y = np.array(rewards).flatten()
    coeffs = np.polyfit(x, y, 2)
    poly_function = np.poly1d(coeffs)
    x_smooth = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_smooth, poly_function(x_smooth), color='black', linewidth=2)

def get_results(csv_files):
    """
    Aggregates data from multiple CSVs (seeds) and finds the max result (mean +/- s.e.).
    """

    # Load data from all seeds
    decays = []
    rewards = []
    for f in csv_files:
        df = pd.read_csv(f, usecols=['decay', 'eval/mean_reward'])
        decays.append(df['decay'].to_numpy())
        rewards.append(df['eval/mean_reward'].to_numpy())

    return decays, rewards

# --- Main Execution ---

idx = 3
envs = ['ip', 'lk', 'll', 'll-h']
env = envs[idx]
fullenvs = ['Inverted Pendulum v5', 'Lane Keeping', 'Lunar Lander v3', 'Lunar Lander v3 Hard']
fullenv = fullenvs[idx]
experiments = [f"ablation_results_EMA_coef/temp-ensemble_{env}_0.csv", f"ablation_results_EMA_coef/temp-ensemble_{env}_1.csv", f"ablation_results_EMA_coef/temp-ensemble_{env}_2.csv"]

decays, rewards = get_results(experiments)

fig, ax = plt.subplots(figsize=(10, 6))

colors = sns.color_palette("deep", n_colors=len(experiments))

plot_experiment(decays, rewards, ax, colors)

ax.set_title(f"{fullenv} Ablation", fontsize=28, pad=15, fontweight='bold')
ax.set_xlabel("EMA coefficient", fontsize=28, fontweight='bold')
ax.set_ylabel("Average Return", fontsize=28, fontweight='bold')
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, frameon=True, ncol=3, fontsize=22)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.xaxis.get_offset_text().set_fontsize(22)

plt.savefig(f"ablation_plots/{env}.pdf", bbox_inches='tight')
plt.savefig(f"ablation_plots/{env}.png", bbox_inches='tight')
# plt.show()