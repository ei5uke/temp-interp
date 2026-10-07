# Temporal Interpretability [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
Codebase for: "Temporally Interpretable Differentiable Decision Trees"

Authors: Eisuke Hirota, Aarav Sane, Rohan Paleja

## Installation
```
conda create --name temp-interp python=3.11
conda activate temp-interp
git clone https://github.com/ei5uke/temp-interp.git
cd temp-interp/
pip install swig  # needed to build Box2D
pip install -r requirements.txt
# optionally: wandb login (otherwise set WANDB_MODE=offline)
```

## Running experiments
Environments (`--env_name`): `lane_keeping` (LK), `cart` (IP), `lunar` (LL), `lunar-hard` (LL-H).
DDT and ITTR settings per environment are in `ITTR_CONFIGS` of `temp_interp/run/train.py`.

| Method | Script | Key flags |
|---|---|---|
| DDT / MLP (no action chunking) | `train.py` / `train_mlp.py` | `--abstraction_type temp-pred --time_horizon 1` |
| Temporal Ensemble | `train.py` / `train_mlp.py` | `--abstraction_type temp-ensemble --time_horizon 10` (EMA coefficient: `--decay`, default 0.75) |
| Temporal Prediction | `train.py` / `train_mlp.py` | `--abstraction_type temp-pred --time_horizon 10` (scaling coefficient: `--pred_coef`) |
| MLP size | `train_mlp.py` | `--mlp_policy_size mid` (MLP, [16, 16]) or `max` (MLP-Big, [64, 64]) |
| CART | `train.py` | `--cart_teacher <MLP-Big model.zip>` (same abstraction type and horizon as the teacher) |
| Warm | `train.py` | `--cart_teacher <MLP-Big model.zip> --warm_start` |

ITTR presets can be overridden with `--epsilon`, `--min_timesteps`, `--min_headstart` and `--min_visitations`.
Add `--no_sweep` to run once without a W&B sweep.

Single run, e.g. DDT with Temporal Ensemble on LL:
```
python -m temp_interp.run.train --env_name lunar --abstraction_type temp-ensemble --time_horizon 10 \
  --num_envs 32 --n_steps 2048 --batch_size 256 --gamma 0.99 --ent-coef 0.0 --training_steps 3000000 \
  --use_individual_alpha --hard_node --submodels --sparse_submodel_type 0 --no_sweep --save_path results/ll/ddt/
```

### Slurm cluster
`temp_interp/run/train_<env>_all.slurm` (DDT, CART, Warm) and `train_<env>_mlp_all.slurm` (MLP), with `<env>` in `ip`, `lk`, `ll`, `ll_hard`, hold the settings of the paper. They are configured through environment variables:
```
sbatch --export=ALL,SEED=0,METHOD=temp-pred,TIME=10 temp_interp/run/train_ll_all.slurm
sbatch --export=ALL,SEED=0,METHOD=temp-ensemble,TIME=10,SIZE=max temp_interp/run/train_ll_mlp_all.slurm
sbatch --export=ALL,SEED=0,METHOD=temp-pred,TIME=10,RUN_NAME=warm_temp-pred,EXTRA_ARGS="--cart_teacher <path> --warm_start" temp_interp/run/train_ll_all.slurm
```
`run_all.sh` submits the main action-chunked runs (H = 10) for 3 seeds.

### Local machine
`run_local.py` runs every experiment of the paper with the slurm settings, a few jobs at a time and without slurm. It starts CART and Warm runs once their MLP-Big teacher has finished, and skips finished runs when restarted.
```
python -m temp_interp.run.run_local --root results --workers 6 --python $(which python)
```

### Results
```
python -m temp_interp.results_scripts.summarize_local --root results              # Table 1 (return / parameters)
python -m temp_interp.results_scripts.plot_leaves --root results --out figures    # number of leaves during training
python -m temp_interp.results_scripts.plot_ablation_results --root results --out figures/ablation  # EMA coefficient ablation
bash temp_interp/run/test_robustness.sh                                           # temporal robustness verification (set the model path inside)
```

## Important Notes
Our contributions introduced in the paper theoretically work for any DDT variant. In this codebase, we particularly test our results with the ICCT-Complete variant, based on [Paleja+2022]. We advise that if one were to implement other variants, this code will likely require adjustments.

[Paleja+2022] Learning Interpretable, High-Performing Policies for Autonomous Driving. RSS 2022.

## Acknowledgements
We acknowledge the following resources that have helped our project.
- [ICCTs](https://github.com/CORE-Robotics-Lab/ICCT?tab=readme-ov-file)
- [Transparent Trees](https://github.com/CORE-Robotics-Lab/Team-Development-with-Transparent-Policies)
- [DDTs](https://github.com/CORE-Robotics-Lab/Interpretable_DDTS_AISTATS2020)
- [StableBaselines3](https://github.com/DLR-RM/stable-baselines3)
- [CleanRL](https://github.com/vwxyzjn/cleanrl/tree/master)
- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [Reasons against explicit evaluation](https://github.com/vwxyzjn/cleanrl/issues/310)
- [InfoNCE loss](https://github.com/sthalles/SimCLR/tree/master)