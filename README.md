# Temporal Interpretability [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
Codebase for: "Action Chunking for Temporal Interpretability in Differential Decision Trees"

Authors: Eisuke Hirota, Rohan Paleja

## Installation
```
module load conda # may be optional
conda create --name temp-interp python=3.9
conda activate temp-interp
git clone https://github.com/ei5uke/temp-interp.git
cd temp-interp/
pip install -e .
```

## Training
Debug / personal computer testing:
```
bash temp_interp/run/train_{environment_name}.sh
```

slurm cluster:
```
sbatch temp_interp/run/train_{environment_name}.slurm
```

## TO-DO
### Overall
- [ ] Add PPO w/ ICCT (one thing to note is that the policy gradient loss is really low)
- [ ] Add PPO w/ ICCT + Action Chunking
- [ ] Add MINE/InfoNCE within PPO `learn()`

### Known bugs
- [x] `ppo.py`: `self.ent_coef` is equal to `'auto'` when printed. It should not be a str() but rather a float.
    - (Oct 17, 2025) Fixed. Issue was that `'auto'` was being passed to PPO() in `train.py` as a parameter. Change to a float. 

## Acknowledgements
We acknowledge the following resources that have helped our project.
- [ICCTs](https://github.com/CORE-Robotics-Lab/ICCT?tab=readme-ov-file)
- [DDTs](https://github.com/CORE-Robotics-Lab/Interpretable_DDTS_AISTATS2020)
- [StableBaselines3](https://github.com/DLR-RM/stable-baselines3)
- [CleanRL](https://github.com/vwxyzjn/cleanrl/tree/master)