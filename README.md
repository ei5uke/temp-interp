# Temporal Interpretability [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
Codebase for: "Action Chunking for Temporal Interpretability in Differential Decision Trees"

Authors: Eisuke Hirota, Aarav Sane, Rohan Paleja

## Installation
```
module load conda # may be optional
conda create --name temp-interp python=3.9
conda activate temp-interp
git clone https://github.com/ei5uke/temp-interp.git
cd temp-interp/
pip install -e .
# optionally: follow wandb api key login
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

## Important Notes
Our contributions introduced in the paper theoretically work for any DDT variant. In this codebase, we particularly test our results with the ICCT-Complete variant, based on [Paleja+2022]. We advise that if one were to implement other variants, this code will likely require adjustments.

[Paleja+2022] Learning Interpretable, High-Performing Policies for Autonomous Driving. RSS 2022.

## TO-DO
### Overall
- [x] Add PPO w/ ICCT ~~(one thing to note is that the policy gradient loss is really low)~~
    - (Oct 20, 2025) performed experiments (IP, LL, LK) with results matching SAC in original paper.
- [x] Action Chunking w/ Temporal Ensemble
    - [x] Add PPO w/ ICCT + Action Chunking
        - ~~(Nov 11, 2025) implemented but results are not equal~~
        - (Nov 12, 2025) fixed issues, LL performance match results
    - [x] The same ^ but for MLPs
        - ~~(Nov 13, 2025) implemented, now testing~~
        - (Nov 15, 2025) similar performance to DDTs, but more parameters. Should in future test with sparse MLPs
- [x] Action Chunking w/ Temporal Prediction
        - (Nov 17, 2025) implemented, now testing with a basic linear schedule and no curriculum
- [x] Add InfoNCE within PPO `learn()`
    - (Oct 28, 2025) infobottleneck with the leaf probabilities
    - (Oct 28, 2025) add policy complexity analysis
- [x] Fix up and merge MLP code
- [x] Policy complexity dynamic deepening / pruning
    - [x] (Dec 15, 2025) Use the pruning code from Overcooked paper
    - [x] (Dec 20, 2025) Use the deepening code from Overcooked paper

### Known bugs
- [ ] (Dec 26, 2025) There's some bug with pruning that exists regarding the linear models in submodels.
- [ ] (Dec 3, 2025) temp pred loss also computes for zero obs tensor when it doesn't need to. We probs want to remove such tensors
- [ ] (Nov 27, 2025) temporal ensemble is *probably* resetting past actions only at beginning of rollout collect, but it should be doing it before every new episode starts instead.
- [x] (Nov 12, 2025) with the addition of action chunking and MLP, infonce stuff doesn't work
- [x] (Nov 10, 2025) LaneKeeping is now outputting dictionary observations; 
- [X] (Nov 10, 2025) InvertedPendulum expects different action dim
    - (Nov 17, 2025) Fixed. Had to change evaluate actions from stable baselines to only pass in the current timestep action when calling ```env.step(action)```. Also had to change logic for zeroing out certain actions in ```policies.py```
- [x] `ppo.py`: `self.ent_coef` is equal to `'auto'` when printed. It should not be a str() but rather a float.
    - (Oct 17, 2025) Fixed. Issue was that `'auto'` was being passed to PPO() in `train.py` as a parameter. Change to a float. 

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