#!/bin/bash
#SBATCH --job-name=temp-pred
#SBATCH --output=temp_interp/run/logs/ll/pred_seed-0.log
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=rpaleja
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=v100
#SBATCH --mem=32G

module load conda
conda activate temp-interp
export OMP_NUM_THREADS=1
curr_user=$(whoami)
# export MKL_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=1

python -m temp_interp.run.train_pred \
  --env_name lunar-hard \
  --num_envs 32\
  --seed 0 \
  --n_steps 2048 \
  --batch_size 256 \
  --gamma 0.99 \
  --ent-coef 0.0 \
  --eval_freq 1500 \
  --min_reward 225 \
  --training_steps 10000000 \
  --log_interval 1 \
  --gpu \
  --save_path /scratch/gilbreth/${curr_user}/temp-interp/temp_interp/run/logs/ll/ \
  --use_individual_alpha \
  --hard_node \
  --submodels \
  --sparse_submodel_type 0 \