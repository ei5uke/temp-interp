#!/bin/bash
#SBATCH --job-name=DDT-AC
#SBATCH --output=temp_interp/run/logs/train_ll_lr-3e-4.log
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --account=rpaleja
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=a100-40gb
#SBATCH --mem=100G

module load conda
conda activate temp-interp
export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=1

python -m temp_interp.run.train \
  --env_name lunar \
  --policy_type ddt \
  --seed 0 \
  --num_leaves 8 \
  --lr 3e-4 \
  --ddt_lr 3e-4 \
  --batch_size 256 \
  --gamma 0.99 \
  --learning_starts 10000 \
  --eval_freq 1500 \
  --min_reward 225 \
  --training_steps 500000 \
  --log_interval 4 \
  --save_path /scratch/gilbreth/hirota/temp-interp/temp_interp/run/logs/ll/ \
  --use_individual_alpha \
  --submodels \
  --hard_node \
  --gpu \
  --argmax_tau 1.0 \
  --sparse_submodel_type 2 \
  --num_sub_features 2 \