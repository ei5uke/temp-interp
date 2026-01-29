#!/bin/bash
#SBATCH --job-name=MLP-TE
#SBATCH --output=temp_interp/run/logs/ll/mlp_seed-0.log
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
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

echo "Running MLP with Action Chunking with seed ${SEED} and method ${METHOD}"
curr_user=$(whoami)

python -m temp_interp.run.train_mlp \
  --env_name lunar-hard \
  --abstraction_type ${METHOD} \
  --num_envs 32\
  --seed ${SEED} \
  --n_steps 2048 \
  --lr 5e-4 \
  --batch_size 256 \
  --gamma 0.99 \
  --ent-coef 0.0 \
  --clip-range 0.2 \
  --eval_freq 1500 \
  --min_reward 225 \
  --training_steps 3000000 \
  --log_interval 1 \
  --save_path /scratch/gilbreth/${curr_user}/temp-interp/temp_interp/run/logs/ll_hard/ \
  --time_horizon ${TIME} \
  --mlp_policy_size mid \
  --gpu \