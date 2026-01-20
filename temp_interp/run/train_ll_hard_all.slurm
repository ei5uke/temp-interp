#!/bin/bash
#SBATCH --job-name=DDT-AC
#SBATCH --output=temp_interp/run/logs/ll_hard/icct.log
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=rpaleja
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=a100-40gb
#SBATCH --mem=32G

module load conda
conda activate temp-interp
export OMP_NUM_THREADS=1

echo "Running ICCT-complete with Action Chunking with seed ${SEED} and method ${METHOD}"
curr_user=$(whoami)

python -m temp_interp.run.train \
  --env_name lunar-hard \
  --abstraction_type ${METHOD} \
  --num_envs 32 \
  --seed ${SEED} \
  --n_steps 2048 \
  --batch_size 256 \
  --gamma 0.99 \
  --ent-coef 0.0 \
  --eval_freq 1500 \
  --min_reward 225 \
  --training_steps 3000000 \
  --log_interval 1 \
  --gpu \
  --save_path /scratch/gilbreth/${curr_user}/temp-interp/temp_interp/run/logs/ll_hard/ICCTComplete_${SEED} \
  --use_individual_alpha \
  --hard_node \
  --submodels \
  --sparse_submodel_type 0 \
  --time_horizon 10 \
