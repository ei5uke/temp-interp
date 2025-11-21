#!/bin/bash
#SBATCH --job-name=DDT-AC
#SBATCH --output=temp_interp/run/logs/ll/complete_lr-5e-4_seeds.log
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --account=rpaleja
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=a100-40gb
#SBATCH --mem=32G

module load conda
conda activate temp-interp
export OMP_NUM_THREADS=1

echo "Running ICCT-complete with Action Chunking with seed ${SEED}"
curr_user=$(whoami)

python -m temp_interp.run.train \
  --env_name lunar \
  --num_envs 32 \
  --seed ${SEED} \
  --num_leaves 8 \
  --n_steps 2048 \
  --lr 5e-4 \
  --ddt_lr 5e-4 \
  --batch_size 256 \
  --gamma 0.99 \
  --ent-coef 0.0 \
  --clip-range 0.2 \
  --eval_freq 1500 \
  --min_reward 225 \
  --training_steps 10000000 \
  --log_interval 1 \
  --save_path /scratch/gilbreth/${curr_user}/temp-interp/temp_interp/run/logs/ll/ICCTComplete_${SEED} \
  --use_individual_alpha \
  --gpu \
  --hard_node \
  --submodels \
  --sparse_submodel_type 0 \
  --argmax_tau 1.0



# Modify based on method
# if [ "${METHOD}" = "M1" ]; then
#   # CDDT (M1) – no submodels, no hard_node
#   echo "Running CDDT (M1) ..."
# elif [ "${METHOD}" = "M2" ]; then
#   # CDDT-Controllers (M2) – submodels, no hard_node
#   CMD="${CMD} --submodels"
#   echo "Running CDDT-Controllers (M2) ..."
# else
#   echo "Unknown METHOD: ${METHOD}"
#   exit 1
# fi

# Execute
# eval $CMD

