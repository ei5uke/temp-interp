
module load conda
conda activate temp-interp
export OMP_NUM_THREADS=1
curr_user=$(whoami)
# export MKL_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=1

python -m temp_interp.run.train \
  --env_name cart \
  --num_envs 2 \
  --seed 0 \
  --num_leaves 2 \
  --n_steps 2048 \
  --lr 3e-4 \
  --ddt_lr 3e-4 \
  --batch_size 256 \
  --gamma 0.99 \
  --ent-coef 0.0 \
  --clip-range 0.2 \
  --eval_freq 1500 \
  --min_reward 900 \
  --training_steps 2000000 \
  --log_interval 1 \
  --save_path /scratch/gilbreth/${curr_user}/temp-interp/temp_interp/run/logs/ip/ \
  --use_individual_alpha \
  --argmax_tau 1.0 \
  --hard_node \
  --submodels \
  --sparse_submodel_type 0 \
  # --num_sub_features 2 \
  # --clip-range-vf 0.2 \
  | tee temp_interp/run/logs/ip/complete_lr-3e-4_seed-0.log