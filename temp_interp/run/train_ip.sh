module load conda
conda activate temp-interp
export OMP_NUM_THREADS=1
curr_user=$(whoami)
# export MKL_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=1

python -m temp_interp.run.train \
  --env_name cart \
  --abstraction_type temp-ensemble \
  --num_envs 2 \
  --seed 0 \
  --n_steps 10 \
  --batch_size 4 \
  --gamma 0.99 \
  --ent-coef 0.0 \
  --eval_freq 10 \
  --min_reward 900 \
  --training_steps 10000000 \
  --log_interval 1 \
  --save_path /scratch/gilbreth/${curr_user}/temp-interp/temp_interp/run/logs/ip/ \
  --use_individual_alpha \
  --hard_node \
  --submodels \
  --sparse_submodel_type 0 \