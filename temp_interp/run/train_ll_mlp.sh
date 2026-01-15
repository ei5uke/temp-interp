module load conda
conda activate temp-interp
export OMP_NUM_THREADS=1
curr_user=$(whoami)
# export MKL_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=1

python -m temp_interp.run.train_mlp \
  --env_name lunar \
  --abstraction_type temp-pred \
  --num_envs 32 \
  --seed 0 \
  --n_steps 2048 \
  --lr 5e-4 \
  --batch_size 256 \
  --gamma 0.99 \
  --ent-coef 0.0 \
  --clip-range 0.2 \
  --eval_freq 500 \
  --min_reward 225 \
  --training_steps 500000 \
  --log_interval 4 \
  --save_path /scratch/gilbreth/${curr_user}/temp-interp/temp_interp/run/logs/ll/ \
  --gpu \
  --time_horizon 1 \
  | tee temp_interp/run/logs/ll/mlp_seed-0.log