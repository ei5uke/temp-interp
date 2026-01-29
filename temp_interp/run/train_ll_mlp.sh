module load conda
conda activate temp-interp
export OMP_NUM_THREADS=1
curr_user=$(whoami)
# export MKL_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=1

python -m temp_interp.run.train_mlp \
  --env_name lunar \
  --abstraction_type temp-pred \
  --num_envs 2 \
  --seed 0 \
  --n_steps 10 \
  --lr 5e-4 \
  --batch_size 4 \
  --gamma 0.99 \
  --ent-coef 0.0 \
  --clip-range 0.2 \
  --eval_freq 10 \
  --min_reward 225 \
  --training_steps 3000000 \
  --log_interval 4 \
  --save_path /scratch/gilbreth/${curr_user}/temp-interp/temp_interp/run/logs/ll/ \
  --gpu \
  --mlp_policy_size small \
  --time_horizon 10 \
  | tee temp_interp/run/logs/ll/mlp_seed-0.log