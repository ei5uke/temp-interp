
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
  --lr 5e-4 \
  --ddt_lr 5e-4 \
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
  --argmax_tau 1.0 \
  --sparse_submodel_type 2 \
  --num_sub_features 2 \
  | tee temp_interp/run/logs/train_ll.log