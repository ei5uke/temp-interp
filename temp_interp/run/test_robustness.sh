module load conda
conda activate temp-interp
export OMP_NUM_THREADS=1
curr_user=$(whoami)
# export MKL_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=1

python -m temp_interp.run.test_robustness \
  --seed 13 \
  --load_path /scratch/gilbreth/${curr_user}/temp-interp/temp_interp/run/logs/ll_hard \
  --load_file best_model \
  --num_episodes 1 \
  --abstraction_type temp-pred \
  --n_steps 2048 \
  --batch_size 256 \
  --gamma 0.99 \
  --ent-coef 0.0 \
  --use_individual_alpha \
  --hard_node \
  --submodels \
  --sparse_submodel_type 0 \
  | tee test_robustness.log