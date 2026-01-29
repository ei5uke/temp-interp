module load conda
conda activate temp-interp
export OMP_NUM_THREADS=1
curr_user=$(whoami)
# export MKL_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=1

python -m temp_interp.run.test_robustness \
  --seed 0 \
  --load_path /scratch/gilbreth/${curr_user}/temp-interp/temp_interp/run/logs/ll_hard/temp-NoAC/ \
  --load_file best_model_seed0 \
  --num_episodes 100 \
  --abstraction_type temp-pred \
  --n_steps 2048 \
  --batch_size 256 \
  --gamma 0.99 \
  --ent-coef 0.0 \
  --use_individual_alpha \
  --hard_node \
  --submodels \
  --sparse_submodel_type 0 \