for SEED in 0 1 2; do
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-ensemble temp_interp/run/train_ll_all.slurm
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-pred temp_interp/run/train_ll_all.slurm
done

for SEED in 0 1 2; do
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-ensemble temp_interp/run/train_ip_all.slurm
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-pred temp_interp/run/train_ip_all.slurm
done

for SEED in 0 1 2; do
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-ensemble temp_interp/run/train_lk_all.slurm
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-pred temp_interp/run/train_lk_all.slurm
done

for SEED in 0 1 2; do
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-ensemble temp_interp/run/train_ll_hard_all.slurm
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-pred temp_interp/run/train_ll_hard_all.slurm
done


# # For M2 (CDDT-Controllers)
# for SEED in 0 1 2; do
#   sbatch --job_name={SEED}_{n_envs} --export=ALL,METHOD=M2,SEED=$SEED temp_interp/run/train_ll_all.slurm
# done

