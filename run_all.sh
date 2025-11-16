# For M1 (CDDT)
for SEED in 0 1 2 3; do
  sbatch --export=ALL,SEED=$SEED temp_interp/run/train_ll_all.slurm
done

for SEED in 0 1 2 3; do
  sbatch --export=ALL,SEED=$SEED temp_interp/run/train_ip_all.slurm
done



# # For M2 (CDDT-Controllers)
# for SEED in 0 1 2; do
#   sbatch --job_name={SEED}_{n_envs} --export=ALL,METHOD=M2,SEED=$SEED temp_interp/run/train_ll_all.slurm
# done

