time_horizon=10

# Inverted Pendulum
for SEED in 0 1 2; do
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-ensemble,TIME=$time_horizon temp_interp/run/train_ip_all.slurm
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-pred,TIME=$time_horizon temp_interp/run/train_ip_all.slurm
done

for SEED in 0 1 2; do
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-ensemble,TIME=$time_horizon temp_interp/run/train_ip_mlp_all.slurm
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-pred,TIME=$time_horizon temp_interp/run/train_ip_mlp_all.slurm
done

# Lunar Lander
for SEED in 0 1 2; do
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-ensemble,TIME=$time_horizon temp_interp/run/train_ll_all.slurm
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-pred,TIME=$time_horizon temp_interp/run/train_ll_all.slurm
done

for SEED in 0 1 2; do
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-ensemble,TIME=$time_horizon temp_interp/run/train_ll_mlp_all.slurm # Need to add this file
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-pred,TIME=$time_horizon temp_interp/run/train_ll_mlp_all.slurm
done


# Lane Keeping 
for SEED in 0 1 2; do
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-ensemble,TIME=$time_horizon temp_interp/run/train_lk_all.slurm
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-pred,TIME=$time_horizon temp_interp/run/train_lk_all.slurm
done

for SEED in 0 1 2; do
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-ensemble temp_interp/run/train_ll_hard_all.slurm
  sbatch --export=ALL,SEED=$SEED,METHOD=temp-pred temp_interp/run/train_ll_hard_all.slurm
done