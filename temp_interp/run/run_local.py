"""
Run the experiments of docs/rerun_todo.md on a local machine (no slurm), a few at a time, in priority order.

Each job runs its slurm file with bash, with the slurm variables set in the environment and the
save path redirected to ``--root``. A job writes a DONE file when it succeeds, so rerunning this
script skips finished jobs and continues with the rest. CART and Warm jobs wait for their MLP-Big teacher.

Example:
    python -m temp_interp.run.run_local --root results --workers 7 --python /path/to/env/bin/python
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENVS = ['ll_hard', 'll', 'ip', 'lk']  # longest runs first
SEEDS = [0, 1, 2]
# LL-H ITTR settings for warm-started chunked trees: pruning only (deepening disabled), CART already provides the structure.
# Check every ~2 updates (n) and prune leaves receiving under ~1/3 of the visits in that window (k), so the tree is compact early
ABLATION_EMA = [0.5, 0.625, 0.875, 1.0]
WARM_COMPACT_ITTR = '--epsilon=-inf --min_timesteps 100000 --min_visitations 4e6'
METHODS = ['temp-ensemble', 'temp-pred']


def build_jobs(root: Path):
    """All jobs in priority order. A job is a dict with name, slurm file, slurm variables, extra args and dependencies."""
    jobs = []

    def add(env, kind, run_name, seed, method, time_horizon, size=None, extra='', deps=()):
        run_dir = root / env / run_name / f'seed{seed}'
        variables = {'SEED': str(seed), 'METHOD': method, 'TIME': str(time_horizon), 'RUN_NAME': run_name}
        if size: variables['SIZE'] = size
        slurm = f'temp_interp/run/train_{env}_mlp_all.slurm' if kind == 'mlp' else f'temp_interp/run/train_{env}_all.slurm'
        jobs.append(dict(name=f'{env}/{run_name}/seed{seed}', slurm=slurm, variables=variables, run_dir=run_dir,
                         extra=f'--no_sweep --save_path {run_dir}/ {extra}'.strip(), deps=list(deps)))

    def teacher(env, method, seed):
        run_name = 'mlp_max_noac' if method == 'noac' else f'mlp_max_{method}'
        return f'{env}/{run_name}/seed{seed}', root / env / run_name / f'seed{seed}' / f'best_model_seed{seed}.zip'

    def tier(make):
        for env in ENVS:
            for seed in SEEDS:
                make(env, seed)

    # R1, R2: DDT-Ensemble and DDT-Prediction
    tier(lambda env, seed: [add(env, 'ddt', f'ddt_{m}', seed, m, 10) for m in METHODS])
    # B1: DDT baseline
    tier(lambda env, seed: add(env, 'ddt', 'ddt_noac', seed, 'temp-pred', 1))
    # R3, R4: MLP-Ensemble and MLP-Prediction
    tier(lambda env, seed: [add(env, 'mlp', f'mlp_mid_{m}', seed, m, 10, size='mid') for m in METHODS])
    # B2, R5, R6: MLP-Big teachers
    tier(lambda env, seed: add(env, 'mlp', 'mlp_max_noac', seed, 'temp-pred', 1, size='max'))
    tier(lambda env, seed: [add(env, 'mlp', f'mlp_max_{m}', seed, m, 10, size='max') for m in METHODS])
    # B3, R7: CART, then B4, R8: Warm
    for prefix, warm in [('cart', ''), ('warm', ' --warm_start')]:
        def make(env, seed):
            for m, method, horizon in [('noac', 'temp-pred', 1)] + [(m, m, 10) for m in METHODS]:
                dep, path = teacher(env, m, seed)
                add(env, 'ddt', f'{prefix}_{m}', seed, method, horizon, extra=f'--cart_teacher {path}{warm}', deps=[dep])
        tier(make)
    # Warm-Ensemble and Warm-Prediction on LL-H with ITTR settings that keep the warm-started tree compact
    for seed in SEEDS:
        for m in METHODS:
            dep, path = teacher('ll_hard', m, seed)
            add('ll_hard', 'ddt', f'warmcompact_{m}', seed, m, 10, deps=[dep],
                extra=f'--cart_teacher {path} --warm_start {WARM_COMPACT_ITTR}')
    # R9: EMA coefficient ablation of DDT-Ensemble (eta = 0.75 is the main ddt_temp-ensemble run)
    tier(lambda env, seed: [add(env, 'ddt', f'ddt_temp-ensemble_ema{eta}', seed, 'temp-ensemble', 10, extra=f'--decay {eta}')
                            for eta in ABLATION_EMA])
    return jobs


def launch(job, python_bin: Path):
    job['run_dir'].mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **job['variables'], 'EXTRA_ARGS': job['extra'], 'WANDB_MODE': 'offline',
           'WANDB_DIR': str(job['run_dir']), 'PYTHONPATH': str(REPO), 'PATH': f'{python_bin.parent}:{os.environ["PATH"]}'}
    log = open(job['run_dir'] / 'stdout.log', 'w')
    return subprocess.Popen(['bash', job['slurm']], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser(description='Run the rerun experiments locally')
    parser.add_argument('--root', type=Path, default=REPO / 'results', help='output directory')
    parser.add_argument('--workers', type=int, default=6, help='number of jobs to run at once')
    parser.add_argument('--python', type=Path, required=True, help='python executable of the environment to use')
    args = parser.parse_args()

    jobs = build_jobs(args.root.resolve())
    done = {j['name'] for j in jobs if (j['run_dir'] / 'DONE').exists()}
    failed, running = set(), {}
    pending = [j for j in jobs if j['name'] not in done]
    print(f'{len(jobs)} jobs, {len(done)} already done', flush=True)

    while pending or running:
        for name, (job, proc, start) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[name]
            minutes = (time.time() - start) / 60
            if proc.returncode == 0:
                (job['run_dir'] / 'DONE').touch()
                done.add(name)
                print(f'{time.strftime("%H:%M")} done   {name} ({minutes:.0f} min)', flush=True)
            else:
                failed.add(name)
                print(f'{time.strftime("%H:%M")} FAILED {name} (exit {proc.returncode}), see {job["run_dir"]}/stdout.log', flush=True)

        for job in list(pending):
            if any(dep in failed for dep in job['deps']):
                pending.remove(job)
                failed.add(job['name'])
                print(f'skipped {job["name"]}: dependency failed', flush=True)
            elif len(running) < args.workers and all(dep in done for dep in job['deps']):
                pending.remove(job)
                running[job['name']] = (job, launch(job, args.python), time.time())
                print(f'{time.strftime("%H:%M")} start  {job["name"]}', flush=True)
        time.sleep(10)

    print(f'finished: {len(done)} done, {len(failed)} failed', flush=True)


if __name__ == '__main__':
    main()
