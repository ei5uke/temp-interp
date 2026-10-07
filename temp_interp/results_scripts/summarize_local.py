"""
Summarize the runs of ``temp_interp/run/run_local.py`` into Table 1 entries.

Return: per seed, the best ``eval/mean_reward`` during training (as in get_main_results.py with no smoothing),
or the CART evaluation return; then mean +/- s.e. over seeds.
Parameters: of the best checkpoint; all tree parameters for DDTs, policy network and action head for MLPs,
and the CART count of ``cart_num_params`` for CART.

Example:
    python -m temp_interp.results_scripts.summarize_local --root results
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch as th
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ENVS = {'lk': 'LK', 'ip': 'IP', 'll': 'LL', 'll_hard': 'LL-H'}
ROWS = {
    'mlp_max_noac': 'MLP-Big', 'mlp_max_temp-ensemble': 'MLP-Big-Ensemble', 'mlp_max_temp-pred': 'MLP-Big-Prediction',
    'mlp_mid_temp-ensemble': 'MLP-Ensemble', 'mlp_mid_temp-pred': 'MLP-Prediction',
    'ddt_noac': 'DDT', 'ddt_temp-ensemble': 'DDT-Ensemble', 'ddt_temp-pred': 'DDT-Prediction',
    'cart_noac': 'CART', 'cart_temp-ensemble': 'CART-Ensemble', 'cart_temp-pred': 'CART-Prediction',
    'warm_noac': 'Warm', 'warm_temp-ensemble': 'Warm-Ensemble', 'warm_temp-pred': 'Warm-Prediction',
    'warmcompact_temp-ensemble': 'Warm-Ensemble (compact ITTR)', 'warmcompact_temp-pred': 'Warm-Prediction (compact ITTR)',
}


def best_eval_return(run_dir: Path):
    values = []
    for events in run_dir.rglob('events.out.tfevents.*'):
        if 'wandb' in events.parts:
            continue
        acc = EventAccumulator(str(events), size_guidance={'scalars': 0})
        acc.Reload()
        if 'eval/mean_reward' in acc.Tags()['scalars']:
            values += [e.value for e in acc.Scalars('eval/mean_reward')]
    return max(values) if values else None


def num_params(model_zip: Path, is_tree: bool):
    with zipfile.ZipFile(model_zip) as archive:
        state = th.load(io.BytesIO(archive.read('policy.pth')), map_location='cpu')
    prefixes = ('action_net.',) if is_tree else ('action_net.', 'mlp_extractor.policy_net.')
    return sum(v.numel() for k, v in state.items() if k.startswith(prefixes))


def summarize_seed(run_dir: Path, run_name: str):
    seed = run_dir.name.removeprefix('seed')
    cart_json = run_dir / f'cart_results_seed{seed}.json'
    if run_name.startswith('cart_'):
        if not cart_json.exists():
            return None
        result = json.loads(cart_json.read_text())
        return result['cart/mean_reward'], result['cart/num_params']
    best_model = run_dir / f'best_model_seed{seed}.zip'
    ret = best_eval_return(run_dir)
    if ret is None or not best_model.exists():
        return None
    return ret, num_params(best_model, is_tree=not run_name.startswith('mlp_'))


def mean_se(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), values.std() / np.sqrt(len(values))


def main():
    parser = argparse.ArgumentParser(description='Summarize local runs into Table 1 entries')
    parser.add_argument('--root', type=Path, default=Path('results'))
    args = parser.parse_args()

    header = '| Algorithm | ' + ' | '.join(ENVS.values()) + ' |'
    print(header + '\n|' + '---|' * (len(ENVS) + 1))
    for run_name, row in ROWS.items():
        cells = []
        for env in ENVS:
            seeds = sorted((args.root / env / run_name).glob('seed*'))
            done = [s for s in seeds if (s / 'DONE').exists()]
            results = [r for r in (summarize_seed(s, run_name) for s in done) if r is not None]
            if not results:
                cells.append(f'running ({len(seeds)} started)' if seeds else '–')
                continue
            ret, ret_se = mean_se([r[0] for r in results])
            par, par_se = mean_se([r[1] for r in results])
            cells.append(f'{ret:.1f} ± {ret_se:.1f} / {par:.1f} ± {par_se:.1f} (n={len(results)})')
        print(f'| {row} | ' + ' | '.join(cells) + ' |')
    print('\nCells: return mean ± s.e. / parameters mean ± s.e. (n = finished seeds)')


if __name__ == '__main__':
    main()
