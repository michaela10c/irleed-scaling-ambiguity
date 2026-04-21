import sys, os
ROOT = os.path.abspath(os.curdir)
sys.path.append(os.path.abspath(os.path.join(ROOT,'src')))

import mix_irl.irleed as I
import numpy as np
import pickle
import argparse
from tqdm import trange
import traceback
import random


def load_stage_rewards_from_file(result_path, stage_iters, use_mean_history=True):
    """
    Load theta snapshots from an earlier run and use them as staged demonstrator rewards.

    result_path: path to a saved pickle [options, data]
    stage_iters: list like [10, 100, 999]
    use_mean_history: if True, average theta history across valid seeds;
                      otherwise use the first valid seed only
    """
    with open(result_path, "rb") as f:
        old_options, old_data = pickle.load(f)

    valid = [d for d in old_data if d is not None and "log" in d and "theta" in d["log"] and len(d["log"]["theta"]) > 0]
    if len(valid) == 0:
        raise RuntimeError(f"No valid theta history found in {result_path}")

    if use_mean_history:
        T = max(len(d["log"]["theta"]) for d in valid)
        F = len(np.array(valid[0]["log"]["theta"][0]).reshape(-1))

        theta_sum = np.zeros((T, F), dtype=np.float64)
        theta_count = np.zeros((T, F), dtype=np.float64)

        for d in valid:
            th = np.stack([np.array(x).reshape(-1) for x in d["log"]["theta"]], axis=0)
            pad = np.full((T, F), np.nan, dtype=np.float64)
            pad[:th.shape[0]] = th
            mask = np.isfinite(pad)
            theta_sum[mask] += pad[mask]
            theta_count[mask] += 1.0

        theta_hist = theta_sum / (theta_count + 1e-12)
    else:
        th = valid[0]["log"]["theta"]
        theta_hist = np.stack([np.array(x).reshape(-1) for x in th], axis=0)

    rewards = []
    T_hist = theta_hist.shape[0]
    for t in stage_iters:
        tt = max(0, min(int(t), T_hist - 1))
        rewards.append(theta_hist[tt].copy())

    return rewards

def run_irleed(options):
    result = {}

    # choose generator betas
    if ARGS.demo_betas is not None:
        weights = np.array(ARGS.demo_betas, dtype=float)
        if len(weights) != ARGS.n_components:
            raise ValueError("--demo_betas must have length equal to --n_components")
    elif ARGS.weight_scale > 20:
        weights = [None] * ARGS.n_components
    else:
        weights = np.array([ARGS.demo_beta] * ARGS.n_components)

    result['true_betas'] = None if weights[0] is None else np.array(weights, dtype=float)

    # optional: staged demonstrator rewards from earlier run
    stage_rewards = None
    if ARGS.stage_demo_file is not None:
        if ARGS.stage_demo_iters is None:
            raise ValueError("If --stage_demo_file is used, also provide --stage_demo_iters")
        stage_rewards = load_stage_rewards_from_file(
            ARGS.stage_demo_file,
            ARGS.stage_demo_iters,
            use_mean_history=not ARGS.stage_demo_use_first_seed
        )
        if len(stage_rewards) != ARGS.n_components:
            raise ValueError("Number of stage rewards must equal n_components")

    # learn_eps is False if we fix epsilons to zero
    algo = I.irleed(learn_eps=not ARGS.fix_eps_zero)

    algo.reset_data(
        options['ratios'],
        weights,
        options['lam'],
        options['n_traj'],
        options,
        stage_rewards=stage_rewards
    )

    result['log'] = algo.run_irleed(outer_eps=1e-4,inner_eps=1e-4,max_steps=options['max_steps'])
    result['dem_rews'] = algo.setup['dem_rews']
    result['dem_lens'] = algo.setup['dem_lens']
    result['mix_e_features'] = algo.setup['mix_e_features']
    result['true_epsilons'] = algo.setup['epsilons']
    result['mix_traj'] = algo.setup['mix_traj']
    return result

def main():
    # pathing 
    beta_tag = (
        f"demo_betas_{'_'.join(f'{b:.3f}' for b in ARGS.demo_betas)}"
        if ARGS.demo_betas is not None
        else f"demo_beta_{ARGS.demo_beta:.3f}"
    )

    save_dir = os.path.join(
        ROOT,
        'results',
        ARGS.save_dir,
        f'env_{ARGS.env_id}',
        beta_tag,
        'noeps' if ARGS.fix_eps_zero else 'eps'
    )
    
    # this will create all parents if needed and not crash if they exist
    os.makedirs(save_dir, exist_ok=True)
    
    # create place to save data and options
    options = {}
    
    # configure options
    options['discount'] = 0.9
    options['horizon'] = 100
    options['n_e_traj'] = 100
    options['n_s_traj'] = 100
    options['env_id'] = ARGS.env_id
    options['lr_betas'] = ARGS.lr_betas
    options['lr_theta'] = ARGS.lr_theta
    options['lr_epsilons'] = ARGS.lr_epsilons
    options['debug'] = ARGS.debug
    options['max_steps'] = ARGS.max_steps
    # options['ratios'] = [0.2]*5
    options['ratios'] = [1.0] if ARGS.n_components == 1 else [1.0/ARGS.n_components]*ARGS.n_components
    # options['n_traj'] = 5000 # increase # of trajectories
    options['n_traj'] = 500
    options['exp_key'] = ARGS.exp_key
    options['causal'] = ARGS.causal
    options['fix_eps_zero'] = ARGS.fix_eps_zero

    if ARGS.fix_eps_zero:
        lam_list = [None] # lambda does not matter if epsilon is zero
    else:
        lam_list = [2] # set lambda = 2 if epsilon is nonzero, so epsilon_i ~ N(0, I/lambda^2 = I/4)

    # values used in experiment
    for lam in lam_list:
        data = []

        # Use this structure for file saving:
        # demo_beta_<BETA>/
        # |---noeps/
        #      |----baseline.p
        # |---eps/
        #      |----lam_<LAMBDA>.p
        if ARGS.fix_eps_zero:
            options['lam'] = 0
            save_name = "baseline.p"
        else:
            options['lam'] = lam
            save_name = f"lam_{lam:.3f}.p"
        
        save_path = os.path.join(save_dir, save_name)

        # skip run if file already exists
        if os.path.isfile(save_path):
            print(f"Skipping existing {save_path}")
            continue

        # for each seed, run IRLEED
        for seed in trange(ARGS.n_seeds):
            try:
                # set reproducibility seed
                np.random.seed(seed)
                random.seed(seed)

                # run IRLEED
                result = run_irleed(options)
            except Exception:
                print(f"skipped seed {seed}")
                traceback.print_exc()
                result = None
            data.append(result)

        # save to file!
        with open(save_path, "wb") as f:
            pickle.dump([options.copy(), data], f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--weight_scale', type=float, default=1, help="scale of demonstrator accuracies")
    parser.add_argument('--lr_betas', type=float, default=0.05, help="learning rate for beta")
    parser.add_argument('--lr_theta', type=float, default=0.2, help="learning rate for theta")
    parser.add_argument('--lr_epsilons', type=float, default=0.1, help="learning rate for epsilon")
    parser.add_argument('--save_dir', type=str, default='irleed', help="directory to save to, will be created")
    parser.add_argument('--n_seeds', type=int, default=100, help="number of seeds to run")
    parser.add_argument('--env_id', type=int, default=1, help="env id")
    parser.add_argument('--max_steps', type=int, default=2, help="number of steps to run for")
    parser.add_argument('--exp_key', type=str, default='1-4', help="key of experiment to run")
    parser.add_argument('--debug', action='store_true', help="displays results while running")
    parser.add_argument('--causal', action='store_true', help="decides if we use causal IRL")
    parser.add_argument('--fix_eps_zero', action='store_true', help="If set, fixes all per-demonstrator epsilons to zero (no epsilon noise)")
    parser.add_argument('--n_components', type=int, default=5)
    parser.add_argument('--demo_beta', type=float, default=1.0)  # used ONLY for data generation

    parser.add_argument('--demo_betas', type=float, nargs='+', default=None,
                    help="Optional list of demonstrator betas, e.g. --demo_betas 0.3 1.0 5.0") # used ONLY for data generation

    parser.add_argument('--stage_demo_file', type=str, default=None,
                        help="Path to an earlier saved result file whose theta history will be used as staged demonstrator rewards")

    parser.add_argument('--stage_demo_iters', type=int, nargs='+', default=None,
                        help="Iteration indices used to extract staged demonstrator rewards, e.g. --stage_demo_iters 10 100 999")

    parser.add_argument('--stage_demo_use_first_seed', action='store_true',
                        help="If set, use the first valid seed from stage_demo_file instead of averaging theta histories across seeds")
        
    ARGS = parser.parse_args()
    print(ARGS)
    
    main()
