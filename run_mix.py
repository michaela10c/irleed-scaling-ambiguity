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

def run_irleed(options):
    result = {}
    if ARGS.weight_scale > 20:
        # weights = [None]*5
        weights = [None] * ARGS.n_components
    else:
        # weights = np.random.rand(5)*ARGS.weight_scale
        weights = np.array([ARGS.demo_beta] * ARGS.n_components)

    # save the generator "true betas" 
    result['true_betas'] = None if weights[0] is None else np.array(weights, dtype=float)
    
    # learn_eps is False if we fix epsilons to zero
    algo = I.irleed(learn_eps=not ARGS.fix_eps_zero)
    
    algo.reset_data(options['ratios'], weights, options['lam'], options['n_traj'], options)
    result['log'] = algo.run_irleed(outer_eps=1e-4,inner_eps=1e-4,max_steps=options['max_steps'])
    result['dem_rews'] = algo.setup['dem_rews']
    result['dem_lens'] = algo.setup['dem_lens']
    result['mix_e_features'] = algo.setup['mix_e_features']
    result['true_epsilons'] = algo.setup['epsilons']
    result['mix_traj'] = algo.setup['mix_traj']
    return result

def main():
    # pathing 
    save_dir = os.path.join(
        ROOT,
        'results',
        ARGS.save_dir,          # e.g. 'gridworld' or 'gridworld_noeps'
        f'env_{ARGS.env_id}',   # e.g. 'env_1'
        f'demo_beta_{ARGS.demo_beta:.3f}',
        'noeps' if ARGS.fix_eps_zero else 'eps'
        # f'{ARGS.weight_scale:.3f}'  # e.g. '0.400'
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
    options['n_traj'] = 5000 # increase # of trajectories
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
    ARGS = parser.parse_args()
    print(ARGS)
    
    main()
