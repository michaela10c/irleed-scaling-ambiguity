# IRLEED Scaling Ambiguity Experiments

## Overview

This project studies scaling ambiguity in [IRLEED (Inverse Reinforcement Learning by Estimating Expertise of Demonstrators)](https://arxiv.org/html/2402.01886v2).

We investigate the following question: ***Why is reward recovery successful while expertise (beta) estimation remains unidentifiable?***

We evaluate three experimental settings:
1. Homogeneous setting (single demonstrator, epsilon OFF)
2. Heterogeneous setting (multiple demonstrators, epsilon OFF)
3. Epsilon-only setting (shared beta, epsilon ON)

---

## Codebase Structure

The implementation builds on the original IRLEED codebase, with a root-level script for running experiments.

```
irleed-scaling-ambiguity/
├── run_mix.py          (main experiment entry point)
├── src/
│   ├── irl_maxent/     (Maximum Entropy IRL baseline)
│   │   ├── gridworld.py
│   │   ├── maxent.py
│   │   ├── optimizer.py
│   │   ├── solver.py
│   │   ├── trajectory.py
│   │   └── plot.py
│   │
│   └── mix_irl/        (IRLEED + extensions)
│       ├── irl.py
│       ├── irleed.py
│       ├── helpers.py
│       └── trajectory.py
```

### Module roles:

- `run_mix.py`  
  Main script used to run all experiments.  
  Handles:
  - multi-seed execution
  - experiment configuration
  - saving results to `results/`

- `irl_maxent/`  
  Standard Maximum Entropy IRL implementation used as a baseline.

- `mix_irl/`  
  IRLEED implementation and extensions used in this project:
  - mixture modeling across components  
  - beta (expertise) estimation  
  - epsilon-based reward perturbations  
  - multi-seed experiment pipeline

---

## Environment

- Gridworld (7 x 7)
- 3 terminal reward corners
- Discount factor gamma = 0.9
- Maximum Entropy IRL framework

---

## Results Directory Structure

All outputs are stored under:

`results/gridworld_simple/`

### 1. Standard IRLEED Experiments (epsilon OFF)

`results/gridworld_simple/irleed/env_1/`

**Homogeneous**:
- `demo_beta_0.100/noeps/baseline.p`
- `demo_beta_1.000/noeps/baseline.p`
- `demo_beta_5.000/noeps/baseline.p`
- `demo_beta_10.000/noeps/baseline.p`

**Heterogeneous**:
- `demo_betas_0.300_1.000_5.000/noeps/baseline.p`

---

### 2. Epsilon-Only Experiments (epsilon ON)

`results/gridworld_simple/irleed_eps_only/env_1/demo_betas_1.000_1.000_1.000/eps/lam_2.000.p`

---

## Precomputed Results 

Due to the large size of the experiment outputs (over 3 GB), results are not stored directly in this repository.

You may download the precomputed results [here](https://drive.google.com/file/d/1iGX_sMs-aOa8uc9sa0RnXUhmNoIPWidK/view?usp=sharing).

### Setup

1. Download the zip file
2. Extract it
3. Place the extracted `results/` folder in the root of the repository

The directory structure should look like:

```
project_root/
├── run_mix.py
├── src/
├── results/
│   └── gridworld_simple/
│       ├── irleed/
│       └── irleed_eps_only/
```

---

### Note

- These results correspond to the experiments described in the report.
- Homogeneous and heterogeneous experiments use 100 seeds, 1000 iterations, and 5000 trajectories per seed.
- The epsilon-only follow-up experiment uses 100 seeds, 1000 iterations, and 500 trajectories per seed due to higher computational cost.
- Re-running all experiments from scratch may take several hours to multiple days depending on hardware.
- The provided results are for convenience; all results can be reproduced using the commands below


## Experiments

---

### 1. Homogeneous Setting (epsilon OFF)

Setup:
- n_components = 1
- beta ∈ {0.1, 1.0, 5.0, 10.0}
- epsilon disabled (--fix_eps_zero)
- 100 seeds, 1000 steps
- 5000 trajectories per seed

#### Run commands:
```
python run_mix.py --save_dir gridworld_simple/irleed --n_components 1 --demo_beta 0.1 --fix_eps_zero --max_steps 1000 
python run_mix.py --save_dir gridworld_simple/irleed --n_components 1 --demo_beta 1 --fix_eps_zero --max_steps 1000 
python run_mix.py --save_dir gridworld_simple/irleed --n_components 1 --demo_beta 5 --fix_eps_zero --max_steps 1000 
python run_mix.py --save_dir gridworld_simple/irleed --n_components 1 --demo_beta 10 --fix_eps_zero --max_steps 1000 
```

---

### 2. Heterogeneous Setting (epsilon OFF)

Setup:
- n_components = 3
- beta = 0.3, 1.0, 5.0
- epsilon disabled (--fix_eps_zero)
- 100 seeds, 1000 steps
- 5000 trajectories per seed

#### Run command:

```
python run_mix.py \
   --save_dir gridworld_simple/irleed \
   --n_components 3 \
   --demo_betas 0.3 1.0 5.0 \
   --fix_eps_zero \
   --max_steps 1000
```
---

### 3. Epsilon-Only Setting (epsilon ON)

Setup:
- n_components = 3
- beta = 1.0, 1.0, 1.0 (shared)
- epsilon enabled
- lambda = 2.0
- 100 seeds, 1000 steps
- 500 trajectories per seed

Due to increased computational cost from learning component-specific epsilon parameters, fewer trajectories were used in this follow-up experiment.

#### Run command:

```
python run_mix.py \
  --save_dir gridworld_simple/irleed_eps_only \
  --n_components 3 \
  --demo_betas 1.0 1.0 1.0 \
  --max_steps 1000 
```

---

Note:
- Homogeneous and heterogeneous settings explicitly disable epsilon via `--fix_eps_zero`
- Epsilon-only setting enables epsilon and is the only configuration where epsilon is learned

---

## Output Files

Each `.p` file contains:
- learned theta (reward)
- learned beta (expertise)
- epsilon (if enabled)
- training histories
- aggregated multi-seed results

---

## Analysis Notebooks 

We provide Jupyter notebooks for analyzing the experiment outputs:

- `analyze_homogeneous_demonstrators.ipynb`
- `analyze_heterogeneous_demonstrators.ipynb`

### Purpose

- **Homogeneous notebook**  
  Used to analyze single-demonstrator experiments (beta recovery, scaling behavior).

- **Heterogeneous notebook**  
  Used to analyze:
  - heterogeneous demonstrator experiments (epsilon OFF)
  - epsilon-only experiments (epsilon ON)

  This notebook visualizes:
  - inferred policies
  - visitation distributions
  - epsilon structure (when enabled)
  - component-wise comparisons

---

### Usage

1. Ensure the `results/` directory is present (either by running experiments or downloading precomputed results)
2. Launch Jupyter: `jupyter notebook`
3. Open the desired notebook and run all cells

---

### Note

- These notebooks are **for visualization and analysis only**
- They are **not required** to reproduce the results
- All figures in the report and poster were generated using these analysis workflows

---

## Important Note on Seeds

- 100 seeds launched
- 2 seeds failed due to numerical instability
- final results use 98 valid seeds

Cause:
- overflow in weighting
- invalid normalization
- NaN probabilities during trajectory sampling

---

## Results Summary

All results are aggregated over multiple random seeds. Homogeneous and heterogeneous experiments use 5000 trajectories per seed, while the epsilon-only follow-up uses 500 trajectories per seed.

### Beta

- converges to approximately 0.9 (epsilon OFF) or ~1.35 (epsilon ON)
- nearly identical across components

Conclusion:
beta is not identifiable due to scaling ambiguity

---

### Theta

- exhibits U-shaped error curve

Interpretation:
- early learning followed by scaling drift (not overfitting)

---

### Epsilon

- learned successfully
- structured spatial patterns
- similar across components

Conclusion:
epsilon does not induce behavioral separation

---

### Policy

- nearly identical across components

---

### Visitation

- minor variation only
- no distinct behavioral modes

---

## Final Conclusion

- reward recovery is robust under the evaluated settings
- beta (expertise) is not identifiable
- components collapse to shared behavior

Scaling ambiguity persists:
- in homogeneous setting
- in heterogeneous setting
- in epsilon-only setting

---

## Files

- `run_mix.py` — main experiment script  
- `src/irl_maxent/` — MaxEnt IRL baseline  
- `src/mix_irl/` — IRLEED + extensions  
- `results/` — experiment outputs  
- `analyze_homogeneous_demonstrators.ipynb` — analysis for homogeneous setting  
- `analyze_heterogeneous_demonstrators.ipynb` — analysis for heterogeneous and epsilon-only settings  
- Additional files from the original IRLEED repository and intermediate/debug experiment outputs are retained for completeness. Only the result files described above are used in the analyses and conclusions presented in this project.

---

## Notes

- experiments follow the setup described in the report
- homogeneous and heterogeneous experiments use 5000 trajectories per seed
- epsilon-only follow-up uses 500 trajectories per seed
- homogeneous and heterogeneous experiments disable epsilon  
- epsilon-only experiment isolates epsilon as the only source of heterogeneity  
- some seeds may fail due to numerical instability
