# IRLEED Scaling Ambiguity Experiments

## Overview

This project studies scaling ambiguity in IRLEED (Inverse Reinforcement Learning by Estimating Expertise of Demonstrators).

We investigate the following question:

Why does reward recovery succeed while expertise (beta) estimation fails?

We evaluate three experimental settings:
1. Homogeneous setting (single demonstrator, epsilon OFF)
2. Heterogeneous setting (multiple demonstrators, epsilon OFF)
3. Epsilon-only setting (shared beta, epsilon ON)

---

## Codebase Structure

The implementation builds on the original IRLEED codebase.

```
src/
├── irl_maxent/        (Maximum Entropy IRL baseline)
│   ├── gridworld.py
│   ├── maxent.py
│   ├── optimizer.py
│   ├── solver.py
│   ├── trajectory.py
│   └── plot.py
│
└── mix_irl/           (IRLEED + extensions)
    ├── irl.py
    ├── irleed.py
    ├── helpers.py
    └── trajectory.py
```

Module roles:

- `irl_maxent/`  
  Standard MaxEnt IRL implementation used as a baseline.

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

Homogeneous:
- `demo_beta_0.100/noeps/baseline.p`
- `demo_beta_1.000/noeps/baseline.p`
- `demo_beta_5.000/noeps/baseline.p`
- `demo_beta_10.000/noeps/baseline.p`

Heterogeneous:
- `demo_betas_0.300_1.000_5.000/noeps/baseline.p`

---

### 2. Epsilon-Only Experiments (epsilon ON)

`results/gridworld_simple/irleed_eps_only/env_1/demo_betas_1.000_1.000_1.000/eps/lam_2.000.p`

---

## Experiments

---

### 1. Homogeneous Setting (epsilon OFF)

Setup:
- n_components = 1
- beta ∈ {0.1, 1.0, 5.0, 10.0}
- epsilon disabled (--fix_eps_zero)
- 100 seeds, 1000 steps

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

#### Run command:

```
python run_mix.py \
  --save_dir gridworld_simple/irleed_eps_only \
  --n_components 3 \
  --demo_betas 1.0 1.0 1.0 \
  --max_steps 1000 
```

---

## Output Files

Each `.p` file contains:
- learned theta (reward)
- learned beta (expertise)
- epsilon (if enabled)
- training histories
- aggregated multi-seed results

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

- reward recovery is strong
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
- `results/` — outputs  

---

## Notes

- experiments follow the setup described in the report (100 seeds, 1000 iterations)    
- homogeneous and heterogeneous experiments disable epsilon  
- epsilon-only experiment isolates epsilon as the only source of heterogeneity  
- no notebook is required to reproduce results  
- some seeds may fail due to numerical instability
