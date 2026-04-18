# Instructions

This repository is now organized as a normal top-level checkout. The old nested `SimpleVLA-RL/` wrapper has been removed, so all commands should be run from the repo root.

## Repository layout

- `verl/` contains the training stack
- `examples/` contains runnable scripts
- `tests/` contains targeted tests
- `figs/`, `modified_codes/`, and top-level setup files remain at the root

## Common commands

### Run LunarLander shaped-reward training

```bash
bash examples/run_lunarlander_rl.sh
```

### Run a specific LunarLander ablation

```bash
ABLATION=terminal_only bash examples/run_lunarlander_rl.sh
ABLATION=terminal_smooth bash examples/run_lunarlander_rl.sh
ABLATION=terminal_sub_prog bash examples/run_lunarlander_rl.sh
ABLATION=full bash examples/run_lunarlander_rl.sh
ABLATION=dense_only bash examples/run_lunarlander_rl.sh
```

### Run focused tests

```bash
pytest -q tests/test_lunarlander_reward.py tests/test_lunarlander_reward_manager.py
```

### Syntax check changed Python files

```bash
python3 -m py_compile \
  verl/utils/lunarlander_shaped_reward.py \
  verl/workers/lunarlander_workers.py \
  verl/trainer/main_ppo.py \
  verl/trainer/ppo/ray_trainer.py \
  verl/utils/dataset/rob_dataset.py
```

## Practical notes

- The LunarLander path currently expects `torch` and `gymnasium[box2d]`.
- The LunarLander worker is intentionally single-process for the first pass.
- The original OpenVLA/LIBERO/Robotwin paths are still present.
- Reward shaping for LunarLander is configured in `verl/trainer/config/ppo_trainer.yaml`.
- Trajectory dumps and experiment outputs are written under the configured `trainer.default_local_dir`.

## Suggested workflow

1. Install dependencies from `SETUP.md` plus LunarLander extras.
2. Run the focused tests.
3. Run `examples/run_lunarlander_rl.sh` with one ablation at a time.
4. Fill in `examples/lunarlander_report_template.md` with metrics and plots.
