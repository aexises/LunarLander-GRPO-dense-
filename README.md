# SimpleVLA-RL

SimpleVLA-RL is an RL framework for vision-language-action training built on top of `veRL`. This checkout is now a regular single-repo layout at the repository root, with the original VLA codepaths plus a lightweight LunarLander sanity-check path for step-distributed shaped rewards.

## What is in this repo

- `verl/`: trainer, rollout, worker, dataset, and utility code
- `examples/`: runnable scripts, including the new LunarLander script
- `tests/`: focused tests for the LunarLander reward module and reward distribution logic
- `SETUP.md`: environment setup notes from the upstream project
- `instructions.md`: practical repo usage and workflow notes for this checkout

## Current additions in this checkout

- Step-distributed LunarLander reward shaping in [verl/utils/lunarlander_shaped_reward.py](/Users/daeron/LunarLander-GRPO-dense-/verl/utils/lunarlander_shaped_reward.py)
- A lightweight LunarLander actor-rollout worker in [verl/workers/lunarlander_workers.py](/Users/daeron/LunarLander-GRPO-dense-/verl/workers/lunarlander_workers.py)
- Reward-manager support for per-step reward placement in [verl/trainer/main_ppo.py](/Users/daeron/LunarLander-GRPO-dense-/verl/trainer/main_ppo.py)
- Trainer logging and dataset plumbing for LunarLander in [verl/trainer/ppo/ray_trainer.py](/Users/daeron/LunarLander-GRPO-dense-/verl/trainer/ppo/ray_trainer.py) and [verl/utils/dataset/rob_dataset.py](/Users/daeron/LunarLander-GRPO-dense-/verl/utils/dataset/rob_dataset.py)
- A runnable ablation script at [examples/run_lunarlander_rl.sh](/Users/daeron/LunarLander-GRPO-dense-/examples/run_lunarlander_rl.sh)

## Quick start

### 1. Set up dependencies

Follow [SETUP.md](/Users/daeron/LunarLander-GRPO-dense-/SETUP.md) first.

For the LunarLander sanity check, you also need:

```bash
pip install torch gymnasium[box2d] pytest
```

### 2. Run the original VLA training scripts

Examples:

```bash
bash examples/run_openvla_oft_rl_libero.sh
bash examples/run_openvla_oft_rl_twin2.sh
```

### 3. Run the LunarLander sanity check

Default full shaped reward:

```bash
bash examples/run_lunarlander_rl.sh
```

Choose an ablation:

```bash
ABLATION=terminal_only bash examples/run_lunarlander_rl.sh
ABLATION=terminal_smooth bash examples/run_lunarlander_rl.sh
ABLATION=terminal_sub_prog bash examples/run_lunarlander_rl.sh
ABLATION=full bash examples/run_lunarlander_rl.sh
ABLATION=dense_only bash examples/run_lunarlander_rl.sh
```

## Important note about LunarLander

The LunarLander path is an engineering sanity check only. It is meant to validate:

- per-step reward computation
- decomposed reward logging
- token/step reward distribution
- shaped-reward ablations

It is not evidence of VLA transfer or manipulation-task generalization.

## Validation

Focused test files:

- [tests/test_lunarlander_reward.py](/Users/daeron/LunarLander-GRPO-dense-/tests/test_lunarlander_reward.py)
- [tests/test_lunarlander_reward_manager.py](/Users/daeron/LunarLander-GRPO-dense-/tests/test_lunarlander_reward_manager.py)

Run them with:

```bash
pytest -q tests/test_lunarlander_reward.py tests/test_lunarlander_reward_manager.py
```

If you want a report scaffold for experiments, use [examples/lunarlander_report_template.md](/Users/daeron/LunarLander-GRPO-dense-/examples/lunarlander_report_template.md).
