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
- Local metric logging and plot generation in [verl/utils/lunarlander_logging.py](/Users/daeron/LunarLander-GRPO-dense-/verl/utils/lunarlander_logging.py)
- Reward-manager support for per-step reward placement in [verl/trainer/main_ppo.py](/Users/daeron/LunarLander-GRPO-dense-/verl/trainer/main_ppo.py)
- Trainer logging and dataset plumbing for LunarLander in [verl/trainer/ppo/ray_trainer.py](/Users/daeron/LunarLander-GRPO-dense-/verl/trainer/ppo/ray_trainer.py) and [verl/utils/dataset/rob_dataset.py](/Users/daeron/LunarLander-GRPO-dense-/verl/utils/dataset/rob_dataset.py)
- A runnable ablation script at [examples/run_lunarlander_rl.sh](/Users/daeron/LunarLander-GRPO-dense-/examples/run_lunarlander_rl.sh)

## Quick start

### 1. Install LunarLander dependencies

If you only want the LunarLander sanity-check path for now, you do not need to set up the full VLA stack yet.

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the LunarLander training, testing, and plotting dependencies:

```bash
pip install -r requirements-lunarlander.txt
```

If you later want to run the original VLA pipelines too, then follow [SETUP.md](/Users/daeron/LunarLander-GRPO-dense-/SETUP.md) for the upstream project environment.

### 2. LunarLander compute requirements

If you only care about LunarLander, you do not need a GPU.

- Minimum for a local smoke run: 4 CPU cores, 8 GB RAM, and roughly 2 GB free disk space
- Recommended for smoother runs: 8 CPU cores, 16 GB RAM
- GPU: optional; the LunarLander script now defaults to CPU mode with `NUM_GPUS=0`
- OS note: `gymnasium[box2d]` must build and import successfully on your machine before training will start
- Gymnasium note: this repo uses `LunarLander-v3`
- Default validation now uses a fixed 32-seed deterministic eval set

### 3. Run LunarLander on this repo only

CPU-only default:

```bash
bash examples/run_lunarlander_rl.sh
```

Explicit CPU mode:

```bash
NUM_GPUS=0 bash examples/run_lunarlander_rl.sh
```

If you later run on a CUDA machine, you can request one GPU:

```bash
NUM_GPUS=1 bash examples/run_lunarlander_rl.sh
```

Choose an ablation:

```bash
ABLATION=terminal_only bash examples/run_lunarlander_rl.sh
ABLATION=terminal_smooth bash examples/run_lunarlander_rl.sh
ABLATION=terminal_sub_prog bash examples/run_lunarlander_rl.sh
ABLATION=full bash examples/run_lunarlander_rl.sh
ABLATION=dense_only bash examples/run_lunarlander_rl.sh
```

Run the full ablation suite and generate an aggregate markdown report:

```bash
bash examples/run_lunarlander_ablation_suite.sh
```

### 4. Run the original VLA training scripts

Examples:

```bash
bash examples/run_openvla_oft_rl_libero.sh
bash examples/run_openvla_oft_rl_twin2.sh
```

You can ignore this section if you only want LunarLander.

## Important note about LunarLander

The LunarLander path is an engineering sanity check only. It is meant to validate:

- per-step reward computation
- decomposed reward logging
- token/step reward distribution
- shaped-reward ablations
- phase-stabilized reward shaping with explicit `APPROACH -> ALIGN -> DESCEND -> TOUCHDOWN` thresholds
- one-time progress bonuses on first entry into a new phase
- delta-based subgoal shaping instead of pure state-penalty accumulation
- audit traces for both train and validation episodes

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

## Logged outputs

LunarLander runs now write local artifacts under `trainer.default_local_dir`, including:

- `metrics_history.jsonl`
- `metrics_history.csv`
- `audit_traces_train.csv`
- `audit_traces_train.jsonl`
- `audit_traces_val.csv`
- `audit_traces_val.jsonl`
- `run_config_snapshot.json`
- `plots/reward_overview.png`
- `plots/reward_components_train.png`
- `plots/task_metrics_train.png`
- `plots/alignment_diagnostics.png`
- `run_report.md`
- `trajectory_dumps/*.json`
- `trajectory_dumps/*.png`

The ablation-suite helper also writes an aggregate report such as `checkpoints/SimpleVLA-RL/lunarlander_grpo_suite_report.md`.

If you rerun the same LunarLander experiment directory, the local metric files, audit traces, plots, and trajectory dumps are reset at the start of the new run so old diagnostics do not get mixed into the new evidence.

The metrics history now also includes:

- phase step counts, phase visit rates, and phase transition counts
- per-window reward dominance summaries for successful vs failed episodes
- audit-trace consistency checks comparing traced success/crash rates against logged metrics when a full audit window is captured
