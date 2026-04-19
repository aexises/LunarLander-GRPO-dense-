# Instructions

This repository is now organized as a normal top-level checkout. The old nested `SimpleVLA-RL/` wrapper has been removed, so all commands should be run from the repo root.

## Repository layout

- `verl/` contains the training stack
- `examples/` contains runnable scripts
- `tests/` contains targeted tests
- `figs/`, `modified_codes/`, and top-level setup files remain at the root

## Install dependencies

For the LunarLander work in this repo, you can skip the full VLA setup for now.

Create and activate a local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the LunarLander runtime, testing, and plotting packages:

```bash
pip install -r requirements-lunarlander.txt
```

Only use [SETUP.md](/Users/daeron/LunarLander-GRPO-dense-/SETUP.md) when you are ready to run the original VLA workflows too.

## LunarLander resource needs

If you only want LunarLander, this repo can now run without a GPU.

- Minimum for a quick local run: 4 CPU cores and 8 GB RAM
- Recommended for longer runs: 8 CPU cores and 16 GB RAM
- Disk: keep a couple of GB free for the virtualenv, Ray temp files, checkpoints, plots, and trajectory dumps
- GPU: optional; use `NUM_GPUS=1` only on a CUDA machine
- Default behavior: `examples/run_lunarlander_rl.sh` now runs in CPU mode unless you override `NUM_GPUS`
- Gymnasium env id: `LunarLander-v3`
- Default eval protocol: fixed 32-seed deterministic validation

## Common commands

### Run LunarLander shaped-reward training

```bash
bash examples/run_lunarlander_rl.sh
```

### Force CPU mode on a laptop

```bash
NUM_GPUS=0 bash examples/run_lunarlander_rl.sh
```

### Use a GPU on a CUDA machine

```bash
NUM_GPUS=1 bash examples/run_lunarlander_rl.sh
```

### Run a specific LunarLander ablation

```bash
ABLATION=terminal_only bash examples/run_lunarlander_rl.sh
ABLATION=terminal_smooth bash examples/run_lunarlander_rl.sh
ABLATION=terminal_sub_prog bash examples/run_lunarlander_rl.sh
ABLATION=full bash examples/run_lunarlander_rl.sh
ABLATION=dense_only bash examples/run_lunarlander_rl.sh
```

### Run the whole ablation suite and build an aggregate report

```bash
bash examples/run_lunarlander_ablation_suite.sh
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

- The dedicated LunarLander dependency list lives in [requirements-lunarlander.txt](/Users/daeron/LunarLander-GRPO-dense-/requirements-lunarlander.txt).
- The LunarLander path currently expects `hydra-core`, `omegaconf`, `ray`, `torch`, `tensordict`, `gymnasium[box2d]`, `matplotlib`, `pandas`, and `numpy`.
- If `matplotlib` is installed, LunarLander runs also emit PNG plots automatically.
- The LunarLander worker is intentionally single-process for the first pass.
- The original OpenVLA/LIBERO/Robotwin paths are still present.
- Reward shaping for LunarLander is configured in `verl/trainer/config/ppo_trainer.yaml`.
- The current shaped reward uses explicit thresholds for `APPROACH`, `ALIGN`, `DESCEND`, and `TOUCHDOWN`, a delta-style subgoal term, and one-time progress bonuses per phase.
- Trajectory dumps and experiment outputs are written under the configured `trainer.default_local_dir`.
- Metric history is written to `metrics_history.jsonl` and `metrics_history.csv`.
- Step-level forensic traces are written to `audit_traces_train.csv` / `.jsonl` and `audit_traces_val.csv` / `.jsonl`.
- The first full audit window now defaults to 128 train episodes and 32 validation episodes, which is enough to compare trace-derived success rates against logged metrics.
- Metrics now include phase occupancy, phase transitions, weighted component totals, successful-vs-failed reward dominance summaries, and audit consistency checks.
- Summary plots are written to `plots/`, and evaluation trajectory graphics are written beside the trajectory JSON dumps.
- Each run also writes `run_report.md`.
- The suite runner writes a combined markdown report across ablations.
- Rerunning the same experiment directory resets the local metrics, plots, audit traces, and trajectory dumps for a clean diagnostic record.

## Suggested workflow

1. Create a local environment and install the LunarLander dependencies above.
2. Run the focused tests.
3. Start with `NUM_GPUS=0 bash examples/run_lunarlander_rl.sh` on a laptop or CPU-only machine.
4. Fill in `examples/lunarlander_report_template.md` with metrics and plots.
