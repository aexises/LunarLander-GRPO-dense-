# SimpleVLA-RL

SimpleVLA-RL is a veRL-based RL codebase for VLA training plus a lightweight LunarLander reward-shaping sanity-check path. This checkout is a normal single-repo layout at the repository root.

## Repo layout

- `verl/`: trainer, workers, reward logic, logging, datasets
- `examples/`: runnable LunarLander and VLA scripts
- `tests/`: focused regression tests
- `requirements-lunarlander.txt`: minimal dependency set for the LunarLander path
- `SETUP.md`: full environment notes for the original VLA workflows
- `instructions.md`: practical day-to-day commands for this checkout

## LunarLander quick start

Create a small local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lunarlander.txt
```

Run a single ablation:

```bash
NUM_GPUS=0 ABLATION=full bash examples/run_lunarlander_rl.sh
```

Run the default anchor-based ablation suite:

```bash
bash examples/run_lunarlander_ablation_suite.sh
```

Run the separate random-init comparison suite:

```bash
bash examples/run_lunarlander_random_init_suite.sh
```

## LunarLander run modes

- Default LunarLander runs use a competent pretrained anchor at [verl/assets/lunarlander/lunarlander_baseline_clean_seed42.pt](/Users/daeron/LunarLander-GRPO-dense-/verl/assets/lunarlander/lunarlander_baseline_clean_seed42.pt).
- Random-init runs are still supported, but they are now opt-in through `LUNARLANDER_POLICY_INIT=random_init` or the dedicated random-init suite script.
- The environment is `LunarLander-v3`.
- Validation uses a fixed deterministic 32-seed set.

## LunarLander ablations

The ablation name maps directly to the weights passed by [examples/run_lunarlander_rl.sh](/Users/daeron/LunarLander-GRPO-dense-/examples/run_lunarlander_rl.sh).

| Ablation | Reward mode | `w_sub` | `w_prog` | `w_smooth` | `w_final` | What it means |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `terminal_only` | `terminal_only` | 0.00 | 0.00 | 0.00 | 1.00 | Pure binary terminal success baseline |
| `terminal_smooth` | `lunarlander_shaped` | 0.00 | 0.00 | 0.005 | 1.00 | Terminal reward plus action smoothness only |
| `terminal_sub_prog` | `lunarlander_shaped` | 0.10 | 0.30 | 0.00 | 1.00 | Terminal reward plus subgoal and phase-progress shaping |
| `full` | `lunarlander_shaped` | 0.10 | 0.30 | 0.005 | 1.00 | Full current shaped reward |
| `dense_only` | `lunarlander_shaped` | 0.10 | 0.30 | 0.005 | 0.00 | Dense shaping only, no terminal success reward |

## Exact reward definition

The per-step reward used by the shaped runs is:

```text
r_total = w_sub * r_sub
        + w_prog * (r_prog + r_micro)
        + w_smooth * r_smooth
        + w_final * r_final
```

Component definitions in the current code:

- `r_final`:
  terminal-only binary success, with success taken from an explicit env success flag when available, otherwise from full-episode return semantics (`episode_return >= 200.0`).
- `r_prog`:
  one-time phase-entry bonuses:
  `enter_align = 0.25`, `enter_descend = 0.25`, `enter_touchdown = 0.25`.
- `r_micro`:
  one-time early corridor bonuses.
  In the current shipped config these are all `0.0`, so micro-progress is present in the code path but inactive by default.
- `r_smooth`:
  vector-L2 action-switch penalty with `switch_penalty = 0.10`.
- `r_sub`:
  clipped phase-local potential improvement,
  `clip(phi(s_t) - phi(s_(t-1)), -1, 0)`.

Current phase-local subgoal potentials:

```text
APPROACH:  -(1.0 * |x| + 0.3 * |vx|)
ALIGN:     -(0.7 * |x| + 1.0 * |theta| + 0.3 * |omega|)
DESCEND:   -(0.5 * |y - 0.5| + 1.0 * |vy| + 0.3 * |x|)
TOUCHDOWN: -(0.7 * |vx| + 1.0 * |vy| + 0.7 * |theta| + 0.3 * |omega|)
```

Current phase thresholds:

- `APPROACH -> ALIGN`: `abs(x) <= 0.35`
- `ALIGN -> DESCEND`: `y <= 0.75`, `abs(x) <= 0.20`, `abs(theta) <= 0.15`, `vy <= -0.05`
- `DESCEND -> TOUCHDOWN`: `y <= 0.30`, `abs(x) <= 0.15`, `abs(theta) <= 0.10`, `abs(vx) <= 0.20`, `abs(vy) <= 0.20`

Current success thresholds:

- `abs(x) <= 0.20`
- `abs(vx) <= 0.20`
- `abs(vy) <= 0.20`
- `abs(theta) <= 0.15`

## Output artifacts

Each LunarLander run writes artifacts under `trainer.default_local_dir`, including:

- `metrics_history.jsonl`
- `metrics_history.csv`
- `metrics_summary.json`
- `run_config_snapshot.json`
- `run_report.md`
- `plots/*.png`
- `audit_traces_train.*`
- `audit_traces_val.*`
- `trajectory_dumps/*.json`
- `trajectory_dumps/*.png`

The suite report generator also writes comparison artifacts such as:

- `checkpoints/SimpleVLA-RL/lunarlander_grpo_suite_report.md`
- `checkpoints/SimpleVLA-RL/lunarlander_grpo_suite_report.csv`
- `checkpoints/SimpleVLA-RL/lunarlander_grpo_suite_report_plots/*.png`

Those suite-level comparison plots use `terminal_only` as the baseline and compare the other competent-anchor runs against it.

## Tests

Focused LunarLander checks:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=. \
pytest -q \
  tests/test_lunarlander_reward.py \
  tests/test_lunarlander_reward_manager.py \
  tests/test_lunarlander_checkpoint_eval.py \
  tests/test_lunarlander_logging.py \
  tests/test_lunarlander_report_script.py
```

## Notes

- LunarLander is an engineering sanity check, not evidence of VLA transfer.
- The current default shaped-reward config in [verl/trainer/config/ppo_trainer.yaml](/Users/daeron/LunarLander-GRPO-dense-/verl/trainer/config/ppo_trainer.yaml) uses `sub=0.10`, `prog=0.30`, `smooth=0.005`, `final=1.0`.
- The current staged shaping uses `APPROACH`, `ALIGN`, `DESCEND`, and `TOUCHDOWN`, plus one-time progress bonuses and terminal binary success.
