# Instructions

All commands in this repo should be run from the repository root.

## Main workflows

### 1. Run a single competent-anchor LunarLander ablation

```bash
NUM_GPUS=0 ABLATION=terminal_only bash examples/run_lunarlander_rl.sh
NUM_GPUS=0 ABLATION=terminal_smooth bash examples/run_lunarlander_rl.sh
NUM_GPUS=0 ABLATION=terminal_sub_prog bash examples/run_lunarlander_rl.sh
NUM_GPUS=0 ABLATION=full bash examples/run_lunarlander_rl.sh
NUM_GPUS=0 ABLATION=dense_only bash examples/run_lunarlander_rl.sh
```

### 2. Run a single random-init ablation

```bash
NUM_GPUS=0 LUNARLANDER_POLICY_INIT=random_init ABLATION=full bash examples/run_lunarlander_rl.sh
```

### 3. Run the default anchor-based suite

```bash
bash examples/run_lunarlander_ablation_suite.sh
```

This script now runs only the competent-anchor ablations and writes a suite report plus suite-level comparison plots.

### 4. Run the separate random-init suite

```bash
bash examples/run_lunarlander_random_init_suite.sh
```

This is the opt-in comparison path for random-start policies.

### 5. Regenerate a suite report from existing run directories

Anchor-only example:

```bash
python3 examples/generate_lunarlander_report.py \
  checkpoints/SimpleVLA-RL/lunarlander_grpo_suite_terminal_only \
  checkpoints/SimpleVLA-RL/lunarlander_grpo_suite_terminal_smooth \
  checkpoints/SimpleVLA-RL/lunarlander_grpo_suite_terminal_sub_prog \
  checkpoints/SimpleVLA-RL/lunarlander_grpo_suite_full \
  checkpoints/SimpleVLA-RL/lunarlander_grpo_suite_dense_only \
  --output checkpoints/SimpleVLA-RL/lunarlander_grpo_suite_report.md
```

## Important artifact locations

- Per-run metrics summary: `checkpoints/.../<run>/metrics_summary.json`
- Per-run markdown report: `checkpoints/.../<run>/run_report.md`
- Suite markdown report: `checkpoints/.../*_report.md`
- Suite CSV summary: `checkpoints/.../*_report.csv`
- Suite comparison plots: `checkpoints/.../*_report_plots/*.png`

## Practical notes

- Competent-anchor runs use the bundled checkpoint at `verl/assets/lunarlander/lunarlander_baseline_clean_seed42.pt`.
- Random-init runs use a fresh `tanh` MLP unless overridden.
- Validation is deterministic on 32 fixed seeds.
- The suite report compares competent-anchor runs against `terminal_only` as the baseline.
- Random-init runs are excluded from the baseline-comparison tables and baseline-comparison plots unless you generate a random-init-only report explicitly.
