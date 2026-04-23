# Setup

This repo supports two practical setup paths:

## Option 1: LunarLander-only setup

Use this if you only need the local LunarLander sanity-check workflows.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lunarlander.txt
```

This installs the packages needed for:

- `gymnasium[box2d]`
- `torch`
- `ray`
- `hydra-core`
- `tensordict`
- plotting and report generation

If Box2D fails to build on your machine, LunarLander training will not start.

## Option 2: Full VLA environment

The original project is built around veRL plus external benchmark/model repos. A typical workspace layout is:

```text
your_workspace/
├── SimpleVLA-RL/
├── verl/
├── openvla-oft/
├── LIBERO/
└── RoboTwin/
```

### veRL

Follow the official veRL installation guide for a Python 3.10 environment. This checkout has been used with the veRL `v0.2.x` / `v0.3.x` line.

### OpenVLA-OFT / LIBERO / RoboTwin

Use the upstream project installation instructions for those stacks. This repo keeps the original scripts:

- [examples/run_openvla_oft_rl_libero.sh](/Users/daeron/LunarLander-GRPO-dense-/examples/run_openvla_oft_rl_libero.sh)
- [examples/run_openvla_oft_rl_twin2.sh](/Users/daeron/LunarLander-GRPO-dense-/examples/run_openvla_oft_rl_twin2.sh)

### RoboTwin integration note

If you are using RoboTwin, the helper script is still:

```bash
bash copy_overwrite_robotwin2.sh <robotwin_path> <simplevla_rl_path>
```

## Repo-specific LunarLander notes

- The default LunarLander ablation suite is now anchor-based.
- Random-init LunarLander runs are separated into [examples/run_lunarlander_random_init_suite.sh](/Users/daeron/LunarLander-GRPO-dense-/examples/run_lunarlander_random_init_suite.sh).
- The bundled competent anchor lives at [verl/assets/lunarlander/lunarlander_baseline_clean_seed42.pt](/Users/daeron/LunarLander-GRPO-dense-/verl/assets/lunarlander/lunarlander_baseline_clean_seed42.pt).
- Suite reports and suite-level comparison plots can be regenerated from existing checkpoint directories without retraining.
