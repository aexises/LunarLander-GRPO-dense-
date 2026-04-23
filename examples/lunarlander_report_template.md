# LunarLander Reward Sanity Check Report

## Setup

- Environment: `LunarLander-v3`
- Training configuration:
- Reward configuration:
- Seeds:

## Reward Definition

- Total reward:
- Subgoal reward:
- Progress reward:
- Smoothness reward:
- Final reward:
- Phase thresholds:
- Success thresholds:

## Ablations

| Ablation | Reward mode | w_sub | w_prog | w_smooth | w_final | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| terminal_only | terminal_only | 0.0 | 0.0 | 0.0 | 1.0 | Binary terminal-success baseline |
| terminal_smooth | lunarlander_shaped | 0.0 | 0.0 | 0.005 | 1.0 | Terminal reward plus smoothness |
| terminal_sub_prog | lunarlander_shaped | 0.10 | 0.30 | 0.0 | 1.0 | Terminal reward plus subgoal and progress |
| full | lunarlander_shaped | 0.10 | 0.30 | 0.005 | 1.0 | Full current shaped reward |
| dense_only | lunarlander_shaped | 0.10 | 0.30 | 0.005 | 0.0 | Dense shaping with terminal reward removed |

## Results

- Success-rate curves:
- Shaped-reward curves:
- Action-switch statistics:
- Fuel proxy statistics:
- Phase-progress statistics:
- Example successful trajectories:
- Example failed trajectories:

## Interpretation

- Did dense reward improve learning speed or stability?
- Did smoothness reduce action switching?
- Did progress reward meaningfully correlate with success?
- Did shaped reward stay aligned with real task success?
- Reminder: LunarLander is a classical control sanity check, not a VLA benchmark.
