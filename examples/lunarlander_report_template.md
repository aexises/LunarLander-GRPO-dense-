# LunarLander Reward Sanity Check Report

## Setup

- Environment: `LunarLander-v2`
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

| Ablation | w_sub | w_prog | w_smooth | w_final | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| terminal_only | 0.0 | 0.0 | 0.0 | 1.0 | Baseline |
| terminal_smooth | 0.0 | 0.0 | >0 | 1.0 | Smoothness regularization only |
| terminal_sub_prog | >0 | >0 | 0.0 | 1.0 | Dense shaping without smoothness |
| full | >0 | >0 | >0 | 1.0 | Full shaped reward |
| dense_only | >0 | >0 | >0 | 0.0 | Diagnostic and potentially misaligned |

## Results

- Success-rate curves:
- Shaped-reward curves:
- Action-switch statistics:
- Phase-progress statistics:
- Example successful trajectories:
- Example failed trajectories:

## Interpretation

- Did dense reward improve learning speed or stability?
- Did smoothness reduce action switching?
- Did progress reward meaningfully correlate with success?
- Did shaped reward stay aligned with real task success?
- Reminder: LunarLander is a classical control sanity check, not a VLA benchmark.
