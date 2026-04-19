# AGENTS.md
## Title
Implement the next improvement pass for LunarLander shaped reward: repair early-phase shaping and restore train-time optimization
---
## 1. Objective
Proceed with the next targeted improvement pass on the LunarLander shaped-reward branch in `SimpleVLA-RL`.
The previous pass improved reward structure and validation behavior, but train optimization regressed:
- final train success: `0.09375`
- best train success: `0.265625`
- final val success: `0.3071428657`
- final train total reward: `0.1704571843`
- final val total reward: `0.4506697819`
- final reward hacking warning: `0.0`  [oai_citation:0‡metrics_summary.json](sediment://file_000000005c3c7246af17d41b13ff1030)
The important structural gains from the last pass must be preserved:
- `DESCEND` and `TOUCHDOWN` are now active,
- `progress` is no longer inert,
- `smoothness` is no longer dominant,
- evaluation coverage is explicit and full.
However, the main remaining defect is that **`APPROACH` is still effectively unused**, which strongly suggests weak early-phase shaping and poor train-time bootstrapping.
This pass must therefore focus on:
1. making `APPROACH` a real and frequently used phase,
2. improving early-flight shaping,
3. slightly rebalancing dense rewards toward early useful progress,
4. preserving the healthier late-phase structure from the last pass.
---
## 2. High-level strategy
Do **not** revert the previous redesign.
Instead, keep the following improvements:
- delta-based subgoal shaping,
- explicit phase progression,
- weak smoothness regularization,
- terminal final reward,
- full evaluation coverage.
Then patch the remaining weak point:
- **the early phase is under-shaped and under-visited**.
This implementation pass should specifically improve:
- train-time phase coverage,
- early-phase learning signal,
- train success recovery,
while preserving:
- validation quality,
- no reward hacking,
- no smoothness domination.
---
## 3. Required changes
---
# 3.1 Make `APPROACH` the default fallback phase
## Problem
`APPROACH` is still effectively absent. This means the policy receives too little meaningful shaping before it is already somewhat aligned.
## Required change
Refactor phase classification so that `APPROACH` becomes the default fallback for all states that do not satisfy stricter later-phase conditions.
## Required logical order
Use this order:
1. `TOUCHDOWN`
2. `DESCEND`
3. `ALIGN`
4. otherwise `APPROACH`
## Required principle
Do **not** make `APPROACH` depend on a narrow positive condition.  
It should be the default for states that are not yet sufficiently good for later phases.
## Implementation note
If the current classifier uses independent `if` blocks or overly broad `ALIGN` conditions, rewrite it into mutually exclusive `if / elif / else` logic.
---
# 3.2 Tighten `ALIGN` slightly so it does not absorb most early-flight states
## Problem
`ALIGN` is still likely catching too many states that should remain in `APPROACH`.
## Required change
Entering `ALIGN` should require moderate centering and reasonable attitude, but not be so broad that almost all non-terminal flight becomes `ALIGN`.
## Required thresholds
Use:
```yaml id="j3n6vs"
phase_thresholds:
  center_x_abs_for_align: 0.30
  center_x_abs_for_descend: 0.25
  center_x_abs_for_touchdown: 0.18
  angle_abs_for_descend: 0.20
  angle_abs_for_touchdown: 0.12
  height_for_descend: 0.90
  height_for_touchdown: 0.35
  horizontal_speed_abs_for_touchdown: 0.25
  vertical_speed_abs_for_touchdown: 0.25
  vertical_speed_min_for_descend: -0.10
  descent_target_height: 0.5

Notes

This is only a small change from the previous pass:

* center_x_abs_for_align becomes stricter (0.30 instead of 0.35)
* later-phase thresholds stay mostly as-is to preserve the improved late structure

This should increase APPROACH occupancy without collapsing later phases.

⸻

3.3 Improve early-phase subgoal shaping

Problem

Delta-based shaping is the right direction, but early-flight subgoal shaping is probably still too weak or noisy.

Required change

Keep delta-based shaping, but update the APPROACH error function to include mild attitude shaping and reduce noisy overreaction.

Required phase-specific error functions

APPROACH

Replace the current APPROACH error with:

[
f_{\text{approach}}(s) = |x| + 0.2|v_x| + 0.1|\theta|
]

ALIGN

Use:

[
f_{\text{align}}(s) = 0.7|x| + 1.0|\theta| + 0.3|\omega|
]

DESCEND

Use:

[
f_{\text{descend}}(s) = 0.5|y-y^*| + 1.0|v_y| + 0.3|x|
]

TOUCHDOWN

Use:

[
f_{\text{touchdown}}(s) = 0.7|v_x| + 1.0|v_y| + 0.7|\theta| + 0.3|\omega|
]

Delta-based shaping rule

Keep:

[
r_t^{sub} = \mathrm{clip}(f(s_{t-1}) - f(s_t), -1, 1)
]

Required early-phase stabilization

Add a small deadband to reduce noise:

* if (|f(s_{t-1}) - f(s_t)| < \epsilon), set r_sub = 0
* use a small default such as:
    * epsilon = 0.01

Intended effect

This should make APPROACH shaping:

* more informative,
* less noisy,
* more useful during early training.

⸻

3.4 Add lightweight micro-progress rewards inside APPROACH

Problem

Current progress reward is meaningful later, but the policy still lacks an early curriculum signal before major phase transitions.

Required change

Add one-time early approach corridor bonuses.

Required new micro-progress events

Award a small one-time bonus when an episode first enters:

* abs(x) < 0.50
* abs(x) < 0.35

Required reward values

Use:

micro_progress_rewards:
  enter_x_corridor_050: 0.05
  enter_x_corridor_035: 0.08

Required behavior

* reward each threshold crossing only once per episode
* do not repeatedly reward hovering within the corridor
* these are in addition to regular phase-transition rewards

Intended effect

This gives the policy an early useful signal before it can reliably reach ALIGN / DESCEND.

⸻

3.5 Rebalance reward weights slightly toward subgoal and slightly away from progress

Problem

The previous pass improved stage structure, but train optimization likely still depends too much on relatively sparse progress events.

Required new weights

Set:

weights:
  sub: 0.20
  prog: 0.40
  smooth: 0.002
  final: 1.0

Rationale

* increase sub because delta-based shaping is now more meaningful
* slightly reduce prog so training depends less on sparse transition events
* keep smooth weak
* keep final dominant

Constraint

Do not increase smooth.
Do not reduce final.

⸻

3.6 Slightly flatten progress rewards across phases

Problem

Later progress rewards may still be too concentrated relative to early progress needs.

Required new phase transition rewards

Set:

progress_rewards:
  enter_align: 0.25
  enter_descend: 0.30
  enter_touchdown: 0.40

Required behavior

* each phase transition reward is granted only once per first entry
* no repeated reward for staying in a phase
* no repeated gain from oscillation unless explicitly intended

Intended effect

This preserves meaningful later-phase reward while making early structured advancement slightly more useful.

⸻

3.7 Keep smoothness weak and unchanged otherwise

Required settings

Keep:

smoothness:
  mode: vector_l2
  switch_penalty: 0.1

and keep:

smooth: 0.002

Constraint

No further smoothness changes in this pass unless code consistency is broken.

The goal is to improve early shaping, not re-open the smoothness problem.

⸻

3.8 Keep final reward terminal and dominant

Required behavior

Keep final reward binary and terminal-only:

[
r_T^{final} \in {0,1}
]

Do not make final reward dense.

⸻

3.9 Keep success thresholds unchanged from the previous pass

Required settings

Keep:

success:
  max_abs_x: 0.25
  max_abs_vx: 0.25
  max_abs_vy: 0.25
  max_abs_theta: 0.18

No further success-threshold loosening in this pass.

⸻

4. Full reward config to implement

Implement this exact reward block:

reward:
  mode: lunarlander_shaped
  distribution_mode: last_token_per_step
  weights:
    sub: 0.20
    prog: 0.40
    smooth: 0.002
    final: 1.0
  phase_thresholds:
    center_x_abs_for_align: 0.30
    center_x_abs_for_descend: 0.25
    center_x_abs_for_touchdown: 0.18
    angle_abs_for_descend: 0.20
    angle_abs_for_touchdown: 0.12
    height_for_descend: 0.90
    height_for_touchdown: 0.35
    horizontal_speed_abs_for_touchdown: 0.25
    vertical_speed_abs_for_touchdown: 0.25
    vertical_speed_min_for_descend: -0.10
    descent_target_height: 0.5
  progress_rewards:
    enter_align: 0.25
    enter_descend: 0.30
    enter_touchdown: 0.40
  micro_progress_rewards:
    enter_x_corridor_050: 0.05
    enter_x_corridor_035: 0.08
  subgoal:
    deadband_eps: 0.01
  smoothness:
    mode: vector_l2
    switch_penalty: 0.1
  success:
    max_abs_x: 0.25
    max_abs_vx: 0.25
    max_abs_vy: 0.25
    max_abs_theta: 0.18

⸻

5. Required code changes

Update all relevant files/functions consistently.

You must update:

1. reward config schema / loading
2. phase classifier
3. subgoal reward computation
4. progress reward computation
5. micro-progress bookkeeping
6. run config snapshot generation
7. report generation
8. trace export fields if needed

Consistency requirement

The following must all reflect the same implemented logic:

* runtime reward computation
* exported traces
* config snapshot
* report text
* tests

Do not leave stale values or obsolete descriptions in any reporting path.

⸻

6. Logging requirements

This is an implementation pass, but logging must still support quick interpretation.

Preserve or log:

Per step:

* phase_cur
* r_sub_raw
* r_prog_raw
* r_smooth_raw
* r_final_raw
* weighted contributions
* r_total_step

Per window:

* success rate
* total reward
* mean R_sub
* mean R_prog
* mean R_smooth
* mean R_final
* phase counts
* mean number of phase transitions
* fuel proxy
* action switches

New required aggregates

Add:

* number of episodes that ever visited APPROACH
* number of episodes that received each micro-progress reward
* share of reward mass coming from:
    * subgoal
    * progress
    * micro-progress
    * smoothness
    * final

⸻

7. Evaluation requirements

Keep the healthier eval setup from the previous pass.

Required behavior

* evaluate on the fixed 32-seed set
* log full eval seed coverage
* preserve deterministic eval behavior if already used
* keep explicit seed hash reporting

Do not reduce evaluation quality.

⸻

8. Run protocol

Run a short implementation-validation pass after the changes.

Do not launch a large ablation suite yet.

Required run type

A short training pass long enough to observe:

* whether APPROACH is now genuinely used,
* whether train success recovers,
* whether val success is preserved or improved,
* whether late phases remain active,
* whether reward hacking stays at zero.

Keep enabled

* trace export
* config snapshot
* run report

⸻

9. Success criteria for this pass

This pass is successful if it preserves the structural improvements of the previous pass while improving early-phase learning.

Desired outcomes

1. APPROACH is now regularly visited in train and val
2. DESCEND and TOUCHDOWN remain active
3. train success improves materially above the previous pass final value 0.09375
4. val success stays at least around the previous pass level ~0.307 or improves
5. total reward remains positive
6. reward hacking warning remains 0
7. progress remains non-inert
8. smoothness remains a weak regularizer

Strong target

A clearly better pass would look like:

* final train success > 0.20
* final val success >= 0.30
* clear APPROACH usage
* preserved late-phase usage
* no reward hacking

⸻

10. Required deliverables after implementation

Return:

1. exact files changed
2. updated reward config snapshot
3. short run metrics
4. phase usage summary
5. micro-progress usage summary
6. comparison vs previous pass
7. short conclusion on whether early-phase shaping improved optimization

⸻

11. Final principle

Do not re-open the whole design.

The current design direction is better than before.
This pass is specifically about repairing the missing early-phase curriculum so that the healthier staged reward becomes easier to optimize during training.

This next pass should be judged mainly on whether it restores train performance while keeping the structural gains from the last run: active `DESCEND`/`TOUCHDOWN`, meaningful progress, reduced smoothness dominance, and no reward hacking.