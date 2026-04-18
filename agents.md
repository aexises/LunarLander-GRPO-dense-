Below is a complete AGENTS.md you can give to an AI coding agent.

# AGENTS.md
## Title
Implement step-distributed shaped reward for LunarLander in `SimpleVLA-RL` as a technical sanity check before VLA-style reward transfer
---
## 1. Objective
Extend the existing `SimpleVLA-RL` codebase to support **step-distributed shaped rewards** on a simple control task based on **LunarLander**, while preserving compatibility with the current training pipeline as much as possible.
This is **not** a claim that LunarLander is a good final benchmark for VLA reinforcement learning. It is only a **technical and algorithmic sanity check** to verify that:
1. the framework can consume and optimize a **dense, decomposed reward** rather than only a terminal outcome reward,
2. reward can be assigned to **individual steps/tokens** instead of only the last valid action token,
3. the resulting training behavior is sensible under several reward ablations,
4. the same implementation pattern can later be transferred to a more VLA-like environment.
`SimpleVLA-RL` is currently described as a VLA RL framework built on veRL, with VLA-specific modules such as `verl/trainer/main_ppo.py`, `verl/trainer/ppo/ray_trainer.py`, and VLA-specific rollout/worker code, and it explicitly highlights **binary (0/1) outcome rewards** as the default reward design. The README also states that `RobRewardManager` in `main_ppo.py` handles reward distribution.  [oai_citation:0‡GitHub](https://github.com/PRIME-RL/SimpleVLA-RL)
The reference dense-reward decomposition to implement is:
\[
r_t = w_1 r_t^{sub} + w_2 r_t^{prog} + w_3 r_t^{smooth} + w_4 r_T^{final}
\]
where:
- \(r_t^{sub}\): subgoal proximity reward,
- \(r_t^{prog}\): progress reward,
- \(r_t^{smooth}\): smoothness penalty,
- \(r_T^{final}\): terminal success reward.
This decomposition is inspired by robot RL reward shaping of the kind used in ReinboT, where reward is broken into subgoal achievement, task progress, smoothness, and terminal success; the public ReinboT repository includes a `reward_calculation.py` file and is tied to the paper *ReinboT: Amplifying Robot Visual-Language Manipulation with Reinforcement Learning*.  [oai_citation:1‡GitHub](https://github.com/COST-97/reinboT)
---
## 2. Scientific intent
The implementation must test the following hypothesis:
> Replacing terminal-only reward assignment with a scientifically defined, step-distributed reward decomposition yields a denser and more informative learning signal, while preserving alignment with final task success.
This hypothesis must be tested on LunarLander as a **controlled engineering benchmark**, not as evidence of VLA-level generalization.
### Important interpretation constraint
A positive result on LunarLander means only:
- the reward decomposition is implementable,
- the reward signal is denser,
- the optimization code can use per-step reward,
- the shaped reward appears behaviorally reasonable.
It does **not** prove that the same reward decomposition will work on VLA manipulation tasks.
---
## 3. Why LunarLander is used here
LunarLander is **not** chosen because it is VLA-like. It is chosen because it is:
- lightweight,
- easy to run locally,
- easy to inspect,
- suitable for controlled reward-engineering experiments,
- sufficient for validating per-step reward plumbing and reward attribution.
### Known limitations
The agent must explicitly acknowledge in code comments and in the final report:
- LunarLander is a classical control environment, not a vision-language-action environment.
- The subgoal and progress rewards here are **phase-based approximations**, not semantic manipulation subgoals.
- The result is only a **sanity check** before moving to a more VLA-like environment.
---
## 4. Existing framework facts that must be respected
The current codebase structure and stated roles in the repository README indicate:
- `verl/trainer/main_ppo.py` is the main entry point and contains `RobRewardManager`,
- `verl/trainer/ppo/ray_trainer.py` contains the main RL loop and advantage computation,
- the framework is built on veRL and currently emphasizes binary outcome reward design.  [oai_citation:2‡GitHub](https://github.com/PRIME-RL/SimpleVLA-RL)
Therefore:
1. **Do not rewrite the whole trainer first.**
2. First modify the reward production and reward distribution interfaces.
3. Preserve existing training logic unless step-distributed reward cannot be passed through without a local refactor.
---
## 5. High-level implementation goal
Implement a shaped-reward pipeline with these properties:
1. reward is computed at **each environment step**,
2. reward components are logged separately,
3. total reward per step is assembled from weighted components,
4. reward is distributed to the correct action-step or token span,
5. training can run in multiple ablation modes,
6. evaluation measures **true task success**, not only shaped reward.
---
## 6. Reward design for LunarLander
Use a **phase-based landing decomposition**.
### 6.1 State variables
Assume standard LunarLander state provides at least:
- horizontal position \(x_t\),
- vertical position \(y_t\),
- horizontal velocity \(v_{x,t}\),
- vertical velocity \(v_{y,t}\),
- angle \(\theta_t\),
- angular velocity \(\omega_t\),
- left-leg contact,
- right-leg contact.
If wrapper/state format differs, adapt accordingly but preserve the same semantics.
### 6.2 Landing phases
Define four phases:
1. `APPROACH`
2. `ALIGN`
3. `DESCEND`
4. `TOUCHDOWN`
These phases are not environment-native; they are analytical abstractions for reward shaping.
#### Suggested phase transition logic
Use deterministic threshold rules.
Example default rules:
- `APPROACH`:
  active when far from center or too high above pad
- `ALIGN`:
  active when close enough horizontally but angle is still not sufficiently stabilized
- `DESCEND`:
  active when mostly aligned and descending toward the pad
- `TOUCHDOWN`:
  active when close to landing and reducing final velocities
A simple threshold implementation is acceptable. The code must expose thresholds in config.
---
## 7. Formal reward definitions
### 7.1 Total reward
For timestep \(t\):
\[
r_t^{total} = w_1 r_t^{sub} + w_2 r_t^{prog} + w_3 r_t^{smooth}
\]
At terminal timestep \(T\):
\[
r_T^{total} = w_1 r_T^{sub} + w_2 r_T^{prog} + w_3 r_T^{smooth} + w_4 r_T^{final}
\]
The terminal term must be applied **only at episode end**.
---
### 7.2 Subgoal reward \(r_t^{sub}\)
This is a **proximity-to-current-phase-target** reward.
It must depend on the active phase.
#### Recommended definitions
##### Phase: `APPROACH`
Encourage horizontal centering and reduction of large lateral motion.
\[
r_t^{sub} = -\alpha_1 |x_t| - \alpha_2 |v_{x,t}|
\]
##### Phase: `ALIGN`
Encourage center alignment and low orientation error.
\[
r_t^{sub} = -\beta_1 |x_t| - \beta_2 |\theta_t| - \beta_3 |\omega_t|
\]
##### Phase: `DESCEND`
Encourage controlled descent.
\[
r_t^{sub} = -\gamma_1 |y_t - y^\star| - \gamma_2 |v_{y,t}| - \gamma_3 |x_t|
\]
where \(y^\star\) is a target descent corridor height or a phase-specific desired height band.
##### Phase: `TOUCHDOWN`
Encourage soft final stabilization.
\[
r_t^{sub} = -\delta_1 |v_{x,t}| - \delta_2 |v_{y,t}| - \delta_3 |\theta_t| - \delta_4 |\omega_t|
\]
#### Implementation note
All coefficients must be configurable.
#### Normalization
Subgoal reward must be scaled so that it does not dominate terminal success. Clip or normalize if needed.
A reasonable default is to keep \(r_t^{sub}\) in a small range such as approximately \([-1, 0]\) or \([-0.5, 0]\) per step after weighting.
---
### 7.3 Progress reward \(r_t^{prog}\)
This is a **one-time transition reward** for advancing through landing phases.
It must **not** be granted repeatedly for remaining in the same phase.
#### Definition
Let \(p_t\) be phase index at time \(t\), with:
- `APPROACH = 0`
- `ALIGN = 1`
- `DESCEND = 2`
- `TOUCHDOWN = 3`
Then:
\[
r_t^{prog} =
\begin{cases}
c_{p_t}, & \text{if } p_t > p_{t-1} \\
0, & \text{otherwise}
\end{cases}
\]
where \(c_{p_t}\) is a configurable positive reward for entering a later phase.
#### Default values
Example:
- entering `ALIGN`: `+0.25`
- entering `DESCEND`: `+0.25`
- entering `TOUCHDOWN`: `+0.25`
- successful final touchdown can be represented by \(r_T^{final}\), not by repeated progress reward
#### Key rule
Progress reward must be **event-based**, not state-persistent.
---
### 7.4 Smoothness reward \(r_t^{smooth}\)
For discrete LunarLander actions, define a low-dimensional actuator vector.
Suggested mapping:
- `0 = NOOP`      → \([0, 0]\)
- `1 = LEFT`      → \([-1, 0]\)
- `2 = MAIN`      → \([0, 1]\)
- `3 = RIGHT`     → \([1, 0]\)
Then define:
\[
r_t^{smooth} = - \|u_t - u_{t-1}\|^2
\]
with \(u_t\) being the vector for action \(a_t\).
This penalizes abrupt switching between engines.
#### First-step rule
If there is no previous action, set:
\[
r_0^{smooth} = 0
\]
#### Optional alternative
The code may additionally support a simpler discrete switch penalty:
\[
r_t^{smooth} =
\begin{cases}
-\lambda, & a_t \neq a_{t-1} \\
0, & a_t = a_{t-1}
\end{cases}
\]
but the vector-norm formulation should be the default.
---
### 7.5 Final reward \(r_T^{final}\)
Use a binary terminal reward.
\[
r_T^{final} =
\begin{cases}
1, & \text{successful landing} \\
0, & \text{otherwise}
\end{cases}
\]
A successful landing should be defined using environment termination signal and landing success logic. If the base environment already provides a success notion, use it. Otherwise define success using a combination of:
- both legs in stable contact,
- low vertical speed,
- low horizontal speed,
- small angle,
- no crash.
The success definition must be explicit and logged.
---
## 8. Default reward weights
Start with:
\[
w_1 = 0.10,\quad
w_2 = 0.30,\quad
w_3 = 0.02,\quad
w_4 = 1.00
\]
Rationale:
- terminal success must dominate,
- progress should be informative,
- subgoal proximity should guide but not override final objective,
- smoothness should regularize, not suppress exploration.
All weights must be configurable.
---
## 9. Required ablation experiments
Implement the following reward configurations.
### A. Terminal only
- `w_sub = 0.0`
- `w_prog = 0.0`
- `w_smooth = 0.0`
- `w_final = 1.0`
### B. Terminal + smoothness
- `w_sub = 0.0`
- `w_prog = 0.0`
- `w_smooth > 0`
- `w_final = 1.0`
### C. Terminal + subgoal + progress
- `w_sub > 0`
- `w_prog > 0`
- `w_smooth = 0.0`
- `w_final = 1.0`
### D. Full reward
- `w_sub > 0`
- `w_prog > 0`
- `w_smooth > 0`
- `w_final = 1.0`
### E. Dense only stress test
- `w_sub > 0`
- `w_prog > 0`
- `w_smooth > 0`
- `w_final = 0.0`
This configuration is mainly diagnostic and should be clearly marked as potentially misaligned.
---
## 10. Implementation constraints
### 10.1 Minimal-intrusion rule
The first implementation pass must:
- avoid unnecessary rewrites,
- preserve existing trainer logic,
- only change interfaces that are necessary to carry per-step rewards.
### 10.2 Backward compatibility
The binary terminal-only mode must still work.
### 10.3 Reproducibility
All new thresholds, reward coefficients, and weights must live in config.
### 10.4 Logging discipline
Every reward component must be logged separately.
---
## 11. Required code changes
### 11.1 Add a reward module
Create a new module, for example:
`verl/utils/lunarlander_shaped_reward.py`
It must contain:
- phase classification,
- subgoal reward computation,
- progress reward computation,
- smoothness reward computation,
- terminal reward computation,
- total reward assembly,
- configuration dataclasses or config-parsing helpers.
#### Required public API
```python
class LunarLanderRewardConfig:
    ...
class LunarLanderRewardState:
    ...
def classify_phase(obs, config) -> int:
    ...
def compute_subgoal_reward(obs, phase, config) -> float:
    ...
def compute_progress_reward(prev_phase, phase, config) -> float:
    ...
def compute_smoothness_reward(prev_action, action, config) -> float:
    ...
def compute_final_reward(obs, terminated, truncated, info, config) -> float:
    ...
def compute_step_reward(
    obs,
    prev_obs,
    action,
    prev_action,
    phase,
    prev_phase,
    terminated,
    truncated,
    info,
    config,
):
    """
    Return:
        {
            "r_sub": float,
            "r_prog": float,
            "r_smooth": float,
            "r_final": float,
            "r_total": float,
            "phase": int,
            "success": bool,
        }
    """

⸻

11.2 Modify rollout collection

Locate the environment interaction / rollout code used by the current framework for data collection. The README describes VLA-specific rollout modules under the verl tree and states that rollout and reward collection are part of the training stack.  ￼

Extend rollout outputs so that each episode stores:

* observation sequence,
* action sequence,
* phase sequence,
* per-step:
    * r_sub
    * r_prog
    * r_smooth
    * r_final
    * r_total
* episode-level:
    * success
    * episode_length
    * crash
    * fuel_proxy if available
    * num_action_switches

Required behavior

For each environment step:

1. compute phase,
2. compute reward components,
3. append component-wise rewards,
4. append total step reward,
5. store enough metadata for debugging.

⸻

11.3 Modify reward distribution logic

The repository states that RobRewardManager in verl/trainer/main_ppo.py is used for reward distribution.  ￼

Current logic appears oriented toward assigning reward to the last valid action token only. Replace or extend this behavior so that the framework can assign reward to the correct step or token range corresponding to each environment step.

Required modes

Implement a config flag:

* reward_distribution_mode = "last_token_per_step"
* optional future mode: "uniform_within_step"

Default must be:

* "last_token_per_step"

Semantics

If one environment step corresponds to a span of action tokens:

* compute reward for the environment step,
* assign that reward to the last action token of that step-span.

This is the simplest correct causal mapping for the first implementation.

Important

Do not collapse all rewards into only the final episode token.

⸻

11.4 Keep trainer update logic mostly unchanged

Do not redesign PPO/GRPO update logic unless absolutely required.

If the trainer already accepts a reward tensor, then:

* pass the new per-step/per-token reward tensor through the existing advantage computation,
* verify shapes and masks carefully,
* preserve current batching conventions.

⸻

11.5 Add configuration support

Create new config entries for:

* environment selection,
* reward mode,
* reward weights,
* phase thresholds,
* coefficient scales,
* success thresholds,
* reward distribution mode.

Example config keys:

reward:
  mode: lunarlander_shaped
  distribution_mode: last_token_per_step
  weights:
    sub: 0.10
    prog: 0.30
    smooth: 0.02
    final: 1.00
  phase_thresholds:
    center_x_abs_for_align: 0.20
    angle_abs_for_descend: 0.15
    height_for_touchdown: 0.30
    vertical_speed_abs_for_touchdown: 0.20
  coeffs:
    approach_x: 1.0
    approach_vx: 0.3
    align_x: 0.7
    align_theta: 1.0
    align_omega: 0.3
    descend_y: 0.5
    descend_vy: 1.0
    descend_x: 0.3
    touchdown_vx: 0.7
    touchdown_vy: 1.0
    touchdown_theta: 0.7
    touchdown_omega: 0.3
  progress_rewards:
    enter_align: 0.25
    enter_descend: 0.25
    enter_touchdown: 0.25
  smoothness:
    mode: vector_l2
  success:
    max_abs_x: 0.20
    max_abs_vx: 0.20
    max_abs_vy: 0.20
    max_abs_theta: 0.15

⸻

12. Metrics that must be logged

12.1 Core training metrics

Per training window / evaluation window, log:

* mean_total_reward
* mean_r_sub
* mean_r_prog
* mean_r_smooth
* mean_r_final
* success_rate
* crash_rate
* mean_episode_length
* mean_num_action_switches
* mean_abs_final_x
* mean_abs_final_vx
* mean_abs_final_vy
* mean_abs_final_theta

12.2 Alignment diagnostics

Also log:

* correlation between episode_total_shaped_reward and success,
* correlation between mean_r_sub and success,
* correlation between mean_r_prog and success,
* correlation between mean_r_smooth and success.

12.3 Reward hacking diagnostics

Flag a warning if:

* shaped reward increases while success rate remains flat or decreases.

⸻

13. Evaluation protocol

13.1 Deterministic evaluation

At regular intervals, run evaluation with deterministic or low-temperature action selection.

13.2 Fixed seeds

Use a fixed evaluation seed set.

13.3 Report both shaped and task metrics

Always report:

* success rate,
* crash rate,
* average shaped reward,
* reward components.

Do not report only shaped reward.

13.4 Example trajectory dump

Save several successful and failed trajectories with:

* state summary,
* phase transitions,
* per-step reward decomposition,
* final outcome.

⸻

14. Acceptance criteria

The implementation is complete only if all items below are satisfied.

14.1 Functional criteria

* terminal-only reward mode still runs,
* shaped reward mode runs without shape/mask errors,
* per-step reward components are correctly logged,
* reward distribution no longer assigns everything only to the final episode token,
* reward config is externalized and reproducible.

14.2 Scientific criteria

The experiments must produce a report showing:

1. whether shaped reward improves learning speed or stability relative to terminal-only,
2. whether smoothness reduces action switching,
3. whether progress reward yields meaningful phase advancement,
4. whether total shaped reward remains aligned with true landing success.

14.3 Negative-result policy

If dense reward increases but success rate does not improve, this must be reported as a likely reward-misalignment result, not hidden.

⸻

15. Non-goals

The agent must not do the following in the first pass:

* redesign the full RL algorithm,
* claim VLA transferability from LunarLander results,
* hard-code environment thresholds without config exposure,
* optimize only shaped reward while ignoring task success,
* silently change success definition between runs.

⸻

16. Recommended implementation order

Phase 1 — reward module

Implement reward computation in isolation with unit tests.

Phase 2 — rollout integration

Attach reward computation to step collection and store component-wise rewards.

Phase 3 — reward distribution integration

Pass per-step reward into the existing reward manager and trainer.

Phase 4 — ablation configs

Add terminal-only, partial, and full shaped-reward configs.

Phase 5 — evaluation and report

Generate plots and summary tables.

⸻

17. Unit tests and sanity tests

Add tests for:

17.1 Phase classification

Synthetic observations should map to expected phases.

17.2 Progress reward

Phase transition gives reward once; remaining in the same phase gives zero.

17.3 Smoothness reward

Repeated same action has no penalty or lower penalty than abrupt switching.

17.4 Final reward

Success state yields 1, failure yields 0.

17.5 Total reward composition

Weighted sum matches expected arithmetic.

17.6 Distribution mapping

Per-step reward maps to the expected token/step position.

⸻

18. Deliverables

The final output of the implementation must include:

1. code changes,
2. config files,
3. tests,
4. a concise experiment script,
5. a short report with:
    * setup,
    * reward definitions,
    * ablation table,
    * plots,
    * interpretation.

⸻

19. Expected final report structure

The final report should contain:

19.1 Setup

* environment version,
* training settings,
* reward settings,
* seed list.

19.2 Reward definition

* exact formulas,
* thresholds,
* weights.

19.3 Ablations

* terminal-only,
* terminal+smooth,
* terminal+sub+prog,
* full,
* dense-only diagnostic.

19.4 Results

* success rate curves,
* shaped reward curves,
* switching behavior,
* phase progress statistics.

19.5 Interpretation

* did dense reward help,
* was it aligned,
* what failed,
* what should transfer to the next, more VLA-like environment.

⸻

20. Final engineering principle

The correct first milestone is not “beat LunarLander with a fancy reward.”

The correct first milestone is:

demonstrate that SimpleVLA-RL can consume and optimize a scientifically defined, step-distributed reward decomposition, while preserving alignment with final task success.

Only after that should the same machinery be moved to a more VLA-like environment.

A clean next step is to turn this into a repo-specific version once you share your actual `simplevla-rl` tree.