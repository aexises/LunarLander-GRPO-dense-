from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verl.utils.lunarlander_shaped_reward import (
    APPROACH,
    PHASE_NAMES,
    LunarLanderRewardConfig,
    LunarLanderRewardState,
    classify_phase,
    compute_step_reward,
    stabilize_phase,
)
from verl.workers.lunarlander_workers import FUEL_PROXY_COST, LunarLanderPolicy


MICRO_PROGRESS_KEYS = (
    "enter_x_corridor_070",
    "enter_x_corridor_050",
    "enter_x_corridor_035",
)


@dataclass(frozen=True)
class CheckpointSelection:
    mode: str
    checkpoint_step: int
    checkpoint_path: Path
    saved_checkpoint_steps: list[int]
    best_logged_step: int | None
    best_logged_value: float | None
    final_logged_step: int | None
    final_logged_value: float | None
    selected_logged_value: float | None
    selection_warning: str | None = None


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _parse_metric_value(raw_value: str | None) -> float | None:
    if raw_value in ("", None):
        return None
    return float(raw_value)


def load_metrics_history(run_dir: Path) -> list[dict[str, str]]:
    history_path = run_dir / "metrics_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing metrics history at {history_path}")
    with history_path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def metric_values_by_step(rows: list[dict[str, str]], metric_key: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for row in rows:
        metric_value = _parse_metric_value(row.get(metric_key))
        step_raw = row.get("step")
        if metric_value is None or step_raw in ("", None):
            continue
        values[int(step_raw)] = metric_value
    return values


def saved_checkpoint_steps(run_dir: Path) -> list[int]:
    actor_dir = run_dir / "actor"
    if not actor_dir.exists():
        return []
    steps = []
    for child in actor_dir.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("global_step_"):
            continue
        checkpoint_path = child / "lunarlander_policy.pt"
        if not checkpoint_path.exists():
            continue
        step = int(child.name.removeprefix("global_step_"))
        steps.append(step)
    return sorted(steps)


def select_checkpoint_for_evaluation(
    run_dir: Path,
    mode: str = "best_saved",
    metric_key: str = "val/test_reward/success_rate",
    explicit_step: int | None = None,
) -> CheckpointSelection:
    run_dir = Path(run_dir)
    history_rows = load_metrics_history(run_dir)
    metrics_by_step = metric_values_by_step(history_rows, metric_key)
    saved_steps = saved_checkpoint_steps(run_dir)
    if not saved_steps:
        raise FileNotFoundError(f"No saved LunarLander checkpoints found under {run_dir / 'actor'}")

    best_logged_step = None
    best_logged_value = None
    if metrics_by_step:
        best_logged_step, best_logged_value = max(metrics_by_step.items(), key=lambda item: (item[1], item[0]))

    final_logged_step = None
    final_logged_value = None
    if metrics_by_step:
        final_logged_step = max(metrics_by_step)
        final_logged_value = metrics_by_step[final_logged_step]

    if explicit_step is not None:
        checkpoint_path = run_dir / "actor" / f"global_step_{explicit_step}" / "lunarlander_policy.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint step {explicit_step} not found at {checkpoint_path}")
        return CheckpointSelection(
            mode="explicit_step",
            checkpoint_step=explicit_step,
            checkpoint_path=checkpoint_path,
            saved_checkpoint_steps=saved_steps,
            best_logged_step=best_logged_step,
            best_logged_value=best_logged_value,
            final_logged_step=final_logged_step,
            final_logged_value=final_logged_value,
            selected_logged_value=metrics_by_step.get(explicit_step),
        )

    if mode == "latest_saved":
        step = max(saved_steps)
        return CheckpointSelection(
            mode=mode,
            checkpoint_step=step,
            checkpoint_path=run_dir / "actor" / f"global_step_{step}" / "lunarlander_policy.pt",
            saved_checkpoint_steps=saved_steps,
            best_logged_step=best_logged_step,
            best_logged_value=best_logged_value,
            final_logged_step=final_logged_step,
            final_logged_value=final_logged_value,
            selected_logged_value=metrics_by_step.get(step),
        )

    if mode != "best_saved":
        raise ValueError(f"Unsupported checkpoint selection mode: {mode}")

    eligible = [(step, metrics_by_step[step]) for step in saved_steps if step in metrics_by_step]
    if eligible:
        step, selected_value = max(eligible, key=lambda item: (item[1], item[0]))
        return CheckpointSelection(
            mode=mode,
            checkpoint_step=step,
            checkpoint_path=run_dir / "actor" / f"global_step_{step}" / "lunarlander_policy.pt",
            saved_checkpoint_steps=saved_steps,
            best_logged_step=best_logged_step,
            best_logged_value=best_logged_value,
            final_logged_step=final_logged_step,
            final_logged_value=final_logged_value,
            selected_logged_value=selected_value,
        )

    step = max(saved_steps)
    return CheckpointSelection(
        mode=mode,
        checkpoint_step=step,
        checkpoint_path=run_dir / "actor" / f"global_step_{step}" / "lunarlander_policy.pt",
        saved_checkpoint_steps=saved_steps,
        best_logged_step=best_logged_step,
        best_logged_value=best_logged_value,
        final_logged_step=final_logged_step,
        final_logged_value=final_logged_value,
        selected_logged_value=metrics_by_step.get(step),
        selection_warning=(
            f"No saved checkpoint step has an exact logged `{metric_key}` value. "
            "Falling back to the latest saved checkpoint."
        ),
    )


def _component_share(weighted_values: np.ndarray, denom: np.ndarray) -> float:
    valid = denom > 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(np.abs(weighted_values[valid]) / denom[valid]))


def _load_policy_from_checkpoint(checkpoint_path: Path) -> LunarLanderPolicy:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hidden_size = int(checkpoint.get("config", {}).get("model", {}).get("hidden_size", 128))
    activation = str(checkpoint.get("config", {}).get("model", {}).get("activation", "tanh"))
    policy = LunarLanderPolicy(hidden_size=hidden_size, activation=activation)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy


def evaluate_saved_checkpoint(run_dir: Path, selection: CheckpointSelection) -> dict[str, Any]:
    import gymnasium as gym

    run_dir = Path(run_dir)
    config = load_json(run_dir / "run_config_snapshot.json")
    summary = load_json(run_dir / "metrics_summary.json")
    eval_cfg = config.get("eval", {})
    rollout_cfg = config.get("actor_rollout_ref", {}).get("rollout", {})
    seeds = list(eval_cfg.get("seed_list", []))
    policy = _load_policy_from_checkpoint(selection.checkpoint_path)
    reward_config = LunarLanderRewardConfig.from_config(config)
    env_name = rollout_cfg.get("env_name", "LunarLander-v3")
    max_steps = int(rollout_cfg.get("max_steps", 400))

    episode_r_sub = []
    episode_r_prog = []
    episode_r_micro = []
    episode_r_smooth = []
    episode_r_final = []
    episode_total = []
    success_values = []
    terminated_values = []
    truncated_values = []
    crash_values = []
    episode_lengths = []
    action_switches = []
    phase_transitions = []
    fuel_proxies = []
    final_abs_x = []
    final_abs_vx = []
    final_abs_vy = []
    final_abs_theta = []
    approach_episodes = 0
    phase_counts = {phase.lower(): 0.0 for phase in PHASE_NAMES.values()}
    phase_episode_counts = {phase.lower(): 0.0 for phase in PHASE_NAMES.values()}
    micro_progress_counts = {name: 0.0 for name in MICRO_PROGRESS_KEYS}

    for seed in seeds:
        env = gym.make(env_name)
        obs, _info = env.reset(seed=seed)
        reward_state = LunarLanderRewardState()
        actions: list[int] = []
        phases: list[int] = []
        ep_r_sub = 0.0
        ep_r_prog = 0.0
        ep_r_micro = 0.0
        ep_r_smooth = 0.0
        ep_r_final = 0.0
        ep_total = 0.0
        success = False
        terminated = False
        truncated = False
        crash = False
        fuel_proxy = 0.0
        episode_env_return = 0.0

        for step in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = policy(obs_tensor)
            action = int(torch.argmax(logits, dim=-1).item())
            next_obs, env_reward, terminated_flag, truncated_flag, info = env.step(action)
            forced_truncated = bool(step == max_steps - 1 and not terminated_flag and not truncated_flag)
            step_terminated = bool(terminated_flag)
            step_truncated = bool(truncated_flag or forced_truncated)
            episode_env_return += float(env_reward)
            raw_phase = classify_phase(next_obs, reward_config)
            phase = stabilize_phase(raw_phase, reward_state.prev_phase)
            reward_dict = compute_step_reward(
                obs=next_obs,
                prev_obs=obs,
                action=action,
                prev_action=reward_state.prev_action,
                phase=phase,
                prev_phase=reward_state.prev_phase,
                terminated=step_terminated,
                truncated=step_truncated,
                info=info,
                config=reward_config,
                visited_phases=reward_state.visited_phases,
                visited_micro_progress=reward_state.visited_micro_progress,
                episode_return=episode_env_return,
            )
            actions.append(action)
            phases.append(phase)
            phase_counts[PHASE_NAMES[phase].lower()] += 1.0
            ep_r_sub += reward_dict["r_sub"]
            ep_r_prog += reward_dict["r_prog"]
            ep_r_micro += reward_dict["r_micro"]
            ep_r_smooth += reward_dict["r_smooth"]
            ep_r_final += reward_dict["r_final"]
            ep_total += reward_dict["r_total"]
            fuel_proxy += FUEL_PROXY_COST.get(action, 0.0)
            obs = next_obs
            reward_state.prev_action = action
            reward_state.prev_phase = phase
            reward_state.visited_phases.add(phase)
            reward_state.visited_micro_progress.update(reward_dict["micro_events"])
            success = bool(reward_dict["success"])
            terminated = step_terminated
            truncated = step_truncated
            if terminated or truncated:
                crash = bool(terminated and not success)
                break

        env.close()

        for phase_name, phase_id in ((name.lower(), phase_id) for phase_id, name in PHASE_NAMES.items()):
            if phase_id in reward_state.visited_phases:
                phase_episode_counts[phase_name] += 1.0
        if APPROACH in reward_state.visited_phases:
            approach_episodes += 1
        for event_name in reward_state.visited_micro_progress:
            micro_progress_counts[event_name] += 1.0

        final_obs = np.asarray(obs, dtype=np.float32)
        episode_r_sub.append(ep_r_sub)
        episode_r_prog.append(ep_r_prog)
        episode_r_micro.append(ep_r_micro)
        episode_r_smooth.append(ep_r_smooth)
        episode_r_final.append(ep_r_final)
        episode_total.append(ep_total)
        success_values.append(1.0 if success else 0.0)
        terminated_values.append(1.0 if terminated else 0.0)
        truncated_values.append(1.0 if truncated else 0.0)
        crash_values.append(1.0 if crash else 0.0)
        episode_lengths.append(float(len(actions)))
        action_switches.append(float(sum(1 for prev_action, cur_action in zip(actions[:-1], actions[1:]) if prev_action != cur_action)))
        phase_transitions.append(float(sum(1 for prev_phase, cur_phase in zip(phases[:-1], phases[1:]) if prev_phase != cur_phase)))
        fuel_proxies.append(float(fuel_proxy))
        final_abs_x.append(float(abs(final_obs[0])))
        final_abs_vx.append(float(abs(final_obs[2])))
        final_abs_vy.append(float(abs(final_obs[3])))
        final_abs_theta.append(float(abs(final_obs[4])))

    weighted_r_sub = reward_config.w_sub * np.asarray(episode_r_sub, dtype=np.float32)
    weighted_r_prog = reward_config.w_prog * np.asarray(episode_r_prog, dtype=np.float32)
    weighted_r_micro = reward_config.w_prog * np.asarray(episode_r_micro, dtype=np.float32)
    weighted_r_smooth = reward_config.w_smooth * np.asarray(episode_r_smooth, dtype=np.float32)
    weighted_r_final = reward_config.w_final * np.asarray(episode_r_final, dtype=np.float32)
    contribution_denom = (
        np.abs(weighted_r_sub)
        + np.abs(weighted_r_prog)
        + np.abs(weighted_r_micro)
        + np.abs(weighted_r_smooth)
        + np.abs(weighted_r_final)
    )

    return {
        "run_dir": str(run_dir),
        "selection": {
            "mode": selection.mode,
            "checkpoint_step": selection.checkpoint_step,
            "checkpoint_path": str(selection.checkpoint_path),
            "saved_checkpoint_steps": selection.saved_checkpoint_steps,
            "best_logged_step": selection.best_logged_step,
            "best_logged_value": selection.best_logged_value,
            "best_logged_step_is_saved": selection.best_logged_step in selection.saved_checkpoint_steps
            if selection.best_logged_step is not None
            else False,
            "final_logged_step": selection.final_logged_step,
            "final_logged_value": selection.final_logged_value,
            "selected_logged_value": selection.selected_logged_value,
            "selection_warning": selection.selection_warning,
        },
        "eval": {
            "env_name": env_name,
            "num_eval_episodes": len(seeds),
            "eval_seed_hash": hashlib.md5(json.dumps(seeds).encode("utf-8")).hexdigest()[:12],
            "eval_seed_list": seeds,
            "success_rate": float(np.mean(success_values)) if success_values else 0.0,
            "terminated_rate": float(np.mean(terminated_values)) if terminated_values else 0.0,
            "truncated_rate": float(np.mean(truncated_values)) if truncated_values else 0.0,
            "crash_rate": float(np.mean(crash_values)) if crash_values else 0.0,
            "mean_total_reward": float(np.mean(episode_total)) if episode_total else 0.0,
            "mean_r_sub": float(np.mean(episode_r_sub)) if episode_r_sub else 0.0,
            "mean_r_prog": float(np.mean(episode_r_prog)) if episode_r_prog else 0.0,
            "mean_r_micro": float(np.mean(episode_r_micro)) if episode_r_micro else 0.0,
            "mean_r_smooth": float(np.mean(episode_r_smooth)) if episode_r_smooth else 0.0,
            "mean_r_final": float(np.mean(episode_r_final)) if episode_r_final else 0.0,
            "mean_weighted_r_sub": float(np.mean(weighted_r_sub)) if weighted_r_sub.size else 0.0,
            "mean_weighted_r_prog": float(np.mean(weighted_r_prog)) if weighted_r_prog.size else 0.0,
            "mean_weighted_r_micro": float(np.mean(weighted_r_micro)) if weighted_r_micro.size else 0.0,
            "mean_weighted_r_smooth": float(np.mean(weighted_r_smooth)) if weighted_r_smooth.size else 0.0,
            "mean_weighted_r_final": float(np.mean(weighted_r_final)) if weighted_r_final.size else 0.0,
            "reward_shares": {
                "subgoal": _component_share(weighted_r_sub, contribution_denom),
                "progress": _component_share(weighted_r_prog, contribution_denom),
                "micro_progress": _component_share(weighted_r_micro, contribution_denom),
                "smoothness": _component_share(weighted_r_smooth, contribution_denom),
                "final": _component_share(weighted_r_final, contribution_denom),
            },
            "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "mean_num_action_switches": float(np.mean(action_switches)) if action_switches else 0.0,
            "mean_num_phase_transitions": float(np.mean(phase_transitions)) if phase_transitions else 0.0,
            "mean_fuel_proxy": float(np.mean(fuel_proxies)) if fuel_proxies else 0.0,
            "mean_abs_final_x": float(np.mean(final_abs_x)) if final_abs_x else 0.0,
            "mean_abs_final_vx": float(np.mean(final_abs_vx)) if final_abs_vx else 0.0,
            "mean_abs_final_vy": float(np.mean(final_abs_vy)) if final_abs_vy else 0.0,
            "mean_abs_final_theta": float(np.mean(final_abs_theta)) if final_abs_theta else 0.0,
            "phase_counts": phase_counts,
            "phase_episode_counts": phase_episode_counts,
            "episodes_with_approach": float(approach_episodes),
            "micro_progress_counts": micro_progress_counts,
        },
        "comparison_to_logged_summary": {
            "summary_final_val_success_rate": summary.get("final_val_success_rate"),
            "summary_best_val_success_rate": summary.get("best_val_success_rate"),
            "summary_final_val_total_reward": summary.get("final_val_total_reward"),
        },
    }


def evaluate_run_checkpoint(
    run_dir: Path,
    mode: str = "best_saved",
    metric_key: str = "val/test_reward/success_rate",
    explicit_step: int | None = None,
) -> dict[str, Any]:
    selection = select_checkpoint_for_evaluation(
        run_dir=Path(run_dir),
        mode=mode,
        metric_key=metric_key,
        explicit_step=explicit_step,
    )
    return evaluate_saved_checkpoint(run_dir=Path(run_dir), selection=selection)
