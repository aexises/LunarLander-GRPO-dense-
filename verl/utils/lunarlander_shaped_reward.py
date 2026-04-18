"""Reward shaping utilities for the LunarLander sanity-check benchmark.

This module intentionally treats LunarLander as a controlled engineering testbed,
not as a vision-language-action benchmark. The phase-based reward decomposition is
only meant to validate per-step reward plumbing before transferring the pattern to
more VLA-like environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


APPROACH = 0
ALIGN = 1
DESCEND = 2
TOUCHDOWN = 3

PHASE_NAMES = {
    APPROACH: "APPROACH",
    ALIGN: "ALIGN",
    DESCEND: "DESCEND",
    TOUCHDOWN: "TOUCHDOWN",
}

ACTION_TO_VECTOR = {
    0: np.array([0.0, 0.0], dtype=np.float32),
    1: np.array([-1.0, 0.0], dtype=np.float32),
    2: np.array([0.0, 1.0], dtype=np.float32),
    3: np.array([1.0, 0.0], dtype=np.float32),
}


@dataclass
class LunarLanderRewardState:
    prev_phase: int | None = None
    prev_action: int | None = None


@dataclass
class LunarLanderRewardConfig:
    w_sub: float = 0.10
    w_prog: float = 0.30
    w_smooth: float = 0.02
    w_final: float = 1.00
    center_x_abs_for_align: float = 0.20
    angle_abs_for_descend: float = 0.15
    height_for_touchdown: float = 0.30
    vertical_speed_abs_for_touchdown: float = 0.20
    descent_target_height: float = 0.50
    approach_x: float = 1.0
    approach_vx: float = 0.3
    align_x: float = 0.7
    align_theta: float = 1.0
    align_omega: float = 0.3
    descend_y: float = 0.5
    descend_vy: float = 1.0
    descend_x: float = 0.3
    touchdown_vx: float = 0.7
    touchdown_vy: float = 1.0
    touchdown_theta: float = 0.7
    touchdown_omega: float = 0.3
    enter_align: float = 0.25
    enter_descend: float = 0.25
    enter_touchdown: float = 0.25
    smoothness_mode: str = "vector_l2"
    switch_penalty: float = 0.10
    subgoal_min: float = -1.0
    subgoal_max: float = 0.0
    max_abs_x: float = 0.20
    max_abs_vx: float = 0.20
    max_abs_vy: float = 0.20
    max_abs_theta: float = 0.15

    @classmethod
    def from_config(cls, config: Any) -> "LunarLanderRewardConfig":
        reward_cfg = config.get("reward", config)
        weights = reward_cfg.get("weights", {})
        thresholds = reward_cfg.get("phase_thresholds", {})
        coeffs = reward_cfg.get("coeffs", {})
        progress = reward_cfg.get("progress_rewards", {})
        smoothness = reward_cfg.get("smoothness", {})
        success = reward_cfg.get("success", {})
        return cls(
            w_sub=float(weights.get("sub", 0.10)),
            w_prog=float(weights.get("prog", 0.30)),
            w_smooth=float(weights.get("smooth", 0.02)),
            w_final=float(weights.get("final", 1.00)),
            center_x_abs_for_align=float(thresholds.get("center_x_abs_for_align", 0.20)),
            angle_abs_for_descend=float(thresholds.get("angle_abs_for_descend", 0.15)),
            height_for_touchdown=float(thresholds.get("height_for_touchdown", 0.30)),
            vertical_speed_abs_for_touchdown=float(thresholds.get("vertical_speed_abs_for_touchdown", 0.20)),
            descent_target_height=float(thresholds.get("descent_target_height", 0.50)),
            approach_x=float(coeffs.get("approach_x", 1.0)),
            approach_vx=float(coeffs.get("approach_vx", 0.3)),
            align_x=float(coeffs.get("align_x", 0.7)),
            align_theta=float(coeffs.get("align_theta", 1.0)),
            align_omega=float(coeffs.get("align_omega", 0.3)),
            descend_y=float(coeffs.get("descend_y", 0.5)),
            descend_vy=float(coeffs.get("descend_vy", 1.0)),
            descend_x=float(coeffs.get("descend_x", 0.3)),
            touchdown_vx=float(coeffs.get("touchdown_vx", 0.7)),
            touchdown_vy=float(coeffs.get("touchdown_vy", 1.0)),
            touchdown_theta=float(coeffs.get("touchdown_theta", 0.7)),
            touchdown_omega=float(coeffs.get("touchdown_omega", 0.3)),
            enter_align=float(progress.get("enter_align", 0.25)),
            enter_descend=float(progress.get("enter_descend", 0.25)),
            enter_touchdown=float(progress.get("enter_touchdown", 0.25)),
            smoothness_mode=str(smoothness.get("mode", "vector_l2")),
            switch_penalty=float(smoothness.get("switch_penalty", 0.10)),
            subgoal_min=float(reward_cfg.get("subgoal_min", -1.0)),
            subgoal_max=float(reward_cfg.get("subgoal_max", 0.0)),
            max_abs_x=float(success.get("max_abs_x", 0.20)),
            max_abs_vx=float(success.get("max_abs_vx", 0.20)),
            max_abs_vy=float(success.get("max_abs_vy", 0.20)),
            max_abs_theta=float(success.get("max_abs_theta", 0.15)),
        )


def _as_obs_array(obs: Any) -> np.ndarray:
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32)
    return np.asarray(obs, dtype=np.float32)


def _parse_obs(obs: Any) -> tuple[float, float, float, float, float, float, float, float]:
    obs_arr = _as_obs_array(obs)
    if obs_arr.shape[-1] < 8:
        raise ValueError(f"Expected LunarLander observation with at least 8 values, got shape {obs_arr.shape}")
    x, y, vx, vy, theta, omega, left_leg, right_leg = obs_arr[:8]
    return float(x), float(y), float(vx), float(vy), float(theta), float(omega), float(left_leg), float(right_leg)


def classify_phase(obs, config: LunarLanderRewardConfig) -> int:
    x, y, _vx, vy, theta, _omega, left_leg, right_leg = _parse_obs(obs)
    both_legs = left_leg > 0.5 and right_leg > 0.5
    if (
        both_legs
        or y <= config.height_for_touchdown
        or (
            abs(x) <= config.center_x_abs_for_align
            and abs(theta) <= config.angle_abs_for_descend
            and abs(vy) <= config.vertical_speed_abs_for_touchdown
        )
    ):
        return TOUCHDOWN
    if abs(x) <= config.center_x_abs_for_align and abs(theta) <= config.angle_abs_for_descend:
        return DESCEND
    if abs(x) <= config.center_x_abs_for_align:
        return ALIGN
    return APPROACH


def compute_subgoal_reward(obs, phase: int, config: LunarLanderRewardConfig) -> float:
    x, y, vx, vy, theta, omega, _left_leg, _right_leg = _parse_obs(obs)
    if phase == APPROACH:
        raw_reward = -(config.approach_x * abs(x) + config.approach_vx * abs(vx))
    elif phase == ALIGN:
        raw_reward = -(config.align_x * abs(x) + config.align_theta * abs(theta) + config.align_omega * abs(omega))
    elif phase == DESCEND:
        raw_reward = -(config.descend_y * abs(y - config.descent_target_height) + config.descend_vy * abs(vy) + config.descend_x * abs(x))
    elif phase == TOUCHDOWN:
        raw_reward = -(config.touchdown_vx * abs(vx) + config.touchdown_vy * abs(vy) + config.touchdown_theta * abs(theta) + config.touchdown_omega * abs(omega))
    else:
        raise ValueError(f"Unknown phase: {phase}")
    return float(np.clip(raw_reward, config.subgoal_min, config.subgoal_max))


def compute_progress_reward(prev_phase: int | None, phase: int, config: LunarLanderRewardConfig) -> float:
    if prev_phase is None or phase <= prev_phase:
        return 0.0
    progress_rewards = {
        ALIGN: config.enter_align,
        DESCEND: config.enter_descend,
        TOUCHDOWN: config.enter_touchdown,
    }
    return float(progress_rewards.get(phase, 0.0))


def compute_smoothness_reward(prev_action: int | None, action: int, config: LunarLanderRewardConfig) -> float:
    if prev_action is None:
        return 0.0
    if config.smoothness_mode == "switch_penalty":
        return 0.0 if prev_action == action else -float(config.switch_penalty)
    if config.smoothness_mode != "vector_l2":
        raise NotImplementedError(f"Unsupported smoothness mode: {config.smoothness_mode}")
    prev_vec = ACTION_TO_VECTOR[int(prev_action)]
    cur_vec = ACTION_TO_VECTOR[int(action)]
    return -float(np.square(cur_vec - prev_vec).sum())


def _infer_success(obs, info: dict | None, config: LunarLanderRewardConfig) -> bool:
    if info is not None:
        for key in ("is_success", "success", "landed"):
            if key in info:
                return bool(info[key])
    x, _y, vx, vy, theta, _omega, left_leg, right_leg = _parse_obs(obs)
    both_legs = left_leg > 0.5 and right_leg > 0.5
    return bool(
        both_legs
        and abs(x) <= config.max_abs_x
        and abs(vx) <= config.max_abs_vx
        and abs(vy) <= config.max_abs_vy
        and abs(theta) <= config.max_abs_theta
    )


def compute_final_reward(obs, terminated: bool, truncated: bool, info: dict | None, config: LunarLanderRewardConfig) -> float:
    if not (terminated or truncated):
        return 0.0
    return 1.0 if _infer_success(obs, info, config) else 0.0


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
    reward_config = config if isinstance(config, LunarLanderRewardConfig) else LunarLanderRewardConfig.from_config(config)
    r_sub = compute_subgoal_reward(obs, phase, reward_config)
    r_prog = compute_progress_reward(prev_phase, phase, reward_config)
    r_smooth = compute_smoothness_reward(prev_action, action, reward_config)
    r_final = compute_final_reward(obs, terminated, truncated, info, reward_config)
    r_total = reward_config.w_sub * r_sub + reward_config.w_prog * r_prog + reward_config.w_smooth * r_smooth
    if terminated or truncated:
        r_total += reward_config.w_final * r_final
    success = bool(r_final > 0.0)
    return {
        "r_sub": float(r_sub),
        "r_prog": float(r_prog),
        "r_smooth": float(r_smooth),
        "r_final": float(r_final),
        "r_total": float(r_total),
        "phase": int(phase),
        "success": success,
    }
