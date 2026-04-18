import torch

from verl.utils.lunarlander_shaped_reward import (
    ALIGN,
    APPROACH,
    DESCEND,
    TOUCHDOWN,
    LunarLanderRewardConfig,
    classify_phase,
    compute_final_reward,
    compute_progress_reward,
    compute_smoothness_reward,
    compute_step_reward,
)


def test_phase_classification_maps_synthetic_states():
    config = LunarLanderRewardConfig()
    assert classify_phase([1.0, 1.0, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0], config) == APPROACH
    assert classify_phase([0.1, 1.0, 0.0, -0.2, 0.4, 0.0, 0.0, 0.0], config) == ALIGN
    assert classify_phase([0.1, 0.8, 0.0, -0.2, 0.05, 0.0, 0.0, 0.0], config) == DESCEND
    assert classify_phase([0.05, 0.2, 0.0, -0.1, 0.02, 0.0, 1.0, 1.0], config) == TOUCHDOWN


def test_progress_reward_fires_only_on_forward_transition():
    config = LunarLanderRewardConfig()
    assert compute_progress_reward(None, APPROACH, config) == 0.0
    assert compute_progress_reward(APPROACH, ALIGN, config) == config.enter_align
    assert compute_progress_reward(ALIGN, ALIGN, config) == 0.0
    assert compute_progress_reward(DESCEND, ALIGN, config) == 0.0


def test_smoothness_reward_prefers_repeated_action():
    config = LunarLanderRewardConfig()
    same_action = compute_smoothness_reward(2, 2, config)
    switched_action = compute_smoothness_reward(1, 2, config)
    assert same_action == 0.0
    assert switched_action < same_action


def test_final_reward_uses_success_thresholds():
    config = LunarLanderRewardConfig()
    success_obs = [0.05, 0.2, 0.05, -0.05, 0.02, 0.0, 1.0, 1.0]
    failure_obs = [0.6, 0.2, 0.5, -0.8, 0.6, 0.0, 1.0, 1.0]
    assert compute_final_reward(success_obs, terminated=True, truncated=False, info={}, config=config) == 1.0
    assert compute_final_reward(failure_obs, terminated=True, truncated=False, info={}, config=config) == 0.0


def test_total_reward_composition_adds_terminal_term_only_at_episode_end():
    config = LunarLanderRewardConfig()
    obs = [0.05, 0.2, 0.05, -0.05, 0.02, 0.0, 1.0, 1.0]
    non_terminal = compute_step_reward(
        obs=obs,
        prev_obs=obs,
        action=2,
        prev_action=2,
        phase=TOUCHDOWN,
        prev_phase=DESCEND,
        terminated=False,
        truncated=False,
        info={},
        config=config,
    )
    terminal = compute_step_reward(
        obs=obs,
        prev_obs=obs,
        action=2,
        prev_action=2,
        phase=TOUCHDOWN,
        prev_phase=DESCEND,
        terminated=True,
        truncated=False,
        info={},
        config=config,
    )
    assert non_terminal["r_final"] == 0.0
    assert terminal["r_final"] == 1.0
    assert terminal["r_total"] > non_terminal["r_total"]
