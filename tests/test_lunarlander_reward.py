import torch

from verl.utils.lunarlander_shaped_reward import (
    ALIGN,
    APPROACH,
    DESCEND,
    TOUCHDOWN,
    LunarLanderRewardConfig,
    classify_phase,
    compute_final_reward,
    compute_micro_progress_reward,
    compute_progress_reward,
    compute_smoothness_reward,
    compute_step_reward,
    phase_threshold_dict,
    stabilize_phase,
)


def test_phase_classification_maps_many_synthetic_states():
    config = LunarLanderRewardConfig()
    cases = [
        ([1.0, 1.2, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0], APPROACH),
        ([0.8, 0.9, 0.1, -0.2, 0.2, 0.0, 0.0, 0.0], APPROACH),
        ([0.6, 0.8, 0.0, -0.1, 0.3, 0.0, 0.0, 0.0], APPROACH),
        ([0.5, 1.0, -0.2, -0.1, -0.3, 0.0, 0.0, 0.0], APPROACH),
        ([0.36, 0.7, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0], APPROACH),
        ([0.3, 1.0, 0.0, -0.2, 0.4, 0.0, 0.0, 0.0], ALIGN),
        ([0.25, 0.9, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0], ALIGN),
        ([0.1, 1.1, 0.0, -0.01, 0.5, 0.0, 0.0, 0.0], ALIGN),
        ([0.05, 1.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0], ALIGN),
        ([0.34, 0.8, 0.0, -0.03, 0.14, 0.0, 0.0, 0.0], ALIGN),
        ([0.15, 0.7, 0.0, -0.1, 0.05, 0.0, 0.0, 0.0], DESCEND),
        ([0.1, 0.6, 0.1, -0.3, 0.1, 0.0, 0.0, 0.0], DESCEND),
        ([0.18, 0.5, -0.05, -0.2, -0.1, 0.0, 0.0, 0.0], DESCEND),
        ([0.0, 0.74, 0.0, -0.06, 0.0, 0.0, 0.0, 0.0], DESCEND),
        ([0.19, 0.4, 0.0, -0.15, 0.14, 0.0, 0.0, 0.0], DESCEND),
        ([0.1, 0.2, 0.0, -0.1, 0.05, 0.0, 0.0, 0.0], TOUCHDOWN),
        ([0.05, 0.25, 0.05, -0.1, 0.02, 0.0, 0.0, 0.0], TOUCHDOWN),
        ([0.0, 0.15, 0.0, -0.05, 0.01, 0.0, 0.0, 0.0], TOUCHDOWN),
        ([0.14, 0.29, 0.15, -0.18, 0.09, 0.0, 0.0, 0.0], TOUCHDOWN),
        ([0.05, 0.2, 0.0, -0.1, 0.02, 0.0, 1.0, 1.0], TOUCHDOWN),
        ([0.15, 0.35, 0.0, -0.2, 0.05, 0.0, 0.0, 0.0], DESCEND),
        ([0.2, 0.76, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0], ALIGN),
        ([0.16, 0.29, 0.0, -0.15, 0.05, 0.0, 0.0, 0.0], DESCEND),
        ([0.12, 0.28, 0.21, -0.1, 0.02, 0.0, 0.0, 0.0], DESCEND),
    ]
    for obs, expected_phase in cases:
        assert classify_phase(obs, config) == expected_phase


def test_phase_threshold_dict_exposes_all_phase_rules():
    config = LunarLanderRewardConfig()
    thresholds = phase_threshold_dict(config)
    assert set(thresholds.keys()) == {"APPROACH", "ALIGN", "DESCEND", "TOUCHDOWN"}
    assert thresholds["APPROACH"]["abs_x_gt"] == config.center_x_abs_for_align
    assert thresholds["DESCEND"]["vy_lte"] == config.vertical_speed_min_for_descend
    assert thresholds["TOUCHDOWN"]["abs_vx_lte"] == config.horizontal_speed_abs_for_touchdown


def test_progress_reward_fires_only_on_forward_transition():
    config = LunarLanderRewardConfig()
    assert compute_progress_reward(None, APPROACH, config) == 0.0
    assert compute_progress_reward(APPROACH, ALIGN, config) == config.enter_align
    assert compute_progress_reward(ALIGN, DESCEND, config) == config.enter_descend
    assert compute_progress_reward(DESCEND, TOUCHDOWN, config) == config.enter_touchdown
    assert compute_progress_reward(ALIGN, ALIGN, config) == 0.0
    assert compute_progress_reward(DESCEND, ALIGN, config) == 0.0
    assert compute_progress_reward(DESCEND, TOUCHDOWN, config, visited_phases={TOUCHDOWN}) == 0.0


def test_micro_progress_rewards_fire_once_per_corridor():
    config = LunarLanderRewardConfig(
        enter_x_corridor_070=0.04,
        enter_x_corridor_050=0.06,
        enter_x_corridor_035=0.08,
    )
    reward, events = compute_micro_progress_reward(
        [0.34, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.80, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        APPROACH,
        config,
    )
    assert reward == config.enter_x_corridor_070 + config.enter_x_corridor_050 + config.enter_x_corridor_035
    assert events == ["enter_x_corridor_070", "enter_x_corridor_050", "enter_x_corridor_035"]

    reward_again, events_again = compute_micro_progress_reward(
        [0.20, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.34, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        APPROACH,
        config,
        visited_micro_progress={"enter_x_corridor_070", "enter_x_corridor_050", "enter_x_corridor_035"},
    )
    assert reward_again == 0.0
    assert events_again == []

    no_cross_reward, no_cross_events = compute_micro_progress_reward(
        [0.20, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        APPROACH,
        config,
    )
    assert no_cross_reward == 0.0
    assert no_cross_events == []

    align_reward, align_events = compute_micro_progress_reward(
        [0.20, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.60, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ALIGN,
        config,
    )
    assert align_reward == 0.0
    assert align_events == []


def test_stabilize_phase_prevents_backward_regressions():
    assert stabilize_phase(APPROACH, None) == APPROACH
    assert stabilize_phase(DESCEND, ALIGN) == DESCEND
    assert stabilize_phase(ALIGN, DESCEND) == DESCEND


def test_smoothness_reward_prefers_repeated_action():
    config = LunarLanderRewardConfig()
    same_action = compute_smoothness_reward(2, 2, config)
    switched_action = compute_smoothness_reward(1, 2, config)
    assert same_action == 0.0
    assert switched_action < same_action
    assert switched_action == -2.0


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


def test_step_reward_adds_micro_progress_with_progress_weight():
    config = LunarLanderRewardConfig(
        enter_x_corridor_070=0.04,
        enter_x_corridor_050=0.06,
        enter_x_corridor_035=0.08,
    )
    prev_obs = [0.8, 1.0, 0.2, -0.1, 0.1, 0.0, 0.0, 0.0]
    obs = [0.34, 1.0, 0.1, -0.1, 0.05, 0.0, 0.0, 0.0]
    reward = compute_step_reward(
        obs=obs,
        prev_obs=prev_obs,
        action=0,
        prev_action=0,
        phase=APPROACH,
        prev_phase=APPROACH,
        terminated=False,
        truncated=False,
        info={},
        config=config,
        visited_micro_progress=set(),
    )
    assert reward["r_micro"] == config.enter_x_corridor_070 + config.enter_x_corridor_050 + config.enter_x_corridor_035
    assert reward["micro_events"] == ["enter_x_corridor_070", "enter_x_corridor_050", "enter_x_corridor_035"]


def test_step_reward_does_not_award_micro_progress_when_episode_starts_inside_corridor():
    config = LunarLanderRewardConfig(
        enter_x_corridor_070=0.04,
        enter_x_corridor_050=0.06,
        enter_x_corridor_035=0.08,
    )
    prev_obs = [0.20, 1.0, 0.4, -0.1, 0.05, 0.0, 0.0, 0.0]
    obs = [0.19, 1.0, 0.3, -0.1, 0.05, 0.0, 0.0, 0.0]
    reward = compute_step_reward(
        obs=obs,
        prev_obs=prev_obs,
        action=0,
        prev_action=0,
        phase=APPROACH,
        prev_phase=APPROACH,
        terminated=False,
        truncated=False,
        info={},
        config=config,
        visited_micro_progress=set(),
    )
    assert reward["r_micro"] == 0.0
    assert reward["micro_events"] == []


def test_subgoal_reward_uses_delta_improvement_within_phase():
    config = LunarLanderRewardConfig()
    prev_obs = [0.8, 1.0, 0.5, -0.2, 0.0, 0.0, 0.0, 0.0]
    obs = [0.4, 1.0, 0.1, -0.2, 0.0, 0.0, 0.0, 0.0]
    reward = compute_step_reward(
        obs=obs,
        prev_obs=prev_obs,
        action=0,
        prev_action=0,
        phase=APPROACH,
        prev_phase=APPROACH,
        terminated=False,
        truncated=False,
        info={},
        config=config,
    )
    assert reward["r_sub"] > 0.0


def test_subgoal_reward_is_zero_on_phase_change():
    config = LunarLanderRewardConfig()
    prev_obs = [0.34, 1.10, 0.0, -0.02, 0.40, 0.20, 0.0, 0.0]
    obs = [0.18, 0.70, 0.0, -0.12, 0.10, 0.05, 0.0, 0.0]
    reward = compute_step_reward(
        obs=obs,
        prev_obs=prev_obs,
        action=2,
        prev_action=0,
        phase=DESCEND,
        prev_phase=ALIGN,
        terminated=False,
        truncated=False,
        info={},
        config=config,
    )
    assert reward["r_sub"] == 0.0


def test_subgoal_reward_is_clipped_after_delta_computation():
    config = LunarLanderRewardConfig()
    prev_obs = [2.0, 1.0, 2.0, -0.2, 0.0, 0.0, 0.0, 0.0]
    obs = [0.0, 1.0, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0]
    reward = compute_step_reward(
        obs=obs,
        prev_obs=prev_obs,
        action=0,
        prev_action=0,
        phase=APPROACH,
        prev_phase=APPROACH,
        terminated=False,
        truncated=False,
        info={},
        config=config,
    )
    assert reward["r_sub"] == 1.0
