import json
from pathlib import Path

from verl.utils.lunarlander_logging import LunarLanderArtifactLogger


def test_artifact_logger_writes_metrics_and_report(tmp_path):
    logger = LunarLanderArtifactLogger(
        output_dir=tmp_path,
        config={
            "trainer": {"experiment_name": "unit_test_run"},
            "data": {"train_batch_size": 4, "val_batch_size": 2, "n_samples": 2},
            "reward": {
                "weights": {"sub": 0.1, "prog": 0.3, "smooth": 0.005, "final": 1.0},
                "phase_thresholds": {"center_x_abs_for_align": 0.35},
                "success": {"max_abs_x": 0.2},
            },
            "eval": {"seed_list": [0, 1]},
            "actor_rollout_ref": {
                "actor": {"ppo_epochs": 2, "optim": {"lr": 3e-4}},
                "rollout": {"env_name": "LunarLander-v3"},
            },
        },
        reset_existing=True,
    )

    logger.log_metrics(
        step=0,
        data={
            "train_reward/mean_total_reward": 0.5,
            "train_reward/success_rate": 0.25,
            "train_reward/mean_fuel_proxy": 10.0,
            "val/test_reward/mean_total_reward": 0.4,
            "val/test_reward/success_rate": 0.20,
            "val/test_reward/mean_fuel_proxy": 9.0,
            "val/test_meta/num_eval_episodes": 2,
            "val/test_meta/num_eval_seeds_configured": 2,
            "val/test_meta/eval_full_seed_coverage": 1.0,
            "val/test_meta/eval_seed_hash": "abc123",
            "val/test_meta/eval_seed_list": "[0, 1]",
            "val/test_meta/deterministic_eval": 1.0,
            "train_reward/phase_counts/approach": 4.0,
            "train_reward/phase_counts/align": 8.0,
            "train_reward/phase_counts/descend": 6.0,
            "train_reward/phase_counts/touchdown": 2.0,
            "train_reward/phase_episode_counts/approach": 1.0,
            "train_reward/phase_episode_counts/align": 2.0,
            "train_reward/phase_episode_counts/descend": 1.0,
            "train_reward/phase_episode_counts/touchdown": 1.0,
            "val/test_reward/phase_counts/approach": 1.0,
            "val/test_reward/phase_counts/align": 2.0,
            "val/test_reward/phase_counts/descend": 3.0,
            "val/test_reward/phase_counts/touchdown": 4.0,
            "val/test_reward/phase_episode_counts/approach": 1.0,
            "val/test_reward/phase_episode_counts/align": 2.0,
            "val/test_reward/phase_episode_counts/descend": 2.0,
            "val/test_reward/phase_episode_counts/touchdown": 2.0,
            "train_reward/episodes_with_micro_progress/enter_x_corridor_070": 4.0,
            "train_reward/episodes_with_micro_progress/enter_x_corridor_050": 3.0,
            "train_reward/episodes_with_micro_progress/enter_x_corridor_035": 2.0,
            "val/test_reward/episodes_with_micro_progress/enter_x_corridor_070": 3.0,
            "val/test_reward/episodes_with_micro_progress/enter_x_corridor_050": 2.0,
            "val/test_reward/episodes_with_micro_progress/enter_x_corridor_035": 1.0,
            "train_reward/share_abs_weighted_r_micro": 0.12,
            "val/test_reward/share_abs_weighted_r_micro": 0.11,
            "diagnostics/reward_hacking_warning": 0.0,
        },
    )

    assert (tmp_path / "metrics_history.jsonl").exists()
    assert (tmp_path / "metrics_history.csv").exists()
    assert (tmp_path / "metrics_summary.json").exists()
    assert (tmp_path / "run_report.md").exists()
    assert (tmp_path / "run_config_snapshot.json").exists()

    summary = json.loads((tmp_path / "metrics_summary.json").read_text(encoding="utf-8"))
    assert summary["final_train_success_rate"] == 0.25
    assert summary["final_val_fuel_proxy"] == 9.0
    assert summary["final_eval_full_seed_coverage"] == 1.0
    assert summary["final_val_phase_counts"]["touchdown"] == 4.0
    assert summary["final_train_micro_progress_counts"]["enter_x_corridor_070"] == 4.0
    assert summary["final_train_micro_progress_counts"]["enter_x_corridor_050"] == 3.0
    assert summary["final_train_reward_shares"]["micro_progress"] == 0.12

    report = (tmp_path / "run_report.md").read_text(encoding="utf-8")
    assert "LunarLander Run Report" in report
    assert "Final validation fuel proxy" in report
    assert "Validation seed coverage" in report
    assert "Final train micro-progress counts" in report
    assert "LunarLander-v3" in report


def test_artifact_logger_generates_terminal_only_report_with_fallback_metrics(tmp_path):
    logger = LunarLanderArtifactLogger(
        output_dir=tmp_path,
        config={
            "trainer": {"experiment_name": "terminal_only_run"},
            "data": {"train_batch_size": 4, "val_batch_size": 2, "n_samples": 2},
            "reward": {
                "mode": "terminal_only",
                "weights": {"sub": 0.0, "prog": 0.0, "smooth": 0.0, "final": 1.0},
                "phase_thresholds": {"center_x_abs_for_align": 0.35},
                "success": {"max_abs_x": 0.2},
            },
            "eval": {"seed_list": [0, 1]},
            "actor_rollout_ref": {
                "actor": {"ppo_epochs": 2, "optim": {"lr": 3e-4}},
                "rollout": {"env_name": "LunarLander-v3"},
            },
        },
        reset_existing=True,
    )

    logger.log_metrics(
        step=3,
        data={
            "train_verify_score/all": 0.375,
            "critic/reward_components/mean_total_reward": 0.375,
            "critic/task/mean_fuel_proxy": 12.5,
            "val/test_score/all": 0.25,
            "val/test_reward/mean_total_reward": 0.25,
            "val/test_reward/mean_fuel_proxy": 10.0,
            "val/test_reward/phase_counts/approach": 0.0,
            "val/test_reward/phase_counts/align": 4.0,
            "val/test_reward/phase_counts/descend": 1.0,
            "val/test_reward/phase_counts/touchdown": 0.0,
            "val/test_reward/phase_episode_counts/approach": 0.0,
            "val/test_reward/phase_episode_counts/align": 2.0,
            "val/test_reward/phase_episode_counts/descend": 1.0,
            "val/test_reward/phase_episode_counts/touchdown": 0.0,
            "val/test_meta/num_eval_episodes": 2,
            "val/test_meta/num_eval_seeds_configured": 2,
            "val/test_meta/eval_full_seed_coverage": 1.0,
            "val/test_meta/eval_seed_hash": "abc123",
            "val/test_meta/eval_seed_list": "[0, 1]",
            "val/test_meta/deterministic_eval": 1.0,
        },
    )

    summary = json.loads((tmp_path / "metrics_summary.json").read_text(encoding="utf-8"))
    assert summary["reward_mode"] == "terminal_only"
    assert summary["final_train_success_rate"] == 0.375
    assert summary["best_train_success_rate"] == 0.375
    assert summary["final_train_total_reward"] == 0.375
    assert summary["final_train_fuel_proxy"] == 12.5
    assert summary["final_val_success_rate"] == 0.25
    assert summary["best_val_success_rate"] == 0.25

    report = (tmp_path / "run_report.md").read_text(encoding="utf-8")
    assert "LunarLander Terminal-Only Run Report" in report
    assert "Reward mode: `terminal_only`" in report
    assert "Terminal-Only Results" in report
    assert "Final train success rate: `0.375`" in report
    assert "Final validation success rate: `0.25`" in report
