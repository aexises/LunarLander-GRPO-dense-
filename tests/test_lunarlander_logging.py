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
                "weights": {"sub": 0.1, "prog": 0.3, "smooth": 0.02, "final": 1.0},
                "phase_thresholds": {"center_x_abs_for_align": 0.2},
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

    report = (tmp_path / "run_report.md").read_text(encoding="utf-8")
    assert "LunarLander Run Report" in report
    assert "Final validation fuel proxy" in report
    assert "LunarLander-v3" in report
