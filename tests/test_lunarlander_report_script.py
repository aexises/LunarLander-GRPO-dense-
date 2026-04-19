import json
import subprocess
import sys
from pathlib import Path


def _write_run(run_dir: Path, name: str, weights: dict, summary: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config_snapshot.json").write_text(
        json.dumps(
            {
                "trainer": {"experiment_name": name},
                "reward": {"weights": weights, "phase_thresholds": {}, "success": {}},
                "eval": {"seed_list": [0, 1, 2]},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_report_script_aggregates_runs(tmp_path):
    run_a = tmp_path / "terminal_only"
    run_b = tmp_path / "full"
    _write_run(
        run_a,
        "terminal_only",
        {"sub": 0.0, "prog": 0.0, "smooth": 0.0, "final": 1.0},
        {
            "final_val_success_rate": 0.20,
            "best_val_success_rate": 0.25,
            "final_val_total_reward": 0.20,
            "final_val_fuel_proxy": 8.0,
            "final_reward_hacking_warning": 0.0,
        },
    )
    _write_run(
        run_b,
        "full",
        {"sub": 0.15, "prog": 0.45, "smooth": 0.002, "final": 1.0},
        {
            "final_val_success_rate": 0.35,
            "best_val_success_rate": 0.40,
            "final_val_total_reward": 0.75,
            "final_val_fuel_proxy": 9.5,
            "final_reward_hacking_warning": 1.0,
        },
    )

    output_path = tmp_path / "report.md"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "examples" / "generate_lunarlander_report.py"),
            str(run_a),
            str(run_b),
            "--output",
            str(output_path),
        ],
        check=True,
    )

    report = output_path.read_text(encoding="utf-8")
    assert "LunarLander Reward Sanity Check Report" in report
    assert "phase-local shaping" in report
    assert "terminal_only" in report
    assert "full" in report
    assert "misalignment warning" in report
