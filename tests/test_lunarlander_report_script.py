import csv
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
                "reward": {
                    "weights": weights,
                    "phase_thresholds": {"center_x_abs_for_align": 0.35},
                    "success": {"max_abs_x": 0.2},
                },
                "eval": {"seed_list": [0, 1, 2]},
                "actor_rollout_ref": {"rollout": {"env_name": "LunarLander-v3"}},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "run_report.md").write_text("# Stub run report\n", encoding="utf-8")
    (run_dir / "metrics_history.csv").write_text("step,value\n0,1\n", encoding="utf-8")
    (run_dir / "metrics_history.jsonl").write_text('{"step": 0}\n', encoding="utf-8")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    (plots_dir / "reward_overview.png").write_text("", encoding="utf-8")
    (plots_dir / "task_metrics_train.png").write_text("", encoding="utf-8")
    (run_dir / "audit_traces_train.csv").write_text("step\n0\n", encoding="utf-8")
    trajectory_dir = run_dir / "trajectory_dumps"
    trajectory_dir.mkdir(exist_ok=True)
    (trajectory_dir / "success_seed_0_step_0.json").write_text("{}", encoding="utf-8")
    (trajectory_dir / "success_seed_0_step_0.png").write_text("", encoding="utf-8")


def test_report_script_generates_markdown_and_csv(tmp_path):
    run_terminal_only = tmp_path / "terminal_only"
    run_terminal_smooth = tmp_path / "terminal_smooth"
    run_terminal_sub_prog = tmp_path / "terminal_sub_prog"
    run_full = tmp_path / "full"

    _write_run(
        run_terminal_only,
        "terminal_only",
        {"sub": 0.0, "prog": 0.0, "smooth": 0.0, "final": 1.0},
        {
            "final_train_success_rate": None,
            "best_train_success_rate": None,
            "final_val_success_rate": 0.20,
            "best_val_success_rate": 0.35,
            "final_train_total_reward": None,
            "final_val_total_reward": 0.20,
            "final_reward_hacking_warning": None,
            "final_eval_num_seeds_configured": 3.0,
            "final_val_num_eval_episodes": 3.0,
            "final_eval_full_seed_coverage": 1.0,
            "final_eval_seed_hash": "seedhash",
            "final_train_phase_counts": {phase: None for phase in ("approach", "align", "descend", "touchdown")},
            "final_val_phase_counts": {"approach": 0.0, "align": 2.0, "descend": 1.0, "touchdown": 0.0},
            "final_train_phase_episode_counts": {phase: None for phase in ("approach", "align", "descend", "touchdown")},
            "final_val_phase_episode_counts": {"approach": 0.0, "align": 2.0, "descend": 1.0, "touchdown": 0.0},
            "final_train_num_phase_transitions": None,
            "final_val_num_phase_transitions": 0.5,
            "final_train_micro_progress_counts": {
                "enter_x_corridor_070": None,
                "enter_x_corridor_050": None,
                "enter_x_corridor_035": None,
            },
            "final_val_micro_progress_counts": {
                "enter_x_corridor_070": 0.0,
                "enter_x_corridor_050": 0.0,
                "enter_x_corridor_035": 0.0,
            },
            "final_train_reward_shares": {
                "subgoal": None,
                "progress": None,
                "micro_progress": None,
                "smoothness": None,
                "final": None,
            },
            "final_val_reward_shares": {
                "subgoal": 0.0,
                "progress": 0.0,
                "micro_progress": 0.0,
                "smoothness": 0.0,
                "final": 1.0,
            },
        },
    )
    _write_run(
        run_terminal_smooth,
        "terminal_smooth",
        {"sub": 0.0, "prog": 0.0, "smooth": 0.005, "final": 1.0},
        {
            "final_train_success_rate": 0.25,
            "best_train_success_rate": 0.38,
            "final_val_success_rate": 0.18,
            "best_val_success_rate": 0.18,
            "final_train_total_reward": 0.25,
            "final_val_total_reward": 0.18,
            "final_reward_hacking_warning": 1.0,
            "final_eval_num_seeds_configured": 3.0,
            "final_val_num_eval_episodes": 3.0,
            "final_eval_full_seed_coverage": 1.0,
            "final_eval_seed_hash": "seedhash",
            "final_train_phase_counts": {"approach": 0.0, "align": 8.0, "descend": 1.0, "touchdown": 0.0},
            "final_val_phase_counts": {"approach": 0.0, "align": 3.0, "descend": 1.0, "touchdown": 0.0},
            "final_train_phase_episode_counts": {"approach": 0.0, "align": 4.0, "descend": 1.0, "touchdown": 0.0},
            "final_val_phase_episode_counts": {"approach": 0.0, "align": 2.0, "descend": 1.0, "touchdown": 0.0},
            "final_train_num_phase_transitions": 1.0,
            "final_val_num_phase_transitions": 1.0,
            "final_train_micro_progress_counts": {
                "enter_x_corridor_070": 0.0,
                "enter_x_corridor_050": 0.0,
                "enter_x_corridor_035": 0.0,
            },
            "final_val_micro_progress_counts": {
                "enter_x_corridor_070": 0.0,
                "enter_x_corridor_050": 0.0,
                "enter_x_corridor_035": 0.0,
            },
            "final_train_reward_shares": {
                "subgoal": 0.0,
                "progress": 0.0,
                "micro_progress": 0.0,
                "smoothness": 0.0,
                "final": 0.25,
            },
            "final_val_reward_shares": {
                "subgoal": 0.0,
                "progress": 0.0,
                "micro_progress": 0.0,
                "smoothness": 0.0,
                "final": 0.18,
            },
        },
    )
    _write_run(
        run_terminal_sub_prog,
        "terminal_sub_prog",
        {"sub": 0.1, "prog": 0.3, "smooth": 0.0, "final": 1.0},
        {
            "final_train_success_rate": 0.29,
            "best_train_success_rate": 0.43,
            "final_val_success_rate": 0.28,
            "best_val_success_rate": 0.31,
            "final_train_total_reward": 0.40,
            "final_val_total_reward": 0.41,
            "final_reward_hacking_warning": 0.0,
            "final_eval_num_seeds_configured": 3.0,
            "final_val_num_eval_episodes": 3.0,
            "final_eval_full_seed_coverage": 1.0,
            "final_eval_seed_hash": "seedhash",
            "final_train_phase_counts": {"approach": 0.0, "align": 7.0, "descend": 2.0, "touchdown": 1.0},
            "final_val_phase_counts": {"approach": 0.0, "align": 4.0, "descend": 2.0, "touchdown": 1.0},
            "final_train_phase_episode_counts": {"approach": 0.0, "align": 4.0, "descend": 2.0, "touchdown": 1.0},
            "final_val_phase_episode_counts": {"approach": 0.0, "align": 3.0, "descend": 2.0, "touchdown": 1.0},
            "final_train_num_phase_transitions": 1.3,
            "final_val_num_phase_transitions": 1.2,
            "final_train_micro_progress_counts": {
                "enter_x_corridor_070": 0.0,
                "enter_x_corridor_050": 0.0,
                "enter_x_corridor_035": 0.0,
            },
            "final_val_micro_progress_counts": {
                "enter_x_corridor_070": 0.0,
                "enter_x_corridor_050": 0.0,
                "enter_x_corridor_035": 0.0,
            },
            "final_train_reward_shares": {
                "subgoal": 0.33,
                "progress": 0.44,
                "micro_progress": 0.0,
                "smoothness": 0.0,
                "final": 0.23,
            },
            "final_val_reward_shares": {
                "subgoal": 0.32,
                "progress": 0.45,
                "micro_progress": 0.0,
                "smoothness": 0.0,
                "final": 0.23,
            },
        },
    )
    _write_run(
        run_full,
        "full",
        {"sub": 0.1, "prog": 0.3, "smooth": 0.005, "final": 1.0},
        {
            "final_train_success_rate": 0.16,
            "best_train_success_rate": 0.41,
            "final_val_success_rate": 0.22,
            "best_val_success_rate": 0.22,
            "final_train_total_reward": 0.20,
            "final_val_total_reward": 0.27,
            "final_reward_hacking_warning": 0.0,
            "final_eval_num_seeds_configured": 3.0,
            "final_val_num_eval_episodes": 3.0,
            "final_eval_full_seed_coverage": 1.0,
            "final_eval_seed_hash": "seedhash",
            "final_train_phase_counts": {"approach": 0.0, "align": 8.0, "descend": 1.0, "touchdown": 0.0},
            "final_val_phase_counts": {"approach": 0.0, "align": 5.0, "descend": 1.0, "touchdown": 0.0},
            "final_train_phase_episode_counts": {"approach": 0.0, "align": 4.0, "descend": 1.0, "touchdown": 0.0},
            "final_val_phase_episode_counts": {"approach": 0.0, "align": 3.0, "descend": 1.0, "touchdown": 0.0},
            "final_train_num_phase_transitions": 1.0,
            "final_val_num_phase_transitions": 1.1,
            "final_train_micro_progress_counts": {
                "enter_x_corridor_070": 0.0,
                "enter_x_corridor_050": 0.0,
                "enter_x_corridor_035": 0.0,
            },
            "final_val_micro_progress_counts": {
                "enter_x_corridor_070": 0.0,
                "enter_x_corridor_050": 0.0,
                "enter_x_corridor_035": 0.0,
            },
            "final_train_reward_shares": {
                "subgoal": 0.41,
                "progress": 0.37,
                "micro_progress": 0.0,
                "smoothness": 0.09,
                "final": 0.13,
            },
            "final_val_reward_shares": {
                "subgoal": 0.39,
                "progress": 0.40,
                "micro_progress": 0.0,
                "smoothness": 0.02,
                "final": 0.18,
            },
        },
    )

    output_path = tmp_path / "report.md"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "examples" / "generate_lunarlander_report.py"),
            str(run_terminal_only),
            str(run_terminal_smooth),
            str(run_terminal_sub_prog),
            str(run_full),
            "--output",
            str(output_path),
        ],
        check=True,
    )

    report = output_path.read_text(encoding="utf-8")
    csv_path = output_path.with_suffix(".csv")
    assert csv_path.exists()

    assert "## Executive Summary" in report
    assert "## Main Ablation Comparison" in report
    assert "## Phase and Curriculum" in report
    assert "## Reward Composition" in report
    assert "## Interpretation" in report
    assert "## Artifact Appendix" in report
    assert "phase-local shaping" in report
    assert "misalignment warning" in report
    assert "Any run with nonzero APPROACH usage: `False`" in report
    assert "Any run with nonzero micro-progress usage: `False`" in report
    assert "`full` trails `terminal_sub_prog` on final train success" in report
    assert "`full` trails `terminal_sub_prog` on final validation success" in report
    assert "train APPROACH=0.0000, val APPROACH=0.0000" in report
    assert "micro-progress total=0.0000/0.0000" in report
    assert "terminal_only" in report
    assert "n/a" in report

    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))

    assert len(rows) == 4
    rows_by_run = {row["run"]: row for row in rows}
    assert rows_by_run["terminal_only"]["final_train_success_rate"] == ""
    assert rows_by_run["terminal_only"]["final_train_phase_counts_approach"] == ""
    assert rows_by_run["terminal_only"]["final_val_reward_shares_final"] == "1.0"
    assert rows_by_run["full"]["final_train_phase_counts_approach"] == "0.0"
    assert rows_by_run["full"]["final_train_micro_progress_counts_enter_x_corridor_050"] == "0.0"
    assert rows_by_run["full"]["final_train_reward_shares_micro_progress"] == "0.0"
    assert rows_by_run["terminal_smooth"]["final_reward_hacking_warning"] == "1.0"
