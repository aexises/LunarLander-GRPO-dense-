import csv
import sys
from types import SimpleNamespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verl.utils.lunarlander_checkpoint_eval import _load_policy_from_checkpoint, select_checkpoint_for_evaluation
from verl.workers.lunarlander_workers import LunarLanderActorRolloutRefWorker, LunarLanderPolicy


def _write_metrics_history(run_dir: Path, rows: list[dict[str, object]]):
    history_path = run_dir / "metrics_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["step", "val/test_reward/success_rate"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _touch_checkpoint(run_dir: Path, step: int):
    checkpoint_dir = run_dir / "actor" / f"global_step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "lunarlander_policy.pt").write_bytes(b"stub")


def test_select_checkpoint_prefers_best_saved_step_over_unsaved_peak(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_metrics_history(
        run_dir,
        [
            {"step": 89, "val/test_reward/success_rate": 0.40},
            {"step": 149, "val/test_reward/success_rate": 0.31},
            {"step": 160, "val/test_reward/success_rate": 0.28},
        ],
    )
    _touch_checkpoint(run_dir, 49)
    _touch_checkpoint(run_dir, 149)

    selection = select_checkpoint_for_evaluation(run_dir, mode="best_saved")

    assert selection.checkpoint_step == 149
    assert selection.best_logged_step == 89
    assert selection.best_logged_value == 0.40
    assert selection.final_logged_step == 160
    assert selection.final_logged_value == 0.28
    assert selection.selected_logged_value == 0.31
    assert selection.selection_warning is None


def test_select_checkpoint_can_choose_latest_saved(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_metrics_history(
        run_dir,
        [
            {"step": 49, "val/test_reward/success_rate": 0.20},
            {"step": 149, "val/test_reward/success_rate": 0.31},
        ],
    )
    _touch_checkpoint(run_dir, 49)
    _touch_checkpoint(run_dir, 149)

    selection = select_checkpoint_for_evaluation(run_dir, mode="latest_saved")

    assert selection.checkpoint_step == 149
    assert selection.selected_logged_value == 0.31
    assert selection.selection_warning is None


def test_select_checkpoint_falls_back_to_latest_when_no_saved_step_has_metric(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_metrics_history(
        run_dir,
        [
            {"step": 89, "val/test_reward/success_rate": 0.40},
            {"step": 160, "val/test_reward/success_rate": 0.28},
        ],
    )
    _touch_checkpoint(run_dir, 24)
    _touch_checkpoint(run_dir, 149)

    selection = select_checkpoint_for_evaluation(run_dir, mode="best_saved")

    assert selection.checkpoint_step == 149
    assert selection.selected_logged_value is None
    assert selection.selection_warning is not None
    assert "Falling back to the latest saved checkpoint" in selection.selection_warning


def test_load_policy_from_checkpoint_restores_saved_activation(tmp_path):
    policy = LunarLanderPolicy(hidden_size=16, activation="relu")
    checkpoint_path = tmp_path / "policy.pt"
    torch.save(
        {
            "config": {"model": {"hidden_size": 16, "activation": "relu"}},
            "policy_state_dict": policy.state_dict(),
        },
        checkpoint_path,
    )

    restored = _load_policy_from_checkpoint(checkpoint_path)

    assert isinstance(restored.net[1], torch.nn.ReLU)
    for key, value in policy.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_worker_can_load_reference_style_actor_checkpoint(tmp_path):
    source_policy = LunarLanderPolicy(hidden_size=16, activation="relu")
    checkpoint_path = tmp_path / "reference_actor.pt"
    torch.save(
        {f"actor.{key[len('net.'):]}": value.clone() for key, value in source_policy.state_dict().items()},
        checkpoint_path,
    )

    worker = LunarLanderActorRolloutRefWorker.__new__(LunarLanderActorRolloutRefWorker)
    worker.device = torch.device("cpu")
    worker.config = SimpleNamespace(model={"pretrained_policy_path": str(checkpoint_path)})
    worker.policy = LunarLanderPolicy(hidden_size=16, activation="relu")

    worker._load_pretrained_policy_if_configured()

    for key, value in source_policy.state_dict().items():
        assert torch.equal(value, worker.policy.state_dict()[key])
