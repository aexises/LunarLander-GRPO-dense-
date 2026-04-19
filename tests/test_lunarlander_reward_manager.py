import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.main_ppo import RobRewardManager
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_advantage


def _base_config():
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"action_token_len": 1, "action_chunks_len": 1},
            },
            "verifier": {"reward_coef": 1.0},
            "reward": {"mode": "terminal_only", "distribution_mode": "last_token_per_step"},
            "algorithm": {"gamma": 1.0, "lam": 1.0, "adv_estimator": "grpo"},
            "data": {"n_samples": 2},
        }
    )


def test_legacy_mode_keeps_final_token_reward_placement():
    config = _base_config()
    manager = RobRewardManager(num_examine=0, config=config)
    batch = {
        "responses": torch.zeros((1, 4, 1), dtype=torch.long),
        "finish_step": torch.tensor([3], dtype=torch.long),
        "complete": torch.tensor([True]),
    }
    proto = DataProto.from_dict(tensors=batch)
    reward_tensors, _metrics = manager(proto)
    assert reward_tensors["all"].shape == (1, 4)
    assert reward_tensors["all"][0, 2].item() == 1.0
    assert reward_tensors["all"][0, :2].sum().item() == 0.0
    assert reward_tensors["all"][0, 3].item() == 0.0


def test_shaped_mode_maps_per_step_rewards_to_step_token_ends():
    config = _base_config()
    config.reward.mode = "lunarlander_shaped"
    manager = RobRewardManager(num_examine=0, config=config)
    proto = DataProto.from_dict(
        tensors={
            "responses": torch.zeros((1, 4, 1), dtype=torch.long),
            "finish_step": torch.tensor([3], dtype=torch.long),
            "complete": torch.tensor([True]),
            "success": torch.tensor([True]),
            "step_token_ends": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            "r_sub": torch.tensor([[0.1, 0.2, 0.3, 0.0]], dtype=torch.float32),
            "r_prog": torch.tensor([[0.0, 0.25, 0.0, 0.0]], dtype=torch.float32),
            "r_micro": torch.tensor([[0.0, 0.05, 0.0, 0.0]], dtype=torch.float32),
            "r_smooth": torch.tensor([[0.0, -0.1, -0.2, 0.0]], dtype=torch.float32),
            "r_final": torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
            "r_total": torch.tensor([[0.1, 0.40, 1.1, 0.0]], dtype=torch.float32),
            "crash": torch.tensor([False]),
            "episode_length": torch.tensor([3]),
            "num_action_switches": torch.tensor([1]),
            "final_x": torch.tensor([0.0]),
            "final_vx": torch.tensor([0.0]),
            "final_vy": torch.tensor([0.0]),
            "final_theta": torch.tensor([0.0]),
        }
    )
    reward_tensors, metrics = manager(proto)
    assert reward_tensors["all"].shape == (1, 4)
    assert torch.allclose(reward_tensors["all"][0, :3], torch.tensor([0.1, 0.40, 1.1]))
    assert reward_tensors["gt_scores"][0, 2].item() == 1.0
    assert metrics["success_rate"] == 1.0
    assert metrics["terminated_rate"] == 1.0


def test_shaped_reward_smoke_path_keeps_advantage_shapes():
    config = _base_config()
    config.reward.mode = "lunarlander_shaped"
    manager = RobRewardManager(num_examine=0, config=config)
    proto = DataProto.from_dict(
        tensors={
            "responses": torch.zeros((2, 4, 1), dtype=torch.long),
            "finish_step": torch.tensor([4, 4], dtype=torch.long),
            "complete": torch.tensor([True, False]),
            "success": torch.tensor([True, False]),
            "step_token_ends": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
            "r_sub": torch.tensor([[0.1, 0.2, 0.3, 0.1], [0.0, 0.1, 0.0, 0.1]], dtype=torch.float32),
            "r_prog": torch.tensor([[0.0, 0.25, 0.0, 0.0], [0.0, 0.25, 0.0, 0.0]], dtype=torch.float32),
            "r_micro": torch.tensor([[0.0, 0.05, 0.0, 0.0], [0.0, 0.0, 0.08, 0.0]], dtype=torch.float32),
            "r_smooth": torch.tensor([[0.0, -0.1, -0.2, 0.0], [0.0, -0.1, 0.0, -0.1]], dtype=torch.float32),
            "r_final": torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            "r_total": torch.tensor([[0.1, 0.40, 0.1, 1.0], [0.0, 0.25, 0.08, 0.0]], dtype=torch.float32),
            "old_log_probs": torch.zeros((2, 4), dtype=torch.float32),
            "crash": torch.tensor([False, True]),
            "episode_length": torch.tensor([4, 4]),
            "num_action_switches": torch.tensor([1, 2]),
            "final_x": torch.tensor([0.0, 0.5]),
            "final_vx": torch.tensor([0.0, 0.5]),
            "final_vy": torch.tensor([0.0, 0.5]),
            "final_theta": torch.tensor([0.0, 0.5]),
            "received_x_corridor_070": torch.tensor([True, True]),
            "received_x_corridor_050": torch.tensor([True, True]),
            "received_x_corridor_035": torch.tensor([False, True]),
            "approach_visited": torch.tensor([True, True]),
        },
        non_tensors={"uid": np.array(["shared", "shared"], dtype=object)},
    )
    reward_tensors, _metrics = manager(proto)
    proto.batch["token_level_scores"] = reward_tensors["all"]
    proto, _kl_metrics = apply_kl_penalty(
        proto,
        kl_ctrl=type("KL", (), {"value": 0.0, "update": lambda self, current_kl, n_steps: None})(),
        kl_penalty="kl",
        action_token_len=1,
        action_chunks_len=1,
    )
    proto = compute_advantage(proto, gamma=1.0, lam=1.0, adv_estimator="grpo", config=config)
    assert proto.batch["token_level_rewards"].shape == (2, 4)
    assert proto.batch["advantages"].shape == (2, 4)
    assert proto.batch["returns"].shape == (2, 4)


def test_terminated_rate_uses_explicit_episode_flag_when_present():
    config = _base_config()
    config.reward.mode = "lunarlander_shaped"
    manager = RobRewardManager(num_examine=0, config=config)
    proto = DataProto.from_dict(
        tensors={
            "responses": torch.zeros((2, 4, 1), dtype=torch.long),
            "finish_step": torch.tensor([4, 4], dtype=torch.long),
            "complete": torch.tensor([False, False]),
            "success": torch.tensor([False, False]),
            "terminated": torch.tensor([True, False]),
            "truncated": torch.tensor([False, True]),
            "step_token_ends": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
            "r_sub": torch.zeros((2, 4), dtype=torch.float32),
            "r_prog": torch.zeros((2, 4), dtype=torch.float32),
            "r_micro": torch.zeros((2, 4), dtype=torch.float32),
            "r_smooth": torch.zeros((2, 4), dtype=torch.float32),
            "r_final": torch.zeros((2, 4), dtype=torch.float32),
            "r_total": torch.zeros((2, 4), dtype=torch.float32),
            "crash": torch.tensor([True, False]),
            "episode_length": torch.tensor([4, 4]),
            "num_action_switches": torch.tensor([1, 2]),
            "final_x": torch.tensor([0.0, 0.5]),
            "final_vx": torch.tensor([0.0, 0.5]),
            "final_vy": torch.tensor([0.0, 0.5]),
            "final_theta": torch.tensor([0.0, 0.5]),
        }
    )
    _reward_tensors, metrics = manager(proto)
    assert metrics["terminated_rate"] == 0.5
    assert metrics["truncated_rate"] == 0.5
