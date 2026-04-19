set -euo pipefail
set -x

PROJECT_NAME="${PROJECT_NAME:-SimpleVLA-RL}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-lunarlander_grpo_sanity}"
CKPT_PATH="${CKPT_PATH:-checkpoints}"
NUM_GPUS="${NUM_GPUS:-0}"
NUM_NODES="${NUM_NODES:-1}"
ABLATION="${ABLATION:-full}"
AUDIT_ENABLED="${AUDIT_ENABLED:-true}"
AUDIT_MAX_TRAIN_EPISODES="${AUDIT_MAX_TRAIN_EPISODES:-128}"
AUDIT_MAX_VAL_EPISODES="${AUDIT_MAX_VAL_EPISODES:-32}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-256}"
N_SAMPLES="${N_SAMPLES:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
TEST_FREQ="${TEST_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-25}"
MAX_STEPS="${MAX_STEPS:-400}"

case "$ABLATION" in
  terminal_only)
    REWARD_MODE=terminal_only
    W_SUB=0.0
    W_PROG=0.0
    W_SMOOTH=0.0
    W_FINAL=1.0
    ;;
  terminal_smooth)
    REWARD_MODE=lunarlander_shaped
    W_SUB=0.0
    W_PROG=0.0
    W_SMOOTH=0.005
    W_FINAL=1.0
    ;;
  terminal_sub_prog)
    REWARD_MODE=lunarlander_shaped
    W_SUB=0.10
    W_PROG=0.30
    W_SMOOTH=0.0
    W_FINAL=1.0
    ;;
  full)
    REWARD_MODE=lunarlander_shaped
    W_SUB=0.10
    W_PROG=0.30
    W_SMOOTH=0.005
    W_FINAL=1.0
    ;;
  dense_only)
    REWARD_MODE=lunarlander_shaped
    W_SUB=0.10
    W_PROG=0.30
    W_SMOOTH=0.005
    W_FINAL=0.0
    ;;
  *)
    echo "Unknown ablation: $ABLATION"
    exit 1
    ;;
esac

HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
  data.task_suite_name=lunarlander \
  data.filter_accuracy=False \
  data.filter_format=False \
  data.oversample_factor=1 \
  actor_rollout_ref.model.vla=lunarlander_mlp \
  actor_rollout_ref.model.path='unused_for_lunarlander' \
  actor_rollout_ref.model.action_token_len=1 \
  actor_rollout_ref.model.action_chunks_len=1 \
  actor_rollout_ref.model.hidden_size=128 \
  actor_rollout_ref.model.seed=0 \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=3e-4 \
  actor_rollout_ref.actor.optim.warmup_style=constant \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.ppo_epochs=2 \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.task_suite_name=lunarlander \
  actor_rollout_ref.rollout.env_name=LunarLander-v3 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.do_sample=True \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.0 \
  verifier.reward_coef=1.0 \
  reward.mode=$REWARD_MODE \
  reward.distribution_mode=last_token_per_step \
  reward.weights.sub=$W_SUB \
  reward.weights.prog=$W_PROG \
  reward.weights.smooth=$W_SMOOTH \
  reward.weights.final=$W_FINAL \
  reward.subgoal_min=-1.0 \
  reward.subgoal_max=0.0 \
  reward.phase_thresholds.center_x_abs_for_align=0.35 \
  reward.phase_thresholds.center_x_abs_for_descend=0.20 \
  reward.phase_thresholds.center_x_abs_for_touchdown=0.15 \
  reward.phase_thresholds.angle_abs_for_descend=0.15 \
  reward.phase_thresholds.angle_abs_for_touchdown=0.10 \
  reward.phase_thresholds.height_for_descend=0.75 \
  reward.phase_thresholds.height_for_touchdown=0.30 \
  reward.phase_thresholds.horizontal_speed_abs_for_touchdown=0.20 \
  reward.phase_thresholds.vertical_speed_abs_for_touchdown=0.20 \
  reward.phase_thresholds.vertical_speed_min_for_descend=-0.05 \
  reward.phase_thresholds.descent_target_height=0.50 \
  reward.coeffs.approach_x=1.0 \
  reward.coeffs.approach_vx=0.3 \
  reward.coeffs.align_x=0.7 \
  reward.coeffs.align_theta=1.0 \
  reward.coeffs.align_omega=0.3 \
  reward.coeffs.descend_y=0.5 \
  reward.coeffs.descend_vy=1.0 \
  reward.coeffs.descend_x=0.3 \
  reward.coeffs.touchdown_vx=0.7 \
  reward.coeffs.touchdown_vy=1.0 \
  reward.coeffs.touchdown_theta=0.7 \
  reward.coeffs.touchdown_omega=0.3 \
  reward.progress_rewards.enter_align=0.25 \
  reward.progress_rewards.enter_descend=0.25 \
  reward.progress_rewards.enter_touchdown=0.25 \
  reward.success.max_abs_x=0.20 \
  reward.success.max_abs_vx=0.20 \
  reward.success.max_abs_vy=0.20 \
  reward.success.max_abs_theta=0.15 \
  audit.enabled=$AUDIT_ENABLED \
  audit.max_train_episodes=$AUDIT_MAX_TRAIN_EPISODES \
  audit.max_val_episodes=$AUDIT_MAX_VAL_EPISODES \
  trainer.logger="['console']" \
  trainer.project_name=$PROJECT_NAME \
  trainer.experiment_name="${EXPERIMENT_NAME}_${ABLATION}" \
  trainer.default_local_dir="$CKPT_PATH/$PROJECT_NAME/${EXPERIMENT_NAME}_${ABLATION}" \
  trainer.n_gpus_per_node=$NUM_GPUS \
  trainer.nnodes=$NUM_NODES \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.total_epochs=$TOTAL_EPOCHS \
  trainer.val_only=False \
  trainer.val_before_train=True \
  actor_rollout_ref.rollout.max_steps=$MAX_STEPS \
  data.num_trials_per_task=$NUM_TRIALS_PER_TASK \
  data.n_samples=$N_SAMPLES \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.val_batch_size=$VAL_BATCH_SIZE
