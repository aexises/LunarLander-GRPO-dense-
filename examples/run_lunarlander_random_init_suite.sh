set -euo pipefail
set -x

PROJECT_NAME="${PROJECT_NAME:-SimpleVLA-RL}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-lunarlander_grpo_suite}"
CKPT_PATH="${CKPT_PATH:-checkpoints}"
NUM_GPUS="${NUM_GPUS:-0}"
NUM_NODES="${NUM_NODES:-1}"
ABLATIONS="${ABLATIONS:-terminal_smooth terminal_sub_prog full dense_only}"
RUN_SUFFIX="${RUN_SUFFIX:-_random_init}"

RUN_DIRS=()
for ABLATION in $ABLATIONS; do
  PROJECT_NAME="$PROJECT_NAME" \
  EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  CKPT_PATH="$CKPT_PATH" \
  NUM_GPUS="$NUM_GPUS" \
  NUM_NODES="$NUM_NODES" \
  ABLATION="$ABLATION" \
  LUNARLANDER_POLICY_INIT=random_init \
  RUN_NAME_SUFFIX="$RUN_SUFFIX" \
  bash examples/run_lunarlander_rl.sh

  RUN_DIRS+=("$CKPT_PATH/$PROJECT_NAME/${EXPERIMENT_NAME}_${ABLATION}${RUN_SUFFIX}")
done

python3 examples/generate_lunarlander_report.py \
  "${RUN_DIRS[@]}" \
  --output "$CKPT_PATH/$PROJECT_NAME/${EXPERIMENT_NAME}${RUN_SUFFIX}_report.md"
