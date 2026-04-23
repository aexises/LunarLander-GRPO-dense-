set -euo pipefail
set -x

PROJECT_NAME="${PROJECT_NAME:-SimpleVLA-RL}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-lunarlander_grpo_suite}"
CKPT_PATH="${CKPT_PATH:-checkpoints}"
NUM_GPUS="${NUM_GPUS:-0}"
NUM_NODES="${NUM_NODES:-1}"
ABLATIONS="${ABLATIONS:-terminal_only terminal_smooth terminal_sub_prog full dense_only}"
RUN_RANDOM_INIT_VARIANTS="${RUN_RANDOM_INIT_VARIANTS:-true}"
RANDOM_INIT_ABLATIONS="${RANDOM_INIT_ABLATIONS:-terminal_smooth terminal_sub_prog full dense_only}"

RUN_DIRS=()
run_ablation() {
  local ablation="$1"
  local init_mode="$2"
  local run_suffix="$3"

  PROJECT_NAME="$PROJECT_NAME" \
  EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  CKPT_PATH="$CKPT_PATH" \
  NUM_GPUS="$NUM_GPUS" \
  NUM_NODES="$NUM_NODES" \
  ABLATION="$ablation" \
  LUNARLANDER_POLICY_INIT="$init_mode" \
  RUN_NAME_SUFFIX="$run_suffix" \
  bash examples/run_lunarlander_rl.sh

  RUN_DIRS+=("$CKPT_PATH/$PROJECT_NAME/${EXPERIMENT_NAME}_${ablation}${run_suffix}")
}

for ABLATION in $ABLATIONS; do
  run_ablation "$ABLATION" competent_anchor ""
done

if [ "$RUN_RANDOM_INIT_VARIANTS" = "true" ]; then
  for ABLATION in $RANDOM_INIT_ABLATIONS; do
    case " $ABLATIONS " in
      *" $ABLATION "*) run_ablation "$ABLATION" random_init "_random_init" ;;
    esac
  done
fi

python3 examples/generate_lunarlander_report.py \
  "${RUN_DIRS[@]}" \
  --output "$CKPT_PATH/$PROJECT_NAME/${EXPERIMENT_NAME}_report.md"
