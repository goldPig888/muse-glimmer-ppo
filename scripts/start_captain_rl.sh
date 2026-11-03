#!/usr/bin/env bash
# Training entry point. No separate smoke/evaluation run.
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"
ENV_FILE=${CAPTAIN_ENV_FILE:-$PROJECT_ROOT/data/captain/training.env}
if [ ! -f "$ENV_FILE" ]; then
  bash scripts/setup_captain_training.sh
fi
source "$ENV_FILE"
: "${PPO_PY:?Set PPO_PY to the existing Glimmer training Python interpreter}"
RUN=${1:-runs/task0_captain_rl_$(date +%Y%m%d_%H%M%S)}
if [ "$#" -gt 0 ]; then shift; fi
H=$PROJECT_ROOT/data/captain/upstream/harness
# Use physical GPU 3 for perception and rendering; learner defaults to 1,2.
export SAM3_GPU=${SAM3_GPU:-3} MOLMO_GPU=${MOLMO_GPU:-3}
export LOG_DIR="$PROJECT_ROOT/data/captain/service_logs"
mkdir -p "$LOG_DIR"
for service in sam3 molmo; do
  if [ "$service" = sam3 ]; then URL=http://127.0.0.1:8114/openapi.json; else URL=http://127.0.0.1:8122/v1/models; fi
  if ! curl -fsS --max-time 3 "$URL" >/dev/null 2>&1; then
    echo "[captain-rl] Starting $service"
    bash "$H/services/start_perception.sh" "$service"
  fi
done
echo '[captain-rl] Waiting for perception services to finish loading'
for attempt in $(seq 1 120); do
  if curl -fsS --max-time 3 http://127.0.0.1:8114/openapi.json >/dev/null 2>&1 &&
     curl -fsS --max-time 3 http://127.0.0.1:8122/v1/models >/dev/null 2>&1; then break; fi
  if [ "$attempt" = 120 ]; then echo "Perception failed to start. Logs: $LOG_DIR" >&2; exit 1; fi
  if [ $((attempt % 5)) = 0 ]; then echo "[captain-rl] Still loading perception; logs: $LOG_DIR"; fi
  sleep 5
done
mkdir -p "$RUN"
export CUDA_VISIBLE_DEVICES=${LEARNER_GPUS:-1,2}
export PYTORCH_ALLOC_CONF=expandable_segments:True
"$PPO_PY" -u scripts/train_captain_rl.py --output "$RUN" \
  --harness-python "$CAPTAIN_PYTHON" --student-python "$STUDENT_PYTHON" \
  --egl-device "${EGL_DEVICE:-3}" "$@" 2>&1 | tee -a "$RUN/train.log"
