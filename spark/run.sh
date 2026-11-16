#!/usr/bin/env bash
# Train and run Laya on a DGX Spark, using Laya's own ARM64 / CUDA 13.0 container.
#
#   spark/run.sh smoke      GPU inference check (the one Laya's docs ask Spark users to run)
#   spark/run.sh dryrun     validate data/train.jsonl without loading weights
#   spark/run.sh train      fine-tune, calibrate, and evaluate on data/test.jsonl
#   spark/run.sh predict [checkpoint] [name]
#                           answer data/test.jsonl; default is the fine-tuned checkpoint.
#                           e.g. `spark/run.sh predict english base` for the untuned baseline
#
# Needs: Docker with the NVIDIA Container Toolkit, and network access to GitHub and Hugging Face.
# Settings (environment): BASE=english|multilingual|typed-decisions  NAME=hb-review
#   MAX_LEN=1024  HEAD_MAX_LEN=256  TRAIN_ARGS="--epochs 8 --loss soft-ce"  LAYA_GPU_ID=0
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
export HB_REPO=$(dirname "$HERE")
export HB_DATA=${HB_DATA:-$HB_REPO/data}
export HB_OUT=${HB_OUT:-$HB_REPO/out}
export LAYA_CACHE_VOLUME=${LAYA_CACHE_VOLUME:-hb_laya_model_cache}

LAYA_DIR=${LAYA_DIR:-$HOME/laya}
LAYA_REF=${LAYA_REF:-b1d656952f0f9581ea691a70cb6f38abf3eb13a9}   # v0.4.35, the revision this was written against
BASE=${BASE:-english}
NAME=${NAME:-hb-review}
MAX_LEN=${MAX_LEN:-1024}
HEAD_MAX_LEN=${HEAD_MAX_LEN:-256}

[ -d "$LAYA_DIR/.git" ] || git clone https://github.com/NandhaKishorM/laya "$LAYA_DIR"
if [ "$(git -C "$LAYA_DIR" rev-parse HEAD)" != "$LAYA_REF" ]; then
  git -C "$LAYA_DIR" fetch origin "$LAYA_REF"
  git -C "$LAYA_DIR" checkout --detach "$LAYA_REF"
fi
mkdir -p "$HB_OUT"
chmod a+rwx "$HB_OUT"   # the container runs as its own non-root user

run() {
  docker compose --project-directory "$LAYA_DIR" \
    -f "$LAYA_DIR/compose.yaml" -f "$LAYA_DIR/compose.spark.yaml" -f "$HERE/compose.hb.yaml" \
    run --build --rm laya "$@"
}

case "${1:-}" in
  smoke)
    run python -c '
import json, platform, torch
from pathlib import Path
from laya import load
assert platform.machine() == "aarch64"
assert torch.cuda.is_available()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
agent = load("convaiinnovations/laya", device="cuda")
request = json.loads(Path("/opt/laya/examples/request.json").read_text())
result = agent.predict(request["state"], request["questions"])
assert next(agent.model.parameters()).device.type == "cuda", "fell back to CPU"
assert set(result["answers"]) == set(request["questions"])
print("CUDA inference passed", torch.cuda.max_memory_allocated())
'
    ;;
  dryrun)
    run laya-train --data /data/train.jsonl --base "$BASE" \
      --max-len "$MAX_LEN" --head-max-len "$HEAD_MAX_LEN" --dry-run
    ;;
  train)
    # shellcheck disable=SC2086
    run laya-train --data /data/train.jsonl --eval /data/test.jsonl --base "$BASE" \
      --out "/out/$NAME" --max-len "$MAX_LEN" --head-max-len "$HEAD_MAX_LEN" \
      --device cuda ${TRAIN_ARGS:-} 2>&1 | tee "$HB_OUT/$NAME.train.log"
    ;;
  predict)
    CKPT=${2:-/out/$NAME}
    LABEL=${3:-laya}
    run python /hb/evaluate.py predict --backend laya --checkpoint "$CKPT" --device cuda \
      --data /data/test.jsonl --max-len "$MAX_LEN" --out "/out/pred.$LABEL.jsonl"
    ;;
  *)
    sed -n '2,13p' "$0"; exit 2 ;;
esac
