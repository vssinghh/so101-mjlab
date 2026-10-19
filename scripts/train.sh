#!/usr/bin/env bash
set -euo pipefail

# Unified so101-mjlab training launcher: runs on a remote Google Colab GPU (--colab)
# or local CPU (--local), streams logs live, and pulls both PyTorch checkpoints
# (model_*.pt) and exported ONNX policies (*.onnx) into checkpoints/<run_name>/.
#
# Usage:
#   ./scripts/train.sh <run_name> [--colab | --local] [train args...]
#
# Examples:
#   ./scripts/train.sh lift_v1 --colab --env.scene.num-envs 2048 --agent.max-iterations 1000
#   ./scripts/train.sh lift_rgb --colab --task Mjlab-Lift-Cube-SO101-Rgb --env.scene.num-envs 1024

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Usage: $0 <run_name> [--colab | --local] [--task <task_id>] [train args...]"
  exit 0
fi

RUN_NAME="$1"
shift

TARGET="colab"
TASK_ID="Mjlab-Lift-Cube-SO101"
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --colab)
      TARGET="colab"
      shift
      ;;
    --local)
      TARGET="local"
      shift
      ;;
    --task)
      TASK_ID="$2"
      shift 2
      ;;
    *)
      TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#TRAIN_ARGS[@]} -eq 0 ]]; then
  if [[ "$TARGET" == "colab" ]]; then
    TRAIN_ARGS=(--env.scene.num-envs 2048 --agent.max-iterations 800)
  else
    TRAIN_ARGS=(--env.scene.num-envs 16 --agent.max-iterations 50)
  fi
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOCAL_SAVE_DIR="$REPO_ROOT/checkpoints/$RUN_NAME"
mkdir -p "$LOCAL_SAVE_DIR"

echo "[train] Packing so101-mjlab snapshot into $LOCAL_SAVE_DIR/code_snapshot.tgz"
tar czf "$LOCAL_SAVE_DIR/code_snapshot.tgz" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  pyproject.toml README.md src tests

if [[ "$TARGET" == "local" ]]; then
  echo "[train] Running locally on Mac into $LOCAL_SAVE_DIR..."
  WANDB_MODE=offline PYTHONPATH="src" "/Users/vipinsingh/Documents/Antigravity/open source/mjlab/.venv/bin/train" \
    "$TASK_ID" "${TRAIN_ARGS[@]}" --log-root "$LOCAL_SAVE_DIR/logs" 2>&1 | tee "$LOCAL_SAVE_DIR/train.log"
  exit 0
fi

GPU_TYPE="${COLAB_GPU:-T4}"
KEEP_VM="${COLAB_KEEP_VM:-0}"
SESSION_CLEAN="$(echo "so101-${RUN_NAME}" | tr -cd '[:alnum:]-' | cut -c1-20)"
SESSION_CREATED=0
TRAINING_STARTED=0
DOWNLOADED_FINAL=0

pull_remote_checkpoints() {
  echo "[train] Pulling checkpoints and ONNX policy from Colab VM into $LOCAL_SAVE_DIR..."
  colab exec -s "$SESSION_CLEAN" --timeout 60 <<'PY' >/dev/null 2>&1 || true
import shutil
from pathlib import Path

out_dir = Path("/content/export_bundle")
out_dir.mkdir(parents=True, exist_ok=True)
log_root = Path("/content/logs/rsl_rl")
if log_root.exists():
    pts = sorted(log_root.rglob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    onnxs = sorted(log_root.rglob("*.onnx"), key=lambda p: p.stat().st_mtime)
    if pts:
        shutil.copy2(pts[-1], out_dir / "latest_model.pt")
        shutil.copy2(pts[-1], out_dir / pts[-1].name)
    if onnxs:
        shutil.copy2(onnxs[-1], out_dir / "policy.onnx")
PY
  colab download -s "$SESSION_CLEAN" /content/train.log "$LOCAL_SAVE_DIR/train.log" >/dev/null 2>&1 || true
  colab download -s "$SESSION_CLEAN" /content/export_bundle/latest_model.pt "$LOCAL_SAVE_DIR/latest_model.pt" >/dev/null 2>&1 || true
  colab download -s "$SESSION_CLEAN" /content/export_bundle/policy.onnx "$LOCAL_SAVE_DIR/policy.onnx" >/dev/null 2>&1 || true
}

cleanup() {
  rm -f "$LOCAL_SAVE_DIR/.remote_train.log" "$LOCAL_SAVE_DIR/.remote_train.exit"
  if [[ "$TRAINING_STARTED" == "1" && "$DOWNLOADED_FINAL" == "0" ]]; then
    pull_remote_checkpoints || true
  fi
  if [[ "$SESSION_CREATED" == "1" ]]; then
    if [[ "$KEEP_VM" == "1" ]]; then
      echo "[train] COLAB_KEEP_VM=1 set; keeping Colab session '$SESSION_CLEAN' alive."
    else
      echo "[train] Stopping Colab session '$SESSION_CLEAN'..."
      colab stop -s "$SESSION_CLEAN" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT INT TERM

echo "[train] Provisioning Colab session '$SESSION_CLEAN' with GPU=$GPU_TYPE..."
colab new -s "$SESSION_CLEAN" --gpu "$GPU_TYPE"
SESSION_CREATED=1

echo "[train] Uploading so101-mjlab snapshot ($LOCAL_SAVE_DIR/code_snapshot.tgz) to Colab VM..."
colab upload -s "$SESSION_CLEAN" "$LOCAL_SAVE_DIR/code_snapshot.tgz" /content/code_snapshot.tgz

TRAIN_ARGS_JSON="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${TRAIN_ARGS[@]}")"

echo "[train] Bootstrapping Colab VM (installing mjlab + so101-mjlab) and starting training..."
colab exec -s "$SESSION_CLEAN" --timeout 900 <<EOF
import json
import os
import subprocess
import sys
import tarfile

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available on this Colab runtime.")
print(f"[VM] GPU verified: {torch.cuda.get_device_name(0)}", flush=True)

repo_dir = "/content/so101-mjlab"
os.makedirs(repo_dir, exist_ok=True)
with tarfile.open("/content/code_snapshot.tgz", "r:gz") as tar:
    tar.extractall(repo_dir)
print("[VM] Extracted so101-mjlab snapshot.", flush=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
subprocess.run(
    [
        "uv",
        "pip",
        "install",
        "--system",
        "-q",
        "git+https://github.com/mujocolab/mjlab.git@main",
        "onnx",
        "onnxscript",
        "onnxruntime",
        "pyserial",
    ],
    check=True,
)
subprocess.run(
    ["uv", "pip", "install", "--system", "-q", "--no-deps", "-e", repo_dir],
    check=True,
)
print("[VM] Installed mjlab and so101-mjlab.", flush=True)

for p in ("/content/train.log", "/content/train.exit", "/content/train.pid"):
    if os.path.exists(p):
        os.remove(p)

train_args = json.loads('''$TRAIN_ARGS_JSON''')
cmd = [
    "train",
    "$TASK_ID",
    *train_args,
    "--log-root",
    "/content/logs/rsl_rl",
]
runner_code = (
    "import os, subprocess\n"
    f"os.chdir({repo_dir!r})\n"
    "os.environ['MUJOCO_GL'] = 'egl'\n"
    "os.environ['WANDB_MODE'] = 'offline'\n"
    "with open('/content/train.log', 'w') as logf:\n"
    f"    proc = subprocess.Popen({cmd!r}, stdout=logf, stderr=subprocess.STDOUT)\n"
    "    rc = proc.wait()\n"
    "with open('/content/train.exit', 'w') as ef:\n"
    "    ef.write(str(rc))\n"
)
with open("/content/run_training_bg.py", "w") as rf:
    rf.write(runner_code)

bg_proc = subprocess.Popen(
    [sys.executable, "/content/run_training_bg.py"],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    close_fds=True,
    start_new_session=True,
)
with open("/content/train.pid", "w") as pf:
    pf.write(str(bg_proc.pid))
print(f"[VM] Started background training (PID={bg_proc.pid}): {' '.join(cmd)}", flush=True)
EOF
TRAINING_STARTED=1

echo "[train] Streaming remote training logs..."
PRINTED_LINES=0
TMP_LOG="$LOCAL_SAVE_DIR/.remote_train.log"
TMP_EXIT="$LOCAL_SAVE_DIR/.remote_train.exit"
rm -f "$TMP_LOG" "$TMP_EXIT"

FAIL_COUNT=0
POLL_ITER=0
while true; do
  POLL_ITER=$((POLL_ITER + 1))
  if (( POLL_ITER % 6 == 0 )); then
    colab status -s "$SESSION_CLEAN" >/dev/null 2>&1 || true
  fi
  if colab download -s "$SESSION_CLEAN" /content/train.log "$TMP_LOG" >/dev/null 2>&1; then
    FAIL_COUNT=0
    TOTAL_LINES="$(wc -l < "$TMP_LOG" | tr -d ' ')"
    if [[ "$TOTAL_LINES" -gt "$PRINTED_LINES" ]]; then
      tail -n +"$((PRINTED_LINES + 1))" "$TMP_LOG"
      PRINTED_LINES="$TOTAL_LINES"
      cp "$TMP_LOG" "$LOCAL_SAVE_DIR/train.log"
    fi
    if (( POLL_ITER % 12 == 0 )); then
      pull_remote_checkpoints || true
    fi
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    if (( FAIL_COUNT >= 4 )); then
      echo "[train] Error: Lost connection to Colab session '$SESSION_CLEAN' after $FAIL_COUNT failed polls."
      exit 1
    fi
  fi
  if colab download -s "$SESSION_CLEAN" /content/train.exit "$TMP_EXIT" >/dev/null 2>&1; then
    if colab download -s "$SESSION_CLEAN" /content/train.log "$TMP_LOG" >/dev/null 2>&1; then
      TOTAL_LINES="$(wc -l < "$TMP_LOG" | tr -d ' ')"
      if [[ "$TOTAL_LINES" -gt "$PRINTED_LINES" ]]; then
        tail -n +"$((PRINTED_LINES + 1))" "$TMP_LOG"
        PRINTED_LINES="$TOTAL_LINES"
      fi
      cp "$TMP_LOG" "$LOCAL_SAVE_DIR/train.log"
    fi
    EXIT_STATUS="$(tr -d '[:space:]' < "$TMP_EXIT")"
    rm -f "$TMP_LOG" "$TMP_EXIT"
    if [[ "$EXIT_STATUS" != "0" ]]; then
      echo "[train] Remote training exited with status $EXIT_STATUS"
      exit "$EXIT_STATUS"
    fi
    break
  fi
  echo "import time; time.sleep(5)" | colab exec -s "$SESSION_CLEAN" --timeout 30 >/dev/null 2>&1 || sleep 5
done

pull_remote_checkpoints
DOWNLOADED_FINAL=1

echo "[train] Done! Checkpoints and ONNX policy saved in $LOCAL_SAVE_DIR:"
ls -lh "$LOCAL_SAVE_DIR"
