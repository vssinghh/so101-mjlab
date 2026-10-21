#!/usr/bin/env bash
set -euo pipefail

# Unified so101-mjlab training launcher: runs on a remote Google Colab GPU (--colab)
# or local CPU (--local), streams logs live, and pulls both PyTorch checkpoints
# (model_*.pt) and exported ONNX policies (*.onnx) into checkpoints/<run_name>/.
#
# Usage:
#   ./scripts/train.sh <run_name> [--colab | --local] [train args...]
#
# Example:
#   ./scripts/train.sh lift_v7_sysid --colab --env.scene.num-envs 2048 --agent.max-iterations 1000

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
    TRAIN_ARGS=(--env.scene.num-envs 2048 --agent.max-iterations 1000)
  else
    TRAIN_ARGS=(--env.scene.num-envs 16 --agent.max-iterations 50)
  fi
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOCAL_SAVE_DIR="$REPO_ROOT/checkpoints/$RUN_NAME"
mkdir -p "$LOCAL_SAVE_DIR"

echo "[train] Packing so101-mjlab snapshot into $LOCAL_SAVE_DIR/code_snapshot.tgz"
SNAPSHOT_ITEMS=(pyproject.toml README.md src tests)
tar czf "$LOCAL_SAVE_DIR/code_snapshot.tgz" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.stl' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  "${SNAPSHOT_ITEMS[@]}"

if [[ "$TARGET" == "local" ]]; then
  echo "[train] Running locally into $LOCAL_SAVE_DIR..."
  WANDB_MODE=offline PYTHONPATH="src" uv run train \
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
  colab download -s "$SESSION_CLEAN" /content/train.log "$LOCAL_SAVE_DIR/train.log" >/dev/null 2>&1 || true
  colab download -s "$SESSION_CLEAN" /content/export_bundle/latest_model.pt "$LOCAL_SAVE_DIR/latest_model.pt" >/dev/null 2>&1 || true
  colab download -s "$SESSION_CLEAN" /content/export_bundle/policy.onnx "$LOCAL_SAVE_DIR/policy.onnx" >/dev/null 2>&1 || true
  if [[ -f "$LOCAL_SAVE_DIR/policy.onnx" ]]; then
    cp "$LOCAL_SAVE_DIR/policy.onnx" "$REPO_ROOT/checkpoints/policy.onnx"
  fi
}

cleanup() {
  rm -f "$LOCAL_SAVE_DIR/.remote_train.log" "$LOCAL_SAVE_DIR/.remote_train.exit"
  if [[ "$TRAINING_STARTED" == "1" && "$DOWNLOADED_FINAL" == "0" ]]; then
    pull_remote_checkpoints || true
  fi
  if [[ "$SESSION_CREATED" == "1" ]]; then
    if [[ "$KEEP_VM" == "1" || "$DOWNLOADED_FINAL" == "0" ]]; then
      echo "[train] Keeping Colab session '$SESSION_CLEAN' alive (DOWNLOADED_FINAL=$DOWNLOADED_FINAL)."
    else
      echo "[train] Stopping Colab session '$SESSION_CLEAN'..."
      python3 -c "import subprocess; subprocess.run(['colab', 'stop', '-s', '$SESSION_CLEAN'], timeout=15)" >/dev/null 2>&1 || true
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
if not os.path.exists(os.path.join(repo_dir, "src/so101_mjlab/so101/xmls/assets")):
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/vssinghh/so101-mjlab.git", repo_dir],
        check=True,
    )
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

for p in ("/content/train.log", "/content/train_tail.txt", "/content/train.exit", "/content/train.pid"):
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
runner_code = f"""import os, shutil, subprocess, time
from pathlib import Path
import torch, torch.nn as nn

os.chdir({repo_dir!r})
os.environ['MUJOCO_GL'] = 'egl'
os.environ['WANDB_MODE'] = 'offline'
os.environ['PYTHONUNBUFFERED'] = '1'
out_dir = Path('/content/export_bundle')
out_dir.mkdir(parents=True, exist_ok=True)
log_root = Path('/content/logs/rsl_rl')

def export_latest():
    if not log_root.exists():
        return
    pts = [p for p in sorted(log_root.rglob('model_*.pt'), key=lambda p: p.stat().st_mtime) if 'warmup' not in str(p)]
    if not pts:
        return
    latest = pts[-1]
    shutil.copy2(latest, out_dir / 'latest_model.pt')
    ckpt = torch.load(latest, map_location='cpu', weights_only=False)
    sd = ckpt.get('actor_state_dict') or ckpt.get('model_state_dict', {{}})
    if 'mlp.0.weight' in sd:
        class ExportActor(nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                self.has_norm = 'obs_normalizer._mean' in state_dict
                if self.has_norm:
                    self.register_buffer('mean', state_dict['obs_normalizer._mean'].clone())
                    self.register_buffer('std', state_dict['obs_normalizer._std'].clone())
                layers = []
                idx = 0
                while f'mlp.{{idx}}.weight' in state_dict:
                    w = state_dict[f'mlp.{{idx}}.weight']
                    b = state_dict[f'mlp.{{idx}}.bias']
                    lin = nn.Linear(w.shape[1], w.shape[0])
                    lin.weight.data.copy_(w)
                    lin.bias.data.copy_(b)
                    layers.append(lin)
                    if f'mlp.{{idx + 2}}.weight' in state_dict:
                        layers.append(nn.ELU())
                    idx += 2
                self.mlp = nn.Sequential(*layers)
            def forward(self, obs):
                if self.has_norm:
                    obs = (obs - self.mean) / (self.std + 1e-8)
                return self.mlp(obs)
        actor = ExportActor(sd).eval()
        obs_dim = sd['mlp.0.weight'].shape[1]
        torch.onnx.export(actor, torch.zeros(1, obs_dim), str(out_dir / 'policy.onnx'), input_names=['obs'], output_names=['actions'], opset_version=17, dynamo=False)

with open('/content/train.log', 'w') as logf:
    proc = subprocess.Popen({cmd!r}, stdout=logf, stderr=subprocess.STDOUT)
    last_exported = None
    while proc.poll() is None:
        time.sleep(5)
        try:
            lines = Path('/content/train.log').read_text(errors='ignore').splitlines()
            Path('/content/train_tail.txt').write_text(chr(10).join(lines[-28:]) + chr(10))
            if log_root.exists():
                pts = [p for p in sorted(log_root.rglob('model_*.pt'), key=lambda p: p.stat().st_mtime) if 'warmup' not in str(p)]
                if pts and pts[-1] != last_exported:
                    export_latest()
                    last_exported = pts[-1]
        except Exception:
            pass
    rc = proc.wait()

try:
    lines = Path('/content/train.log').read_text(errors='ignore').splitlines()
    Path('/content/train_tail.txt').write_text(chr(10).join(lines[-28:]) + chr(10))
    export_latest()
except Exception:
    pass

with open('/content/train.exit', 'w') as ef:
    ef.write(str(rc))
"""
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
LAST_TAIL=""
TMP_LOG="$LOCAL_SAVE_DIR/.remote_train.log"
TMP_EXIT="$LOCAL_SAVE_DIR/.remote_train.exit"
rm -f "$TMP_LOG" "$TMP_EXIT"

FAIL_COUNT=0
POLL_ITER=0
while true; do
  POLL_ITER=$((POLL_ITER + 1))
  if (( POLL_ITER % 4 == 0 )); then
    echo "1" | colab exec -s "$SESSION_CLEAN" --timeout 20 >/dev/null 2>&1 || true
  fi
  if colab download -s "$SESSION_CLEAN" /content/train_tail.txt "$TMP_LOG" >/dev/null 2>&1; then
    FAIL_COUNT=0
    CUR_TAIL="$(cat "$TMP_LOG")"
    if [[ "$CUR_TAIL" != "$LAST_TAIL" && -n "$CUR_TAIL" ]]; then
      echo "$CUR_TAIL"
      LAST_TAIL="$CUR_TAIL"
    fi
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    if (( FAIL_COUNT >= 12 )); then
      echo "[train] Warning: Poll failed $FAIL_COUNT times on '$SESSION_CLEAN'; sleeping and retrying..."
      FAIL_COUNT=0
    fi
  fi
  if colab download -s "$SESSION_CLEAN" /content/train.exit "$TMP_EXIT" >/dev/null 2>&1; then
    EXIT_STATUS="$(tr -d '[:space:]' < "$TMP_EXIT")"
    rm -f "$TMP_LOG" "$TMP_EXIT"
    if [[ "$EXIT_STATUS" != "0" ]]; then
      echo "[train] Remote training exited with status $EXIT_STATUS"
      exit "$EXIT_STATUS"
    fi
    break
  fi
  sleep 12
done

pull_remote_checkpoints
DOWNLOADED_FINAL=1
echo "[train] Stopping Colab session '$SESSION_CLEAN'..."
python3 -c "import subprocess; subprocess.run(['colab', 'stop', '-s', '$SESSION_CLEAN'], timeout=15)" >/dev/null 2>&1 || true

echo "[train] Done! Checkpoints and ONNX policy saved in $LOCAL_SAVE_DIR:"
ls -lh "$LOCAL_SAVE_DIR"
