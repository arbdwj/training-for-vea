#!/bin/bash
# Bootstrap the 4xGH200 node (aarch64) for training-for-vea.
# Installs the vendored open-instruct + our pipeline deps, then runs a preflight.
# Usage:  bash scripts/node_bootstrap.sh
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OI="$REPO/third_party/open-instruct"
echo "== repo: $REPO"
echo "== arch: $(uname -m)   (expect aarch64 on GH200)"
# SLURM: the login node has no GPUs. Installing deps is fine here; the GPU preflight
# (torch.cuda, vLLM load) only passes inside an allocation. Grab one with e.g.:
#   salloc --nodes=1 --gpus=4 --time=00:30:00     then re-run this script, or just
#   run the preflight alone inside the allocation:  python scripts/preflight.py
if nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null; then
  HAVE_GPU=1
else
  echo "!! no GPUs visible here (SLURM login node?) — install will proceed; run the"
  echo "   GPU preflight inside an allocation:  salloc --gpus=4 -t 00:30:00  then  python scripts/preflight.py"
  HAVE_GPU=0
fi

# ---------------------------------------------------------------------------
# 1. open-instruct env (uv preferred; it knows the aarch64 markers in uv.lock)
# ---------------------------------------------------------------------------
cd "$OI"
if command -v uv >/dev/null; then
  echo "== uv sync (open-instruct; flash-attn auto-skipped on aarch64) ..."
  # cuda13 group targets the GH200 cu130 torch index per open-instruct pyproject
  uv sync --group cuda13 || uv sync || { echo "!! uv sync failed — see output above"; }
  VENV="$OI/.venv"
else
  echo "!! uv not found. Install it (https://docs.astral.sh/uv) or create a venv manually:"
  echo "   python3.12 -m venv $OI/.venv && source $OI/.venv/bin/activate && pip install -e $OI"
  VENV="$OI/.venv"
fi

# ---------------------------------------------------------------------------
# 2. our pipeline deps into the same env
# ---------------------------------------------------------------------------
if [ -d "$VENV" ]; then
  echo "== installing pipeline deps into $VENV ..."
  "$VENV/bin/pip" install -q pyyaml rich anthropic openai datasets huggingface-hub || true
fi

# ---------------------------------------------------------------------------
# 3. preflight
# ---------------------------------------------------------------------------
PY="$VENV/bin/python"; [ -x "$PY" ] || PY="python3"
if [ "${HAVE_GPU:-0}" = "1" ]; then
  echo "== preflight ($PY) =="
  "$PY" "$REPO/scripts/preflight.py" || echo "!! preflight reported problems (see above)"
else
  echo "== skipping GPU preflight (no GPUs on this node). Inside an allocation run:"
  echo "   $PY scripts/preflight.py"
fi

cat <<EOF

Next — submit the whole pipeline to SLURM (one allocation, logs to logs/vea_e2e_<jobid>.log):
  export HF_TOKEN=...
  sbatch scripts/run_e2e.sbatch configs/breadth_32b_fast.yaml

Or interactively inside an allocation (salloc --nodes=1 --gpus=4 -t 04:00:00):
  source third_party/open-instruct/.venv/bin/activate
  python scripts/run.py run configs/breadth_32b_fast.yaml --stages model --backend local 2>&1 | tee logs/model.log
  python scripts/run.py run configs/breadth_32b_fast.yaml --stages eval  --backend local 2>&1 | tee logs/eval.log
  python scripts/run.py run configs/breadth_32b_fast.yaml --stages judge                   2>&1 | tee logs/judge.log
EOF
