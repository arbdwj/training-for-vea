#!/bin/bash
# Bootstrap the 4xGH200 node (aarch64) for training-for-vea.
# Installs the vendored open-instruct + our pipeline deps, then runs a preflight.
# Usage:  bash scripts/node_bootstrap.sh
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OI="$REPO/third_party/open-instruct"
echo "== repo: $REPO"
echo "== arch: $(uname -m)   (expect aarch64 on GH200)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "!! no nvidia-smi"

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
echo "== preflight ($PY) =="
"$PY" "$REPO/scripts/preflight.py" || echo "!! preflight reported problems (see above)"

cat <<EOF

Next (run stages as SEPARATE processes so each frees its GPU before the next):
  $PY scripts/run.py run configs/e2e_32b.yaml --stages model --backend local   # ~100-step train
  $PY scripts/run.py run configs/e2e_32b.yaml --stages eval  --backend local   # eval the ckpt
  $PY scripts/run.py run configs/e2e_32b.yaml --stages judge                    # Nemotron-120B judge

Set HF_TOKEN in the environment first (gated model downloads).
EOF
