#!/bin/bash
# Rebuild the vendored open-instruct venv ON LUSTRE (not on the capped /home), via uv sync.
#
# WHY: the venv was a uv environment whose package files linked into ~/.cache/uv. Deleting
# that cache to free /home quota destroyed the venv (python/activate gone -> every rung died
# with "python: command not found", exit 127). Building the venv on Lustre means (a) it does
# not consume the ~101GB /home quota, and (b) a future /home cache-clear can't break it.
#
# USAGE (login node, from repo root — login has network for the wheel downloads):
#   bash scripts/rebuild_venv.sh                 # uses VEA_STORE or the default project path
#   bash scripts/rebuild_venv.sh /lus/.../vea-store
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OI="$REPO/third_party/open-instruct"
STORE="${1:-${VEA_STORE:-/lus/lfs1aip2/projects/a6ab/$USER/vea-store}}"

echo "== rebuilding open-instruct venv on Lustre =="
echo "   REPO=$REPO"
echo "   STORE=$STORE"
mkdir -p "$STORE/cache/uv" "$STORE/venv-oi" || { echo "!! cannot write under $STORE"; exit 1; }

# put uv's cache + the project env on Lustre
export UV_CACHE_DIR="$STORE/cache/uv"
export UV_PROJECT_ENVIRONMENT="$STORE/venv-oi"

# replace the broken .venv in the repo with a symlink to the Lustre env, so setup_env.sh's
# hardcoded third_party/open-instruct/.venv/bin/activate resolves to the real env.
if [ ! -L "$OI/.venv" ]; then rm -rf "$OI/.venv"; fi
ln -sfn "$STORE/venv-oi" "$OI/.venv"
echo "   .venv -> $(readlink -f "$OI/.venv")"

# build it (node_bootstrap honors UV_PROJECT_ENVIRONMENT, so uv sync installs onto Lustre)
bash "$REPO/scripts/node_bootstrap.sh"

# make the uv-on-Lustre choice durable for future syncs via storage_env.sh (idempotent)
ENVF="$REPO/scripts/storage_env.sh"
if [ -f "$ENVF" ] && ! grep -q UV_CACHE_DIR "$ENVF"; then
  {
    echo "# uv cache + project env on Lustre (rebuild_venv.sh)"
    echo "export UV_CACHE_DIR=\"$STORE/cache/uv\""
    echo "export UV_PROJECT_ENVIRONMENT=\"$STORE/venv-oi\""
  } >> "$ENVF"
  echo "== appended UV_CACHE_DIR/UV_PROJECT_ENVIRONMENT to $ENVF"
fi

echo
echo "== verify the env works:"
echo "   \"$OI/.venv/bin/python\" -c 'import torch, vllm, deepspeed; print(torch.__version__)'"
echo "== then resubmit:  sbatch scripts/run_breadth_rungs.sbatch"
