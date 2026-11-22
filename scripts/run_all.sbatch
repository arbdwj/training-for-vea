#!/bin/bash
# ONE script: install -> preflight -> train -> eval -> judge, all in a single SLURM
# allocation, all output to logs/vea_all_<jobid>.log.
#
# Submit from the repo root:
#   export HF_TOKEN=...                                   # carried into the job
#   sbatch scripts/run_all.sbatch configs/breadth_32b_fast.yaml
#
# Env knobs:  SKIP_INSTALL=1 (skip uv sync on reruns)   CUDA_GROUP=cuda12 (CUDA-12 node)
# Watch live:  tail -f logs/vea_all_*.log
#
#SBATCH --job-name=vea-all
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00          # generous: first run downloads ~300GB of weights
#SBATCH --output=logs/vea_all_%j.log
## ---- set for your cluster ----
## #SBATCH --partition=gpu
## #SBATCH --account=YOUR_ACCOUNT
## #SBATCH --mem=0

set -uo pipefail
CONFIG="${1:-configs/breadth_32b_fast.yaml}"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p logs

echo "########## node=$(hostname) job=${SLURM_JOB_ID:-none} config=$CONFIG  $(date) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "!! no GPUs visible"

# 1. install + preflight (skip with SKIP_INSTALL=1 on reruns)
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  echo "########## INSTALL + PREFLIGHT  $(date) ##########"
  bash scripts/node_bootstrap.sh || echo "!! bootstrap reported problems (continuing)"
fi

source third_party/open-instruct/.venv/bin/activate || { echo "!! no venv — install failed"; exit 1; }
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
# vLLM v1 engine-core msgpack-serializes rollout outputs; Olmo-3 outputs carry a torch.dtype that
# msgpack can't encode. Allow the pickle fallback — safe here: engine traffic is localhost/trusted.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# --- CUDA toolkit for DeepSpeed (needs CUDA_HOME/nvcc at import) ------------
module load "${CUDA_MODULE:-cudatoolkit/24.11_12.6}" 2>/dev/null \
  || module load cuda/12.6 2>/dev/null || module load cudatoolkit 2>/dev/null || true
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/triton_$USER}"; mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true
echo "CUDA_HOME=${CUDA_HOME:-UNSET}  nvcc=$(command -v nvcc || echo none)"
[ -z "${CUDA_HOME:-}" ] && echo "!! CUDA_HOME unset — DeepSpeed will fail. Set CUDA_MODULE=<module> (module avail cuda)."

# Ray needs a high open-file limit (default 4000 here starves its sockets -> raylet/worker EOF).
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
# ...and a bounded worker count: Ray prestarts ~1 worker/CPU; on a 64+ core node that can exceed
# the per-user process limit (raylet EAGAIN crash). Cap Ray's CPUs (>= actors' need ~20).
ulimit -u 65536 2>/dev/null || ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
export OPEN_INSTRUCT_RAY_NUM_CPUS="${OPEN_INSTRUCT_RAY_NUM_CPUS:-32}"
echo "ulimit -n=$(ulimit -n)  ulimit -u=$(ulimit -u)  OPEN_INSTRUCT_RAY_NUM_CPUS=$OPEN_INSTRUCT_RAY_NUM_CPUS"

# 2. pipeline (separate processes so each frees its GPUs before the next)
stage() { echo "########## STAGE $1  $(date) ##########"; python scripts/run.py run "$CONFIG" "${@:2}"; }
stage model --stages model --backend local || { echo "!! model stage failed"; exit 1; }
stage eval  --stages eval  --backend local || { echo "!! eval stage failed";  exit 1; }
stage judge --stages judge                  || { echo "!! judge stage failed"; exit 1; }

OUT=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['output_dir'])" 2>/dev/null)
echo "########## DONE  $(date)   results -> $OUT/summary.json ##########"
