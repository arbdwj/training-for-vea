#!/bin/bash
# Run the full train -> eval -> judge pipeline inside ONE SLURM allocation (GPU node).
# Submit from the repo root:
#   export HF_TOKEN=...                                   # carried into the job (--export=ALL default)
#   sbatch scripts/run_e2e.sbatch configs/breadth_32b_fast.yaml
# All stdout+stderr goes to logs/vea_e2e_<jobid>.log (the #SBATCH --output below).
#
#SBATCH --job-name=vea-e2e
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=logs/vea_e2e_%j.log
## ---- set these for your cluster (uncomment / edit) ----
## #SBATCH --partition=gpu
## #SBATCH --account=YOUR_ACCOUNT
## #SBATCH --mem=0            # all node memory (Grace LPDDR for ZeRO offload)

set -uo pipefail
CONFIG="${1:-configs/breadth_32b_fast.yaml}"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p logs

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-none}  config: $CONFIG ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "!! no GPUs visible"

# use the env node_bootstrap.sh built (so grpo_fast's subprocess uses the same python)
source third_party/open-instruct/.venv/bin/activate || { echo "!! run node_bootstrap.sh first"; exit 1; }
# Use HF_TOKEN if passed; else fall back to the stored ~/.cache/huggingface/token (hf auth login).
# An empty HF_TOKEN env var shadows the stored token and breaks whoami(), so unset it when blank.
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; else unset HF_TOKEN; fi
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1          # stream python prints to the log live (not block-buffered to file)
# vLLM v1 engine-core msgpack-serializes rollout outputs; Olmo-3 outputs carry a torch.dtype that
# msgpack can't encode (TypeError: Object of type torch.dtype is not serializable). Allow the pickle
# fallback — safe here: all engine traffic is localhost between our own trusted processes.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# --- CUDA toolkit for DeepSpeed (needs CUDA_HOME/nvcc at import) ------------
# Set CUDA_MODULE=<your cluster's module> to pin it; else we try common names.
module load "${CUDA_MODULE:-cudatoolkit/24.11_12.6}" 2>/dev/null \
  || module load cuda/12.6 2>/dev/null || module load cudatoolkit 2>/dev/null || true
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
# DeepSpeed JIT-compiles CPUAdam (optimizer offload) and hard-asserts the toolkit CUDA major.minor
# equals torch's. The cluster's newest toolkit is 12.6 but torch is cu130 and there is no CUDA-13
# module, so skip the check: CPUAdam's CUDA use is pinned-memory calls (stable 12.6->13.0) and its
# device code runs forward-compatibly on the 13.0 driver.
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/triton_$USER}"; mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true
echo "CUDA_HOME=${CUDA_HOME:-UNSET}  nvcc=$(command -v nvcc || echo none)  DS_SKIP_CUDA_CHECK=$DS_SKIP_CUDA_CHECK"
[ -z "${CUDA_HOME:-}" ] && echo "!! CUDA_HOME unset — DeepSpeed will fail. Set CUDA_MODULE=<module> (module avail cuda)."

# Ray needs a high open-file limit (default 4000 here starves its sockets -> raylet/worker EOF).
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
# ...and a bounded worker count: Ray prestarts ~1 worker/CPU; on a 64+ core node that can exceed
# the per-user process limit (raylet EAGAIN crash). Cap Ray's CPUs (>= actors' need ~20).
ulimit -u 65536 2>/dev/null || ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
export OPEN_INSTRUCT_RAY_NUM_CPUS="${OPEN_INSTRUCT_RAY_NUM_CPUS:-32}"
echo "ulimit -n=$(ulimit -n)  ulimit -u=$(ulimit -u)  OPEN_INSTRUCT_RAY_NUM_CPUS=$OPEN_INSTRUCT_RAY_NUM_CPUS"

run() { echo "=== STAGE $1  ($(date '+%H:%M:%S')) ==="; python scripts/run.py run "$CONFIG" "${@:2}"; }

run model --stages model --backend local || { echo "!! model stage failed"; exit 1; }
run eval  --stages eval  --backend local || { echo "!! eval stage failed";  exit 1; }
run judge --stages judge                  || { echo "!! judge stage failed"; exit 1; }

echo "=== DONE  ($(date '+%H:%M:%S'))  summary -> runs/$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['output_dir'])" 2>/dev/null)/summary.json ==="
