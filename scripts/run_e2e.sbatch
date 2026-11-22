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
#SBATCH --mem=0             # all node memory (Grace LPDDR for ZeRO/CPUAdam offload) — essential

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
# 64GB base replicated per learner leaves little headroom on a 95GB GPU; expandable segments lets the
# allocator grow/shrink without fragmenting, so transient spikes (merge/unmerge for the LoRA vLLM sync)
# don't OOM on a 50MB alloc with memory nominally free.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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

# DeepSpeed/torch JIT ops (CPUAdam) need GCC>=9, but the default compute-node c++ is older. Load a
# modern GNU compiler and point the JIT build at it via CXX/CC (both torch and DeepSpeed honor them).
# gcc-native/12.3 stays within nvcc 12.6's supported GCC range for any op that does use nvcc.
module load "${GCC_MODULE:-gcc-native/12.3}" 2>/dev/null \
  || module load gcc-native/13.2 2>/dev/null || module load gcc 2>/dev/null || true
if [ -z "${CXX:-}" ]; then for g in g++-12 g++-13 g++ g++-14 g++-11 g++-10; do
    p="$(command -v $g 2>/dev/null)" || continue
    v="$("$p" -dumpversion 2>/dev/null | cut -d. -f1)"
    if [ -n "$v" ] && [ "$v" -ge 9 ] 2>/dev/null; then export CXX="$p"; break; fi
  done; fi
if [ -z "${CC:-}" ]; then for c in gcc-12 gcc-13 gcc gcc-14 gcc-11 gcc-10; do
    p="$(command -v $c 2>/dev/null)" || continue
    v="$("$p" -dumpversion 2>/dev/null | cut -d. -f1)"
    if [ -n "$v" ] && [ "$v" -ge 9 ] 2>/dev/null; then export CC="$p"; break; fi
  done; fi
# NVIDIA HPC SDK keeps math libs (curand/cublas/...) in math_libs/<ver>, NOT cuda/<ver>/lib64, but
# DeepSpeed links -lcurand against CUDA_HOME/lib64 -> "cannot find -lcurand". Add the math_libs dir
# to the link-time (LIBRARY_PATH) and runtime (LD_LIBRARY_PATH) search paths.
if [ -n "${CUDA_HOME:-}" ]; then
  _ml="$(dirname "$(dirname "$CUDA_HOME")")/math_libs/$(basename "$CUDA_HOME")/lib64"
  [ -e "$_ml/libcurand.so" ] || _ml="$(dirname "$(find /opt/nvidia/hpc_sdk -name 'libcurand.so*' 2>/dev/null | head -1)" 2>/dev/null)"
  if [ -n "$_ml" ] && [ -d "$_ml" ]; then
    export LIBRARY_PATH="$_ml:${LIBRARY_PATH:-}"; export LD_LIBRARY_PATH="$_ml:${LD_LIBRARY_PATH:-}"
    echo "math_libs=$_ml"
  else
    echo "!! libcurand not found under /opt/nvidia/hpc_sdk — CPUAdam link will fail."
  fi
fi
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/triton_$USER}"; mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true
echo "CUDA_HOME=${CUDA_HOME:-UNSET}  nvcc=$(command -v nvcc || echo none)  DS_SKIP_CUDA_CHECK=$DS_SKIP_CUDA_CHECK"
echo "CXX=${CXX:-unset} ($([ -n "${CXX:-}" ] && "$CXX" -dumpversion 2>/dev/null))  CC=${CC:-unset}"
[ -z "${CXX:-}" ] && echo "!! no GCC>=9 found — CPUAdam JIT will fail. Set GCC_MODULE=<module avail gcc>."
[ -z "${CUDA_HOME:-}" ] && echo "!! CUDA_HOME unset — DeepSpeed will fail. Set CUDA_MODULE=<module> (module avail cuda)."

# Ray needs a high open-file limit (default 4000 here starves its sockets -> raylet/worker EOF).
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
# ...and a bounded worker count: Ray prestarts ~1 worker/CPU; on a 64+ core node that can exceed
# the per-user process limit (raylet EAGAIN crash). Cap Ray's CPUs (>= actors' need ~20).
ulimit -u 65536 2>/dev/null || ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
# Must be >= the full topology's CPU request: ~10/learner + dataprep 2 + manager 1 + vLLM 1
# (1 learner -> 14, 3 learners -> 34). 64 covers the 4-GPU/3-learner config and is fork-safe on a
# compute node (ulimit -u ~487k). Raise if a run logs "full topology requires at least CPU=N".
export OPEN_INSTRUCT_RAY_NUM_CPUS="${OPEN_INSTRUCT_RAY_NUM_CPUS:-64}"
# Cap Ray's object store (default ~30% of RAM = ~200GB on a GH200 node) so it doesn't compete with
# the CPU-offloaded optimizer's host buffers and trip the OOM killer. 30GB is plenty for rollouts.
export OPEN_INSTRUCT_RAY_OBJECT_STORE_MEMORY="${OPEN_INSTRUCT_RAY_OBJECT_STORE_MEMORY:-30000000000}"
echo "ulimit -n=$(ulimit -n)  ulimit -u=$(ulimit -u)  OPEN_INSTRUCT_RAY_NUM_CPUS=$OPEN_INSTRUCT_RAY_NUM_CPUS  RAY_OSM=$OPEN_INSTRUCT_RAY_OBJECT_STORE_MEMORY"

run() { echo "=== STAGE $1  ($(date '+%H:%M:%S')) ==="; python scripts/run.py run "$CONFIG" "${@:2}"; }

run model --stages model --backend local || { echo "!! model stage failed"; exit 1; }
run eval  --stages eval  --backend local || { echo "!! eval stage failed";  exit 1; }
run judge --stages judge                  || { echo "!! judge stage failed"; exit 1; }

echo "=== DONE  ($(date '+%H:%M:%S'))  summary -> runs/$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['output_dir'])" 2>/dev/null)/summary.json ==="
