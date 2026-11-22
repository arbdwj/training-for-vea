#!/bin/bash
# Sourceable env setup for the GH200 node — shared by the sbatch runners AND interactive sessions.
#
# Interactive debugging loop (NO re-queue per iteration):
#   salloc -N1 --gpus=4 --cpus-per-task=64 --mem=0 -t 02:00:00     # hold a 4-GPU node
#   # once you land on the node:
#   cd ~/vea_complexity/training-for-vea
#   source scripts/setup_env.sh
#   python scripts/run.py run configs/breadth_32b_lora.yaml --stages model --backend local
#   # error? edit (or `git pull`) and just re-run the python line — same allocation, ~mins not a queue.
#
# Safe to `source` from an interactive shell: it never `exit`s the shell (uses `return` on failure).
_fail() { echo "$1" >&2; return 1 2>/dev/null || exit 1; }

# use the env node_bootstrap.sh built (so grpo_fast's subprocess uses the same python)
source third_party/open-instruct/.venv/bin/activate || _fail "!! no venv — run scripts/node_bootstrap.sh first"
# Use HF_TOKEN if passed; else fall back to the stored ~/.cache/huggingface/token (hf auth login).
# An empty HF_TOKEN env var shadows the stored token and breaks whoami(), so unset it when blank.
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; else unset HF_TOKEN; fi
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1          # stream python prints live
# 64GB base replicated per learner leaves little headroom on a 95GB GPU; expandable segments lets the
# allocator grow/shrink without fragmenting, so transient spikes (merge/unmerge for the LoRA vLLM sync)
# don't OOM on a small alloc with memory nominally free.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# vLLM v1 engine-core msgpack-serializes rollout outputs; Olmo-3 outputs carry a torch.dtype msgpack
# can't encode. Allow the pickle fallback — safe here: engine traffic is localhost between our procs.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# --- CUDA toolkit for DeepSpeed (needs CUDA_HOME/nvcc at import) ------------
module load "${CUDA_MODULE:-cudatoolkit/24.11_12.6}" 2>/dev/null \
  || module load cuda/12.6 2>/dev/null || module load cudatoolkit 2>/dev/null || true
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
# DeepSpeed CPUAdam JIT asserts toolkit CUDA major.minor == torch's; 12.6 toolkit vs cu130 torch, no
# CUDA-13 module -> skip (CPUAdam's CUDA use is stable 12.6->13.0, runs forward-compat on the driver).
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
# CPUAdam JIT needs GCC>=9; default compute-node c++ is older. Load a modern GNU compiler, set CXX/CC.
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
# NVIDIA HPC SDK keeps math libs (curand/...) in math_libs/<ver>, NOT cuda/<ver>/lib64; DeepSpeed links
# -lcurand against CUDA_HOME/lib64. Add the math_libs dir to link-time + runtime search paths.
if [ -n "${CUDA_HOME:-}" ]; then
  _ml="$(dirname "$(dirname "$CUDA_HOME")")/math_libs/$(basename "$CUDA_HOME")/lib64"
  [ -e "$_ml/libcurand.so" ] || _ml="$(dirname "$(find /opt/nvidia/hpc_sdk -name 'libcurand.so*' 2>/dev/null | head -1)" 2>/dev/null)"
  if [ -n "$_ml" ] && [ -d "$_ml" ]; then
    export LIBRARY_PATH="$_ml:${LIBRARY_PATH:-}"; export LD_LIBRARY_PATH="$_ml:${LD_LIBRARY_PATH:-}"
  fi
fi
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/triton_$USER}"; mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true

# Ray: high open-file limit (default 4000 starves its sockets) + bounded worker count (prestarts
# ~1/CPU; can exceed the per-user process limit). Cap Ray's CPUs at >= topology need (~10/learner + 4).
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
ulimit -u 65536 2>/dev/null || ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
export OPEN_INSTRUCT_RAY_NUM_CPUS="${OPEN_INSTRUCT_RAY_NUM_CPUS:-64}"
# Cap Ray's object store (default ~200GB on GH200) so it doesn't compete with host buffers / OOM-kill.
export OPEN_INSTRUCT_RAY_OBJECT_STORE_MEMORY="${OPEN_INSTRUCT_RAY_OBJECT_STORE_MEMORY:-30000000000}"

echo "CUDA_HOME=${CUDA_HOME:-UNSET}  nvcc=$(command -v nvcc || echo none)  DS_SKIP_CUDA_CHECK=$DS_SKIP_CUDA_CHECK"
echo "CXX=${CXX:-unset} ($([ -n "${CXX:-}" ] && "$CXX" -dumpversion 2>/dev/null))  ulimit -n=$(ulimit -n)  RAY_CPUS=$OPEN_INSTRUCT_RAY_NUM_CPUS"
if [ -z "${CXX:-}" ]; then echo "!! no GCC>=9 found — CPUAdam JIT will fail. Set GCC_MODULE=<module avail gcc>."; fi
if [ -z "${CUDA_HOME:-}" ]; then echo "!! CUDA_HOME unset — DeepSpeed will fail. Set CUDA_MODULE=<module> (module avail cuda)."; fi
# IMPORTANT: end on a success status. A sourced script returns its last command's exit code, and a
# bare `[ -z X ] && echo` returns 1 whenever X is set (the healthy case) — which would abort a
# `source setup_env.sh && python ...` chain before python ever runs.
return 0 2>/dev/null || true
