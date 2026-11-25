#!/bin/bash
# Train the breadth sweep with ONE PROCESS PER RUNG (process isolation).
#
# WHY one process per rung:
#   The local backend runs every sweep rung inside ONE long-lived run.py process
#   (vea/stages/model.py:_run_local_train loops subprocess.Popen). At a rung
#   boundary the next rung starts loading a fresh bf16 copy of the 32B THROUGH
#   HOST RAM for 3 learners + the vLLM engine while the previous rung's Ray
#   workers / torch allocator have not yet returned theirs to the OS -> host
#   memory overflows -> the Linux OOM killer SIGKILLs the process (signal 9, so
#   NO Python traceback; GPUs freed instantly). That is exactly what killed the
#   first attempt: k=1 trained + saved its adapter fine, then the process died
#   the instant it echoed the k=2 command. k=1 only survived because nothing
#   precedes the first rung.
#   Running each rung as its own `srun` step means every rung is a separate
#   process that FULLY EXITS (releasing all host + GPU memory) before the next
#   one starts -> the fatal overlap cannot happen, and a rung that does die
#   cannot cascade into the rungs after it (the loop records its exit and
#   continues).
#
# Submit from the repo root (no practical time cap — it runs until the sweep is done):
#   sbatch scripts/run_breadth_rungs.sbatch                 # default: rungs 2 3 4 5 (k=1 already banked)
#   sbatch scripts/run_breadth_rungs.sbatch "1 2 3 4 5"     # full fresh sweep
#   RUN_K="3 5" sbatch scripts/run_breadth_rungs.sbatch     # arbitrary subset
# If SLURM rejects the 24h request ("time limit exceeds partition limit"),
# override on the CLI, e.g.:  sbatch --time=12:00:00 scripts/run_breadth_rungs.sbatch
#
# Driver log:  logs/vea_rungs_<jobid>.log        Per-rung log:  ~/vea_breadth_k<K>.log
#
#SBATCH --job-name=vea-breadth
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --mem=0                 # all node memory — essential for the 32B load
#SBATCH --output=logs/vea_rungs_%j.log

set -uo pipefail
CONFIG="${CONFIG:-configs/breadth_32b_lora.yaml}"
RUNGS="${1:-${RUN_K:-2 3 4 5}}"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p logs

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-none}  config $CONFIG  rungs: [$RUNGS]  start $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "!! no GPUs visible"

declare -A RC
for K in $RUNGS; do
  LOG="$HOME/vea_breadth_k${K}.log"
  echo ">>> rung k=$K  START $(date -u)  -> $LOG"
  # One fresh process per rung. All the cluster env (venv, CUDA, Ray caps, ulimits)
  # is re-sourced inside the step, and HF offline avoids the dataset 429 rate-limit.
  srun --gpus=4 --cpus-per-task=64 bash -c "
    cd '$PWD' &&
    source scripts/setup_env.sh &&
    [ -f scripts/storage_env.sh ] && source scripts/storage_env.sh &&   # HF cache/runs on scratch (quota)
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 &&
    export PYTHONPATH=\"$PWD/third_party/open-instruct:\${PYTHONPATH:-}\" &&   # make the vendored open_instruct package importable; the editable install broke when the repo was moved. Path is repo-root-relative so this is move-proof.
    export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True &&   # defragment the CUDA allocator -> fixes the marginal OOM in backward() that killed k2/k4 at ~step 75 (tiny 256MiB allocs failing with ~25GB nominal free = fragmentation, not a true ceiling; k3/k5 survived the identical config)
    export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID:-\$\$}_k$K && mkdir -p /tmp/ray_\${SLURM_JOB_ID:-\$\$}_k$K &&     # node-local Ray scratch we create ourselves -> fixes k1 dying at ray.init on PermissionError: /local/user/<id> (node-local dir not writable at first srun step)
    python scripts/run.py run '$CONFIG' --stages model --only-k $K
  " > "$LOG" 2>&1
  RC[$K]=$?
  echo ">>> rung k=$K  EXIT ${RC[$K]}  $(date -u)"
done

echo "=== SWEEP DONE $(date -u) — adapter audit ==="
OUT=$(grep -E '^output_dir:' "$CONFIG" | head -1 | awk '{print $2}')   # audit the CONFIG's real output_dir, not a hardcoded one (was runs/breadth_32b_lora -> audited the wrong dir for breadth_32b_100)
for K in $RUNGS; do
  CKPT=$(ls ${OUT:-runs/breadth_32b_lora}/ckpt/k${K}/*/step_*/adapter_model.safetensors 2>/dev/null | tail -1)
  echo "  k=$K  exit=${RC[$K]:-?}  adapter=${CKPT:-MISSING}"
done
