#!/bin/bash
# Eval + judge the breadth sweep in ONE submission, each stage its own srun (process
# isolation), so a stage fully exits and frees all host+GPU memory before the next loads.
#
# Pipeline:
#   1) MERGE  — fold each trained LoRA adapter into the frozen 32B base -> a full model
#               vLLM can load (runs/<name>/merged/k1..k5). One GPU; one rung at a time.
#   2) EVAL   — load each merged model at TP=4 and generate rollouts on the constraint-
#               stacking probe (datasets/vshift/if.jsonl, k=1..5) -> runs/<name>/rollouts/.
#   3) JUDGE  — Nemotron-120B-FP8 auditor at TP=4 scores every rollout for VEA, grouped by
#               (model, k) -> runs/<name>/summary.json  == the VEA-vs-constraint-breadth table.
#
# WHY separate srun steps: eval's 32B vLLM engine and the judge's 120B engine must never
# co-reside; and if the judge model isn't cached, steps 1-2 (the expensive GPU work) are
# already banked, so you can judge later without regenerating rollouts.
#
# Submit from the repo root (no practical time cap):
#   sbatch scripts/run_eval_judge.sbatch                       # default config
#   sbatch scripts/run_eval_judge.sbatch configs/breadth_32b_lora.yaml
#
# Driver log: logs/vea_evaljudge_<jobid>.log   Per-stage: ~/vea_{merge,eval,judge}.log
#
#SBATCH --job-name=vea-evaljudge
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --output=logs/vea_evaljudge_%j.log

set -uo pipefail
CONFIG="${1:-${CONFIG:-configs/breadth_32b_lora.yaml}}"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p logs

# env prelude re-sourced inside every step (venv, CUDA, Ray caps, storage, HF offline)
PRELUDE="cd '$PWD' && source scripts/setup_env.sh && { [ -f scripts/storage_env.sh ] && source scripts/storage_env.sh; } && export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false"

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-none}  config $CONFIG  start $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "!! no GPUs visible"

# ---- STAGE 1: merge adapters -> full models -------------------------------------------
MLOG="$HOME/vea_merge.log"
echo ">>> MERGE  START $(date -u)  -> $MLOG"
srun --gpus=4 --cpus-per-task=64 bash -c "$PRELUDE && python -m vea.eval.merge '$CONFIG'" > "$MLOG" 2>&1
MRC=$?
echo ">>> MERGE  EXIT $MRC  $(date -u)"
if [ $MRC -ne 0 ]; then
  echo "!! merge failed (rc=$MRC) — see $MLOG. Not all adapters merged; eval would be incomplete."
  echo "   (fix the failing rung, then resubmit — merge is idempotent and skips finished rungs.)"
fi
# Abort before eval if NOTHING merged — eval can't load a bare adapter dir, so running it
# would just burn a node. (If SOME merged, continue and eval what's present.)
NMERGED=$(ls -d runs/*/merged/*/ 2>/dev/null | wc -l | tr -d ' ')
if [ "${NMERGED:-0}" -eq 0 ]; then
  echo "=== PIPELINE DONE $(date -u)  (merge=$MRC eval=skipped judge=skipped) ==="
  echo "!! no merged models produced — aborting before eval. See $MLOG for the merge error."
  exit 1
fi
echo ">>> $NMERGED merged model(s) present — proceeding to eval"

# ---- STAGE 2: generate rollouts (32B merged models at TP=4) ---------------------------
ELOG="$HOME/vea_eval.log"
echo ">>> EVAL   START $(date -u)  -> $ELOG"
srun --gpus=4 --cpus-per-task=64 bash -c "$PRELUDE && python scripts/run.py run '$CONFIG' --stages eval" > "$ELOG" 2>&1
ERC=$?
echo ">>> EVAL   EXIT $ERC  $(date -u)"

# ---- STAGE 3: judge for VEA (120B auditor at TP=4) ------------------------------------
JLOG="$HOME/vea_judge.log"
# Pre-check the judge model is in the offline HF cache; if not, bank rollouts and stop
# cleanly with instructions (don't burn a vLLM startup just to fail).
JUDGE_MODEL="$(python -c "import sys;from vea import config as c;cfg=c.load('$CONFIG');print(cfg.judge.judges[0].model if cfg.judge else '')" 2>/dev/null)"
CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
SLUG="models--$(echo "$JUDGE_MODEL" | sed 's#/#--#g')"
if [ -n "$JUDGE_MODEL" ] && [ -d "$CACHE/$SLUG" ]; then
  echo ">>> JUDGE  START $(date -u)  model=$JUDGE_MODEL  -> $JLOG"
  srun --gpus=4 --cpus-per-task=64 bash -c "$PRELUDE && python scripts/run.py run '$CONFIG' --stages judge" > "$JLOG" 2>&1
  JRC=$?
  echo ">>> JUDGE  EXIT $JRC  $(date -u)"
else
  JRC=127
  echo ">>> JUDGE  SKIPPED — judge model not in offline cache"
  echo "     model : ${JUDGE_MODEL:-<none>}"
  echo "     looked: $CACHE/$SLUG"
  echo "   Rollouts are banked in runs/*/rollouts — judge later once the model is available:"
  echo "     # option A: pre-download on the login node (has network), then resubmit this script"
  echo "     python -c \"from huggingface_hub import snapshot_download as s; s('$JUDGE_MODEL')\""
  echo "     # option B: judge with an API model instead (edit judge.model to e.g. claude-* + set ANTHROPIC_API_KEY)"
  echo "     python scripts/run.py run '$CONFIG' --stages judge --limit 200   # cheap first look"
fi

# ---- summary --------------------------------------------------------------------------
echo "=== PIPELINE DONE $(date -u)  (merge=$MRC eval=$ERC judge=$JRC) ==="
echo "--- merged models ---";  ls -d runs/*/merged/*/ 2>/dev/null || echo "  (none)"
echo "--- rollout cells ---";  ls runs/*/rollouts/*.jsonl 2>/dev/null | wc -l | sed 's/^/  files: /'
echo "--- summary ---";        ls runs/*/summary.json 2>/dev/null && echo "  (VEA-vs-breadth table is in summary.json)" || echo "  (no summary yet — judge did not complete)"
