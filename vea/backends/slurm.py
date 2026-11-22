"""SLURM backend for a single 4×GH200 node — net-new (open-instruct has no SLURM path).

Renders an `sbatch` script that starts a single-node Ray head and runs the vendored
`open_instruct/grpo_fast.py` with the flags our renderer produced, plus a DeepSpeed
ZeRO-3 (+CPU offload) config. On a login node it submits with `sbatch` and tails the
job log into the ProgressStore; off-cluster (e.g. this Mac) it just writes the files
and prints the command, so the scripts can be inspected and committed.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)

from ..ui.progress import ProgressStore
from .base import Backend, Job, parse_train_log_line

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --gpus={gpus}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time_limit}
#SBATCH --output={log_path}

set -euo pipefail

# --- environment (edit for your GH200 node) --------------------------------
# module load cuda/13.0 || true
# source {oi_dir}/.venv/bin/activate   # a uv/venv with open-instruct deps (aarch64: no flash-attn)
export HF_TOKEN="${{HF_TOKEN:-}}"
export WANDB_MODE="${{WANDB_MODE:-offline}}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

cd {oi_dir}

# --- single-node Ray head (grpo_fast.py uses Ray for actor/learner) --------
ray stop --force || true
ray start --head --num-gpus={gpus} --disable-usage-stats
sleep 5

# --- GRPO training ---------------------------------------------------------
{command}

ray stop --force || true
"""

DEEPSPEED_ZERO3_OFFLOAD = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "offload_param": {"device": "cpu", "pin_memory": True},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "bf16": {"enabled": True},
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "comment": "ZeRO-3 + CPU offload — GH200 Grace CPU (~480GB LPDDR) holds optimizer+params",
}


class SlurmBackend(Backend):
    name = "slurm"

    def __init__(self, gpus: int = 4, nodes: int = 1, cpus: int = 64,
                 time_limit: str = "24:00:00", oi_dir: str = "third_party/open-instruct",
                 partition: str | None = None):
        self.gpus = gpus
        self.nodes = nodes
        self.cpus = cpus
        self.time_limit = time_limit
        self.oi_dir = str(Path(oi_dir).resolve())

    def render_sbatch(self, job: Job, out_dir: Path) -> dict:
        """Write the sbatch script + deepspeed config; return their paths."""
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"{_safe(job.task_id)}.log"
        ds_path = out_dir / "deepspeed_zero3_offload.json"
        ds_path.write_text(json.dumps(DEEPSPEED_ZERO3_OFFLOAD, indent=2))

        argv = list(job.argv)
        if "--deepspeed_config" not in argv and "--deepspeed_config_file" not in argv:
            argv += ["--deepspeed_config_file", str(ds_path)]
        command = " \\\n  ".join(_group(argv))

        script = SBATCH_TEMPLATE.format(
            job_name=_safe(job.task_id), nodes=self.nodes, gpus=self.gpus,
            cpus=self.cpus, time_limit=self.time_limit, log_path=str(log_path),
            oi_dir=self.oi_dir, command=command)
        sh_path = out_dir / f"{_safe(job.task_id)}.sbatch"
        sh_path.write_text(script)
        return {"sbatch": str(sh_path), "deepspeed": str(ds_path), "log": str(log_path)}

    def submit(self, job: Job, store: ProgressStore) -> str:
        out_dir = Path(job.meta.get("out_dir", "runs")) / "slurm"
        files = self.render_sbatch(job, out_dir)
        store.update(job.task_id, total=job.total, note="sbatch written")
        if shutil.which("sbatch") is None:
            store.update(job.task_id, status="pending",
                         note=f"off-cluster: run `sbatch {files['sbatch']}` on the node")
            print(f"[slurm] no sbatch here; wrote {files['sbatch']}\n"
                  f"        submit on the GH200 node: sbatch {files['sbatch']}")
            return files["sbatch"]
        res = subprocess.run(["sbatch", files["sbatch"]], capture_output=True, text=True)
        jobid = res.stdout.strip().split()[-1] if res.returncode == 0 else ""
        store.update(job.task_id, status="running", note=f"slurm {jobid}")
        threading.Thread(target=self._tail, args=(files["log"], job, store), daemon=True).start()
        return jobid

    def _tail(self, log_path: str, job: Job, store: ProgressStore):
        p = Path(log_path)
        while not p.exists():
            time.sleep(2)
        with open(p) as f:
            while True:
                line = f.readline()
                if not line:
                    time.sleep(1)
                    continue
                step, reward = parse_train_log_line(line)
                if step is not None:
                    store.update(job.task_id, done=step)
                if reward is not None:
                    store.update(job.task_id, metric=reward)


def _group(argv: list[str]) -> list[str]:
    """Group flag + following values for a readable multi-line command."""
    out, i = [], 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            vals = [tok]; j = i + 1
            while j < len(argv) and not argv[j].startswith("--"):
                vals.append(argv[j]); j += 1
            out.append(" ".join(vals)); i = j
        else:
            out.append(tok); i += 1
    return out
