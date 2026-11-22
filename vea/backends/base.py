"""Backend interface: submit a Job, stream progress into the shared ProgressStore.

A Job is one unit of GPU work (a training rung or a generation cell). A Backend
turns it into something runnable — an in-process call (local), a Modal function
(modal), or an sbatch submission (slurm) — and keeps the store's task row updated.
The UI never knows which backend produced a row.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..ui.progress import ProgressStore


@dataclass
class Job:
    task_id: str
    kind: str                      # "train" | "generate"
    argv: list[str] = field(default_factory=list)   # command to run (train)
    cwd: str | None = None
    env: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    total: int = 0


class Backend:
    name = "base"

    def submit(self, job: Job, store: ProgressStore) -> str:
        raise NotImplementedError

    def wait(self, handle: str, store: ProgressStore):
        raise NotImplementedError


# ---- shared log-parsing helpers (SLURM + local open-instruct) --------------

# grpo_fast.py logs episodes + a scalar reward; exact keys vary by version, so we
# match a few shapes and let the caller refine. Returns (progress, metric) or None.
_EPISODE = re.compile(r"(?:episode|global_step|step)[^0-9]{0,8}(\d+)", re.I)
_REWARD = re.compile(r"(?:reward|scores?/mean|objective/scores)[^0-9\-]{0,8}(-?\d+\.?\d*)", re.I)


def parse_train_log_line(line: str) -> tuple[int | None, float | None]:
    ep = _EPISODE.search(line)
    rw = _REWARD.search(line)
    return (int(ep.group(1)) if ep else None,
            float(rw.group(1)) if rw else None)
