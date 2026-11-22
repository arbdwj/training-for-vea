"""STAGE 1 — produce a ModelRef (or several, for a sweep).

source: train       -> prepare per-rung data, render the grpo_fast.py command, and
                       submit it to the chosen backend (slurm/modal/local).
source: pretrained  -> ModelRef to an HF id (training skipped).
source: closed      -> ModelRef to an API model id (eval generates via API).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import PipelineCfg
from ..train import open_instruct as oi
from ..ui.progress import ProgressStore


@dataclass
class ModelRef:
    kind: str          # checkpoint | hf | api
    ref: str           # path / HF id / API model id
    label: str         # short label used downstream (e.g. "k=5")


def _rung_label(ms, rung: dict) -> str:
    axes = ms.dataset.sweep_axes() if ms.dataset else []
    if axes:
        key = axes[0][0]
        return f"{key}={rung.get(key)}"
    return "trained"


def plan_training(cfg: PipelineCfg):
    """Return [(label, filter, argv, warnings, data_path, out_dir)] for each rung."""
    ms = cfg.model
    out_root = Path(cfg.output_dir)
    jobs = []
    for rung in ms.rungs():
        label = _rung_label(ms, rung)
        safe = label.replace("=", "")
        data_path = str(out_root / "data" / f"{ms.dataset.name.split('/')[-1]}_{safe}.jsonl")
        out_dir = str(out_root / "ckpt" / safe)
        exp = f"{cfg.name}_{safe}"
        argv, warns = oi.render(ms, data_path=data_path, output_dir=out_dir, exp_name=exp)
        jobs.append((label, rung, argv, warns, data_path, out_dir))
    return jobs


def run(cfg: PipelineCfg, store: ProgressStore | None = None,
        backend_override: str | None = None, prepare_data: bool = True) -> list[ModelRef]:
    ms = cfg.model
    if ms is None:
        return []
    if ms.source == "pretrained":
        return [ModelRef("hf", ms.model, "pretrained")]
    if ms.source == "closed":
        return [ModelRef("api", ms.model, "closed")]

    # source == train
    from ..backends.slurm import SlurmBackend
    be_type = backend_override or ms.backend.type
    refs: list[ModelRef] = []
    plans = plan_training(cfg)
    for label, rung, argv, warns, data_path, out_dir in plans:
        tid = f"train:{label}"
        if store is not None:
            store.update(tid, note=f"{ms.framework} {ms.base}")
        for w in warns:
            print(f"[train:{label}] note: {w}")

        if prepare_data:
            _prepare_rung_data(ms, rung, data_path)

        if be_type == "slurm":
            be = SlurmBackend(gpus=ms.backend.gpus, nodes=ms.backend.nodes,
                              time_limit=ms.backend.extra.get("time_limit", "24:00:00"))
            from ..backends.base import Job
            job = Job(task_id=tid, kind="train", argv=argv, cwd=be.oi_dir,
                      meta={"out_dir": cfg.output_dir}, total=_steps(ms))
            handle = be.submit(job, store or ProgressStore())
            print(f"[train:{label}] slurm handle: {handle}")
        elif be_type == "modal":
            print(f"[train:{label}] modal backend: deploy vea/backends/modal_app.py and invoke "
                  f"with argv; see README. Command:\n  {oi.command_string(argv)}")
        else:  # local
            print(f"[train:{label}] local run command (needs a GPU + open-instruct env):\n"
                  f"  cd {Path('third_party/open-instruct').resolve()} && {oi.command_string(argv)}")
        refs.append(ModelRef("checkpoint", out_dir, label))
    return refs


def _steps(ms) -> int:
    ep = oi.derive_total_episodes(ms.params) or 0
    g = int(ms.params.get("num_samples_per_prompt_rollout", 8))
    u = int(ms.params.get("num_unique_prompts_rollout", 32))
    return ep // (g * u) if (g and u) else 0


def _prepare_rung_data(ms, rung: dict, data_path: str):
    try:
        from ..data import prepare
        info = prepare.prepare_rung(ms.dataset.name, rung, data_path,
                                    verifier=ms.dataset.verifier,
                                    messages_key=ms.dataset.messages_key,
                                    ground_truths_key=ms.dataset.ground_truths_key)
        print(f"[data] {ms.dataset.name} {rung} -> {info['path']} ({info['n']} rows)")
    except Exception as e:  # datasets not installed / offline
        print(f"[data] SKIPPED prep for {rung} ({type(e).__name__}: {e}). "
              f"Prepare it on the node before submitting: {data_path}")
