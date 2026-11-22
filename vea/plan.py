"""Turn a PipelineCfg into a concrete task graph (for --dry-run and orchestration).

A task is one unit the UI shows a row for: one training rung, one generation
cell (model × dataset × bucket), or one judge pass (judge × everything).  We also
attach a *rough* ETA so --dry-run can print a wall-clock guess; these are coarse
per-unit constants, clearly labelled, not a promise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import PipelineCfg

# rough per-unit seconds (coarse; override via backend.extra["sec_per_*"])
SEC_PER_ROLLOUT = {"local": 6.0, "modal": 6.0, "slurm": 6.0}   # ~A100 7B@16k ≈ 5.3s
SEC_PER_JUDGE = 0.2                                              # batched LLM judge, amortized
SEC_TRAIN_PER_STEP = 90.0                                       # very coarse 32B@8k guess


@dataclass
class PlannedTask:
    id: str
    stage: str
    backend: str
    total: int = 0
    metric_name: str | None = None
    eta_seconds: float | None = None
    note: str = ""
    meta: dict = field(default_factory=dict)


def _buckets(ds) -> list:
    """Concrete bucket values for an eval dataset (or [None] for whole-dataset)."""
    if ds.bucket_by and ds.filter.get(ds.bucket_by) and isinstance(ds.filter[ds.bucket_by], list):
        return list(ds.filter[ds.bucket_by])
    if ds.bucket_by == "k":
        return [1, 2, 3, 4, 5]
    return [None]


def model_labels(cfg: PipelineCfg) -> list[str]:
    """Names of the model(s) that flow into eval/judge."""
    m = cfg.model
    if m is None:
        return ["model"]
    if m.source == "train":
        rungs = m.rungs()
        if not rungs:
            return [m.base or "trained"]
        axes = m.dataset.sweep_axes() if m.dataset else []
        if axes:
            key = axes[0][0]
            return [f"{key}={r.get(key)}" for r in rungs]
        return ["trained"]
    return [m.model or m.source]


def build(cfg: PipelineCfg) -> list[PlannedTask]:
    tasks: list[PlannedTask] = []
    models = model_labels(cfg)

    # STAGE 1 — model (one task per training rung)
    if cfg.model and cfg.model.source == "train":
        be = cfg.model.backend.type
        steps = int(cfg.model.params.get("max_steps")
                    or cfg.model.params.get("num_training_steps")
                    or _steps_from_episodes(cfg.model.params))
        sec = cfg.model.backend.extra.get("sec_per_step", SEC_TRAIN_PER_STEP)
        for lbl in models:
            tasks.append(PlannedTask(
                id=f"train:{lbl}", stage="model", backend=be, total=steps or 0,
                metric_name="reward", eta_seconds=(steps * sec if steps else None),
                note=f"{cfg.model.framework} {cfg.model.base}",
                meta={"rung": lbl}))
    elif cfg.model:
        tasks.append(PlannedTask(id=f"use:{models[0]}", stage="model",
                                 backend=cfg.model.backend.type, total=0,
                                 note=f"{cfg.model.source}: {cfg.model.model}"))

    # STAGE 2 — eval (model × dataset × bucket)
    if cfg.eval:
        be = cfg.eval.backend.type
        R = cfg.eval.generation.rollouts
        sec = cfg.eval.backend.extra.get("sec_per_rollout", SEC_PER_ROLLOUT.get(be, 6.0))
        for lbl in models:
            for ds in cfg.eval.datasets:
                for b in _buckets(ds):
                    n = ds.n_per_bucket if b is not None else (ds.n or 100)
                    total = int(n) * R
                    blab = f"/{ds.bucket_by}{b}" if b is not None else ""
                    tasks.append(PlannedTask(
                        id=f"gen:{lbl}:{ds.name}{blab}", stage="eval", backend=be,
                        total=total, eta_seconds=total * sec,
                        note=f"{n}×{R} @ {cfg.eval.generation.max_tokens}tok",
                        meta={"model": lbl, "dataset": ds.name, "bucket": b}))

    # STAGE 3 — judge (one task per judge, over all rollouts of all models)
    if cfg.judge:
        total_rollouts = _count_rollouts(cfg, models)
        for j in cfg.judge.judges:
            be = "local" if j.kind == "llm" else "local"
            eta = total_rollouts * SEC_PER_JUDGE if j.kind == "llm" else total_rollouts * 0.001
            tasks.append(PlannedTask(
                id=f"judge:{j.name}", stage="judge", backend=be, total=total_rollouts,
                metric_name="VEA%" if j.kind == "llm" else "sat%", eta_seconds=eta,
                note=f"{j.kind}" + (f" {j.model}" if j.model else ""),
                meta={"judge": j.name, "kind": j.kind}))
    return tasks


def _steps_from_episodes(params: dict) -> int:
    ep = params.get("total_episodes")
    g = params.get("num_samples_per_prompt_rollout") or params.get("num_generations") or 1
    u = params.get("num_unique_prompts_rollout") or params.get("per_device_train_batch_size") or 1
    if ep and g and u:
        return int(ep // (g * u))
    return 0


def _count_rollouts(cfg: PipelineCfg, models: list[str]) -> int:
    if not cfg.eval:
        return 0
    R = cfg.eval.generation.rollouts
    total = 0
    for ds in cfg.eval.datasets:
        for b in _buckets(ds):
            n = ds.n_per_bucket if b is not None else (ds.n or 100)
            total += int(n) * R
    return total * len(models)


def total_eta(tasks: list[PlannedTask]) -> dict[str, float]:
    """Per-stage ETA (seconds), summed within a stage (stages run sequentially)."""
    out: dict[str, float] = {}
    for t in tasks:
        if t.eta_seconds:
            out[t.stage] = out.get(t.stage, 0.0) + t.eta_seconds
    return out
