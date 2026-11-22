#!/usr/bin/env python
"""vea — run a staged VEA train -> eval -> judge pipeline from a YAML.

    python scripts/run.py run CONFIG [--dry-run] [--simulate]
                               [--stages model,eval,judge] [--backend local|modal|slurm]

--dry-run   print the planned task graph + rough ETA and exit (no GPU, no API).
--simulate  drive the live dashboard with a fake progress feed (UI smoke test).
otherwise   execute the selected stages for real (judge is API-only; eval/model
            need a GPU backend).
"""
from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from pathlib import Path

# make the repo root importable (vea package lives one level up from scripts/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vea import config as cfgmod            # noqa: E402
from vea import plan as planmod             # noqa: E402
from vea.ui.progress import Dashboard, ProgressStore, _fmt_time, render_static  # noqa: E402


def _load_plan(cfg, stages: set[str]):
    tasks = [t for t in planmod.build(cfg) if t.stage in stages]
    return tasks


def cmd_dry_run(cfg, tasks):
    store = ProgressStore(title=f"{cfg.name}  (DRY RUN)",
                          subtitle=f"stages: {' → '.join(cfg.stages)}   •   config: {cfg.config_path}")
    for t in tasks:
        store.add(t.id, t.stage, total=t.total, backend=t.backend,
                  metric_name=t.metric_name, note=t.note)
    print(render_static(store))
    eta = planmod.total_eta(tasks)
    print("\n  rough ETA (stages run sequentially):")
    grand = 0.0
    for stage in ("model", "eval", "judge"):
        if stage in eta:
            print(f"    {stage:<7} ~{_fmt_time(eta[stage])}")
            grand += eta[stage]
    if grand:
        print(f"    {'TOTAL':<7} ~{_fmt_time(grand)}   (coarse per-unit estimate)")
    # extra context
    n_models = len(planmod.model_labels(cfg))
    print(f"\n  models: {n_models}   tasks: {len(tasks)}")
    if cfg.model and cfg.model.source == "train":
        rungs = cfg.model.rungs()
        if rungs:
            print(f"  training sweep: {len(rungs)} rung(s) -> {rungs}")
    print()


def cmd_simulate(cfg, tasks):
    store = ProgressStore(title=f"{cfg.name}  (SIMULATED)",
                          subtitle=f"stages: {' → '.join(cfg.stages)}")
    for t in tasks:
        store.add(t.id, t.stage, total=max(t.total, 20), backend=t.backend,
                  metric_name=t.metric_name, note=t.note)

    def feed():
        # run stages in order; within a stage, tasks advance concurrently-ish
        for stage in ("model", "eval", "judge"):
            st = [t for t in store.snapshot() if t.stage == stage]
            for t in st:
                store.update(t.id, status="running")
            for _ in range(20):
                for t in st:
                    inc = max(1, t.total // 15)
                    nd = min(t.total, t.done + random.randint(0, inc))
                    m = None
                    if t.metric_name == "reward":
                        m = 0.2 + 0.6 * (nd / max(t.total, 1)) + random.uniform(-0.03, 0.03)
                    elif t.metric_name and "%" in t.metric_name:
                        m = 100 * (0.05 + 0.15 * random.random())
                    store.update(t.id, done=nd, metric=m)
                time.sleep(0.12)
            for t in st:
                store.update(t.id, status="done", done=t.total)

    with Dashboard(store) as dash:
        th = threading.Thread(target=feed, daemon=True)
        th.start()
        th.join()
        time.sleep(0.4)
    print("\n[simulate] done — this was a fake feed; no models or APIs were touched.")


def cmd_execute(cfg, tasks, stages, backend_override, limit=None):
    """Run the selected stages in pipeline order.

    judge runs live here (API only). model prepares data + renders the open-instruct
    command and emits an sbatch (slurm) / prints the command (modal/local) — the GPU
    run itself happens on the node/Modal. eval prints its plan + how to launch it.
    """
    from vea.ui.progress import ProgressStore
    store = ProgressStore(title=cfg.name, subtitle=f"stages: {' → '.join(sorted(stages))}")
    for t in tasks:
        store.add(t.id, t.stage, total=t.total, backend=t.backend, metric_name=t.metric_name)

    from pathlib import Path as _P
    refs = []

    # resolve model refs (train if selected; else resolve the ckpt/pretrained/closed paths
    # so a separate eval/judge process still finds the trained checkpoints)
    if cfg.model:
        from vea.stages import model as modelstage
        if "model" in stages:
            print(f"\n### STAGE model ({cfg.model.source}) ###")
            refs = modelstage.run(cfg, store, backend_override)
        else:
            refs = modelstage.run(cfg, store, backend_override, resolve_only=True)
        if refs:
            print(f"[model] model ref(s): " + ", ".join(f"{r.label}->{r.kind}:{r.ref}" for r in refs))

    if "eval" in stages and cfg.eval:
        from vea import plan as planmod
        from vea.stages import evaluate as evalstage
        print("\n### STAGE eval ###")
        be = backend_override or cfg.eval.backend.type
        labels = [r.label for r in refs] or planmod.model_labels(cfg)
        cells = evalstage.plan_cells(cfg.eval, labels)
        print(f"[eval] {len(cells)} generation cell(s) on backend '{be}'")
        if be == "local":
            rollouts_dir = _P(cfg.output_dir) / "rollouts"
            prompts_path = _resolve_eval_prompts(cfg, _P(cfg.output_dir))
            model_map = {r.label: r.ref for r in refs} or {labels[0]: (cfg.model.model if cfg.model else labels[0])}
            tp = cfg.eval.backend.extra.get("tensor_parallel", cfg.eval.backend.gpus or 1)
            for lbl, mref in model_map.items():
                print(f"[eval] generating rollouts for {lbl} ({mref}) -> {rollouts_dir}")
                evalstage.run_local(cfg.eval, mref, lbl, prompts_path, rollouts_dir, store, tp)
        elif be == "modal":
            print("[eval] launch via Modal: `modal deploy vea/backends/modal_app.py`, invoke generate(); "
                  "UI polls the vea-progress Dict.")
        else:
            print("[eval] slurm backend: run the eval on the node via evaluate.run_local (see README).")

    if "judge" in stages and cfg.judge:
        from vea.stages import judge as judgestage
        print("\n### STAGE judge ###")
        # default rollouts_dir to this run's eval output if not set
        if not cfg.judge.rollouts_dir and (_P(cfg.output_dir) / "rollouts").is_dir():
            cfg.judge.rollouts_dir = str(_P(cfg.output_dir) / "rollouts")
        if not cfg.judge.prompts:
            cfg.judge.prompts = _resolve_eval_prompts(cfg, _P(cfg.output_dir), create=False)
        judgestage.run(cfg, backend_override, limit=limit)


def _resolve_eval_prompts(cfg, out_dir, create=True):
    """Return a local eval-prompt jsonl path, building it from hf:// Dolci if needed."""
    from pathlib import Path as _P
    if not cfg.eval:
        return None
    ds0 = cfg.eval.datasets[0]
    if not ds0.path.startswith("hf://"):
        return ds0.path
    local = _P(out_dir) / "eval_prompts.jsonl"
    if local.exists() or not create:
        return str(local)
    from vea.data import prepare
    buckets = ds0.filter.get("k", [1, 2, 3, 4, 5]) if ds0.bucket_by == "k" else [1, 2, 3, 4, 5]
    info = prepare.sample_eval_prompts(ds0.path[len("hf://"):], local,
                                       buckets=tuple(buckets),
                                       n_per_bucket=ds0.n_per_bucket or 100,
                                       seed=cfg.eval.generation.seed)
    print(f"[eval] built eval prompts: {info['path']} ({info['n']} prompts)")
    return str(local)


def main():
    ap = argparse.ArgumentParser(prog="vea")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a pipeline config")
    r.add_argument("config")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--simulate", action="store_true")
    r.add_argument("--stages", default=None, help="comma list subset of model,eval,judge")
    r.add_argument("--backend", default=None, choices=["local", "modal", "slurm"])
    r.add_argument("--limit", type=int, default=None, help="judge: cap rollouts (stratified) for a cheap live test")
    args = ap.parse_args()

    if args.cmd == "run":
        try:
            cfg = cfgmod.load(args.config)
        except cfgmod.ConfigError as e:
            print(f"config error: {e}", file=sys.stderr)
            sys.exit(1)
        want = set(cfg.stages)
        if args.stages:
            want = {s.strip() for s in args.stages.split(",") if s.strip()}
            bad = want - set(cfg.stages)
            if bad:
                print(f"--stages {sorted(bad)} not present in config (has {cfg.stages})",
                      file=sys.stderr)
                sys.exit(1)
        tasks = _load_plan(cfg, want)
        if args.dry_run:
            cmd_dry_run(cfg, tasks)
        elif args.simulate:
            cmd_simulate(cfg, tasks)
        else:
            cmd_execute(cfg, tasks, want, args.backend, limit=args.limit)


if __name__ == "__main__":
    main()
