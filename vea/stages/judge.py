"""STAGE 3 — judge rollouts for VEA (and optional per-constraint satisfaction).

Reads rollouts from a directory of `{model}__{domain}_k{N}.jsonl` files (records
{prompt_id, completion, k}), runs each configured judge, and reports the per-rollout
VEA rate with Wilson CIs, grouped by `group_by` (default [model], e.g. [model, k]).

Validation target: on the existing 7B-IF rollouts this must reproduce
DPO-7B 4.8% / Think-7B 7.6% (sonnet-4.5, temp-1.0/16k, 32B-matched).
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
from pathlib import Path

from ..config import PipelineCfg
from ..judges import clients
from ..ui.progress import Dashboard, ProgressStore

_CELL = re.compile(r"^(?P<model>.+?)__(?P<domain>.+?)_k(?P<k>\d+)$")


def _repo_root(cfg: PipelineCfg) -> Path:
    # configs/<name>.yaml  ->  repo root is two parents up
    return cfg.config_path.parents[1] if cfg.config_path else Path.cwd()


def _load_prompt_map(prompts_path: str | None) -> dict[str, str]:
    pm: dict[str, str] = {}
    if not prompts_path:
        return pm
    for p in str(prompts_path).split(","):
        p = p.strip()
        if not p or not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            x = json.loads(line)
            pid = x.get("prompt_id") or x.get("id")
            if pid is not None and "prompt" in x:
                pm[pid] = x["prompt"]
    return pm


def _read_rollouts(rollouts_dir: str):
    """Yield (model, domain, k, record) for every rollout line."""
    for fp in sorted(glob.glob(os.path.join(rollouts_dir, "*.jsonl"))):
        base = os.path.basename(fp)[:-6]
        m = _CELL.match(base)
        if not m:
            continue
        model, domain, k = m["model"], m["domain"], int(m["k"])
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            yield model, domain, k, json.loads(line)


def run(cfg: PipelineCfg, backend_override=None, limit: int | None = None):
    js = cfg.judge
    if js is None:
        raise SystemExit("config has no judge stage")
    rollouts_dir = js.rollouts_dir or os.path.join(cfg.output_dir, "rollouts")
    if not os.path.isdir(rollouts_dir):
        raise SystemExit(f"rollouts dir not found: {rollouts_dir}\n"
                         "  (run the eval stage first, or set judge.rollouts_dir)")
    repo_root = _repo_root(cfg)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_map = _load_prompt_map(js.prompts)
    # materialize rollouts once (id-stable)
    rolls = []
    for i, (model, domain, k, rec) in enumerate(_read_rollouts(rollouts_dir)):
        rolls.append({"cid": f"r{i}", "model": model, "domain": domain,  # cid: ^[a-zA-Z0-9_-]{1,64}$
                      "k": k, "prompt_id": rec.get("prompt_id"),
                      "completion": rec.get("completion", "")})
    if not rolls:
        raise SystemExit(f"no rollouts parsed from {rollouts_dir} "
                         "(expected files named {model}__{domain}_k{N}.jsonl)")
    if limit and limit < len(rolls):
        step = len(rolls) / limit                         # stratified across cells
        rolls = [rolls[int(i * step)] for i in range(limit)]
        print(f"[--limit {limit}] stratified subsample of rollouts")
    print(f"loaded {len(rolls)} rollouts from {rollouts_dir}")
    if prompt_map:
        miss = sum(1 for r in rolls if r["prompt_id"] not in prompt_map)
        print(f"prompt map: {len(prompt_map)} prompts ({miss} rollouts without a matched prompt)")

    store = ProgressStore(title=f"{cfg.name} · judge",
                          subtitle=f"{len(rolls)} rollouts · {rollouts_dir}")
    for j in js.judges:
        store.add(f"judge:{j.name}", "judge", total=len(rolls), backend="local",
                  metric_name="VEA%" if j.kind == "llm" else "sat%", note=j.kind)

    all_summaries = {}
    with Dashboard(store):
        for j in js.judges:
            tid = f"judge:{j.name}"
            if j.kind == "llm":
                recs = _run_llm_judge(j, rolls, prompt_map, repo_root, store, tid)
            elif j.kind == "verifier":
                store.update(tid, status="error", note="verifier not yet wired")
                print(f"\n[judge:{j.name}] verifier judge not yet implemented "
                      "(needs open-instruct IFEvalG) — skipping.")
                continue
            else:
                continue
            store.update(tid, status="done", done=len(rolls))
            summ = _aggregate(recs, js.group_by)
            all_summaries[j.name] = summ
            json.dump(recs, open(out_dir / f"judged_{j.name}.json", "w"))

    _print_summaries(all_summaries, js.group_by)
    json.dump(all_summaries, open(out_dir / "summary.json", "w"), indent=2)
    print(f"\nsummary -> {out_dir / 'summary.json'}")
    return all_summaries


def _run_llm_judge(j, rolls, prompt_map, repo_root, store, tid):
    prompt = clients.load_prompt(j.prompt, repo_root)
    judge = clients.make_judge(j.model, provider=j.provider, **j.opts)
    reqs = [clients.JudgeRequest(
                custom_id=r["cid"], system=prompt.system,
                user=prompt.render_user(prompt_map.get(r["prompt_id"], ""), r["completion"]))
            for r in rolls]
    store.update(tid, status="running")
    prov = getattr(judge, "__class__").__name__
    print(f"[judge:{j.name}] judging {len(reqs)} rollouts with {j.model} ({prov})")
    results = judge.judge_all(reqs, progress_cb=lambda done, total: store.update(tid, done=done))
    byid = {r["cid"]: r for r in rolls}
    recs = []
    for cid, text, ok in results:
        base = byid.get(cid, {})
        score, quote, purposes = clients.parse_score_quote(text) if ok else (0, None, [])
        recs.append({"model": base.get("model"), "domain": base.get("domain"),
                     "k": base.get("k"), "vea": score, "quote": quote,
                     "purposes": purposes, "ok": ok})
    # live VEA% metric
    ok_recs = [r for r in recs if r["ok"]]
    if ok_recs:
        store.update(tid, metric=100 * sum(r["vea"] for r in ok_recs) / len(ok_recs))
    return recs


def _aggregate(recs, group_by):
    """Per-rollout VEA rate overall-per-model and broken down by group_by keys."""
    out = {}
    models = sorted({r["model"] for r in recs if r["ok"]})
    extra_keys = [g for g in group_by if g != "model"]
    for mk in models:
        mrecs = [r for r in recs if r["model"] == mk and r["ok"]]
        k_ok = sum(r["vea"] for r in mrecs)
        p, lo, hi = clients.wilson(k_ok, len(mrecs))
        entry = {"pct": round(p, 2), "ci": [round(lo, 2), round(hi, 2)], "n": len(mrecs)}
        for gk in extra_keys:
            buckets = collections.defaultdict(lambda: [0, 0])
            for r in mrecs:
                buckets[r.get(gk)][1] += 1
                buckets[r.get(gk)][0] += r["vea"]
            entry[f"by_{gk}"] = {
                str(b): round(100 * v[0] / v[1], 2) if v[1] else 0.0
                for b, v in sorted(buckets.items(), key=lambda x: (x[0] is None, x[0]))}
        out[mk] = entry
    return out


def _print_summaries(all_summaries, group_by):
    extra = [g for g in group_by if g != "model"]
    for jname, summ in all_summaries.items():
        print(f"\n=== judge '{jname}' · per-rollout VEA ===")
        for mk, e in summ.items():
            print(f"  {mk:<16} {e['pct']:5.2f}% [{e['ci'][0]:.1f},{e['ci'][1]:.1f}]  n={e['n']}")
            for gk in extra:
                bybk = e.get(f"by_{gk}", {})
                if bybk:
                    print(f"      by {gk}: " + "  ".join(f"{gk}{b}:{v:.1f}%" for b, v in bybk.items()))
