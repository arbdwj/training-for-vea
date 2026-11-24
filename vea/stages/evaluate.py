"""STAGE 2 — generate rollouts on eval datasets with vLLM (or an API for closed models).

Ported from environment_vs_vea/scale_stage_gen/generate.py: forced-`<think>` Olmo
template, per-bucket prompt selection, resumable JSONL. Writes one file per
(model, dataset, bucket) named `{model}__{domain}_k{N}.jsonl` with records
{prompt_id, completion, k} — exactly the shape the judge stage consumes.

vLLM is imported lazily so this module loads on a machine without a GPU (dry-run,
planning, tests). Actual generation runs on Modal or the GH200 node.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import EvalStage, GenCfg

# Olmo "think" chat template — forces the assistant turn to open with <think>,
# matching our fig2a protocol so VEA numbers are comparable.
OLMO_THINK_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>"


def render_prompt(template: str, prompt: str) -> str:
    if template in ("olmo_think", "olmo_thinker"):
        return OLMO_THINK_TEMPLATE.format(prompt=prompt)
    return prompt


def load_prompts(path: str) -> list[dict]:
    """Load a prompt jsonl (records with prompt_id/prompt and optional k/domain)."""
    if path.startswith("hf://"):
        raise NotImplementedError(
            "hf:// eval prompt loading runs on the GPU node (needs `datasets`); "
            "for local/dry use a jsonl path. See data/prepare.sample_eval_prompts().")
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def select_bucket(prompts: list[dict], bucket_by: str | None, bucket_val, n: int) -> list[dict]:
    if bucket_by is not None and bucket_val is not None:
        prompts = [p for p in prompts if p.get(bucket_by) == bucket_val]
    return prompts[:n] if n else prompts


def generate_cell(llm, prompts: list[dict], gen: GenCfg, out_path: Path,
                  model_label: str, domain: str, k, progress_cb=None):
    """Generate `gen.rollouts` completions per prompt; resumable; write JSONL."""
    from vllm import SamplingParams
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    if out_path.exists():
        done = sum(1 for _ in open(out_path))      # resume
    sp = SamplingParams(n=gen.rollouts, temperature=gen.temperature, top_p=gen.top_p,
                        max_tokens=gen.max_tokens, seed=gen.seed)
    texts = [render_prompt(gen.template, p["prompt"]) for p in prompts]
    with open(out_path, "a") as f:
        written = 0
        outs = llm.generate(texts, sp)
        for p, o in zip(prompts, outs):
            for comp in o.outputs:
                rec = {"prompt_id": p.get("prompt_id") or p.get("id"),
                       "k": p.get(k) if isinstance(k, str) else k,
                       "completion": comp.text, "model": model_label, "domain": domain}
                if written >= done:
                    f.write(json.dumps(rec) + "\n")
                written += 1
                if progress_cb:
                    progress_cb(written)
    return written


def plan_cells(stage: EvalStage, model_labels: list[str]):
    """[(model_label, dataset, bucket_val, n)] the eval stage will generate."""
    cells = []
    for lbl in model_labels:
        for ds in stage.datasets:
            buckets = (ds.filter.get(ds.bucket_by) if ds.bucket_by and
                       isinstance(ds.filter.get(ds.bucket_by), list) else [None])
            if buckets == [None] and ds.bucket_by == "k":
                buckets = [1, 2, 3, 4, 5]
            for b in buckets:
                n = ds.n_per_bucket if b is not None else (ds.n or 100)
                cells.append((lbl, ds, b, n))
    return cells


def cell_filename(model_label: str, domain: str, bucket_val) -> str:
    safe = model_label.replace("=", "").replace(":", "_")
    kv = bucket_val if bucket_val is not None else 0
    return f"{safe}__{domain}_k{kv}.jsonl"


def run_local(stage: EvalStage, model_path: str, model_label: str, prompts_path: str,
              out_dir, store=None, tensor_parallel: int = 1):
    """Generate every cell for one model on a local GPU (the node). Writes rollouts/*.jsonl."""
    from pathlib import Path as _P
    from vllm import LLM
    out_dir = _P(out_dir)
    (out_dir).mkdir(parents=True, exist_ok=True)
    gen = stage.generation
    # GH200/aarch64: vLLM's CUDA-graph capture / torch.compile path is fragile and crashes
    # a worker during engine init (this is why the training config sets vllm_enforce_eager).
    # Default eager here too; let the eval backend override via extra.enforce_eager.
    extra = getattr(stage.backend, "extra", {}) or {}
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel,
              max_model_len=gen.max_tokens + 2048, trust_remote_code=True,
              enforce_eager=bool(extra.get("enforce_eager", True)),
              gpu_memory_utilization=float(extra.get("gpu_memory_utilization", 0.90)))
    all_prompts = load_prompts(prompts_path)
    for ds in stage.datasets:
        buckets = (ds.filter.get(ds.bucket_by) if ds.bucket_by and
                   isinstance(ds.filter.get(ds.bucket_by), list) else [None])
        if buckets == [None] and ds.bucket_by == "k":
            buckets = [1, 2, 3, 4, 5]
        for b in buckets:
            n = ds.n_per_bucket if b is not None else (ds.n or 100)
            sel = select_bucket(all_prompts, ds.bucket_by, b, n)
            fname = cell_filename(model_label, ds.name if ds.bucket_by != "k" else "if", b)
            tid = f"gen:{model_label}:{ds.name}/{ds.bucket_by}{b}" if b is not None else f"gen:{model_label}:{ds.name}"
            if store is not None:
                store.update(tid, status="running", total=len(sel) * gen.rollouts)
            cb = (lambda done, _tid=tid: store.update(_tid, done=done)) if store is not None else None
            generate_cell(llm, sel, gen, out_dir / fname, model_label,
                          ds.name if ds.bucket_by != "k" else "if", b, progress_cb=cb)
            if store is not None:
                store.update(tid, status="done")
    _free_vllm(llm)
    return str(out_dir)


def _free_vllm(llm):
    """Release the vLLM engine + GPU memory so a later in-process model can load."""
    try:
        import contextlib
        import gc

        import torch
        del llm
        with contextlib.suppress(Exception):
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
