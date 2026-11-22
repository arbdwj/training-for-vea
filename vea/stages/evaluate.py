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
