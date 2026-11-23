"""Prepare Dolci IF data for open-instruct GRPO training, filtered by constraint count k.

The Dolci-RL-Zero-IF dataset stores, per row: a `prompt`, a `ground_truth` whose
first element is a Python-repr string holding the IFEval `instruction_id` list (so
k = number of constraints), and a `constraint`/`key` field.

open-instruct's grpo_fast.py reads prompts from a `messages` column and the verifier
label from a `ground_truth` column, routing to a verifier by a per-row dataset name.
We emit exactly that shape, one local JSONL per k-rung.

NOTE: the exact per-row verifier-routing column for the vendored open-instruct commit
must be confirmed against third_party/open-instruct/open_instruct/ground_truth_utils.py
(`build_all_verifiers` / `apply_verifiable_reward`) before the first real train run;
we default to a `dataset` field = the configured verifier name and flag it in the README.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

# instruction ids look like 'keywords:palindrome', 'length_constraints:number_paragraphs'
_IID = re.compile(r"'[\w.]+:[\w.]+'")


def constraint_count(ground_truth) -> int:
    """Count IFEval constraints in a row's ground_truth.

    ground_truth is a list of one repr-string: "[{'instruction_id': [id1, id2, ...],
    'kwargs': [...]}]".  k = total number of ids across the instruction_id list(s).
    """
    gt = ground_truth
    if isinstance(gt, (list, tuple)) and gt:
        gt = gt[0]
    if isinstance(gt, str):
        try:
            return len(_collect_ids(ast.literal_eval(gt)))
        except Exception:
            return len(_IID.findall(gt))
    return len(_collect_ids(gt))


def _collect_ids(obj) -> list[str]:
    """Flatten every instruction_id (which is itself a list) into one list of ids."""
    out: list[str] = []
    if isinstance(obj, dict):
        iid = obj.get("instruction_id")
        if isinstance(iid, (list, tuple)):
            out += [str(x) for x in iid]
        elif iid is not None:
            out.append(str(iid))
        for key, v in obj.items():
            if key == "instruction_id":
                continue
            out += _collect_ids(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _collect_ids(v)
    return out


def constraint_count_from_column(constraint: str) -> int:
    """Direct k from the tab-separated `constraint` column (authoritative cross-check)."""
    return len([c for c in str(constraint).split("\t") if c.strip()])


def load_dolci_if(name: str, split: str = "train"):
    """Load the HF dataset (lazy import so this module works without `datasets`)."""
    from datasets import load_dataset
    if name.startswith("hf://"):
        name = name[len("hf://"):]
    return load_dataset(name, split=split)


def to_open_instruct_row(row: dict, verifier: str = "ifeval",
                         messages_key: str = "messages",
                         ground_truths_key: str = "ground_truth") -> dict:
    prompt = row.get("prompt") or row.get("question") or ""
    gt = row.get("ground_truth")
    # Dolci stores ground_truth as a list-of-one repr-string: ["[{'instruction_id': ...}]"].
    # The open-instruct ifeval verifier (ground_truth_utils.IFEvalVerifier) does
    # `ast.literal_eval(label)[0]`, so `label` must be the STRING "[{...}]" itself, not the
    # ["[{...}]"] list wrapper. The grpo_fast async reward path (vllm_utils.compute_rewards)
    # passes ground_truth straight through as `label` (unlike apply_verifiable_reward, which
    # unwraps a non-str), so a list wrapper here makes literal_eval raise "malformed node or
    # string" and every rollout's reward silently fails -> DataPreparationActor hangs forever.
    if isinstance(gt, (list, tuple)) and len(gt) == 1 and isinstance(gt[0], str):
        gt = gt[0]
    return {
        messages_key: [{"role": "user", "content": prompt}],
        ground_truths_key: gt,
        "dataset": verifier,          # per-row verifier routing (confirm column vs vendored loader)
        "constraint_count": constraint_count(gt),
    }


def prepare_rung(name: str, k_filter: dict, out_path: str | Path, *, verifier: str = "ifeval",
                 split: str = "train", messages_key: str = "messages",
                 ground_truths_key: str = "ground_truth", limit: int | None = None) -> dict:
    """Write one local JSONL for a single k-rung. Returns {path, n, k}."""
    ds = load_dolci_if(name, split)
    k = k_filter.get("k")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as f:
        for row in ds:
            if k is not None and constraint_count(row.get("ground_truth")) != k:
                continue
            f.write(json.dumps(to_open_instruct_row(
                row, verifier, messages_key, ground_truths_key)) + "\n")
            n += 1
            if limit and n >= limit:
                break
    return {"path": str(out_path), "n": n, "k": k}


def sample_eval_prompts(name: str, out_path: str | Path, *, buckets=(1, 2, 3, 4, 5),
                        n_per_bucket: int = 100, split: str = "train", seed: int = 0) -> dict:
    """Build a local eval-prompt JSONL from Dolci-IF: {prompt_id, prompt, k, domain}.

    Deterministic per-k sampling so eval/judge are reproducible. prompt_id is stable
    (f"if_k{k}_{i}") so a judge can look prompts back up by id.
    """
    import random
    ds = load_dolci_if(name, split)
    by_k: dict[int, list[dict]] = {b: [] for b in buckets}
    for row in ds:
        k = constraint_count(row.get("ground_truth"))
        if k in by_k:
            by_k[k].append(row)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written = 0
    with open(out_path, "w") as f:
        for k in buckets:
            rows = by_k[k]
            rng.shuffle(rows)
            for i, row in enumerate(rows[:n_per_bucket]):
                prompt = row.get("prompt") or ""
                # strip a leading "user:" the Dolci prompts carry
                prompt = prompt[5:].lstrip() if prompt.lower().startswith("user:") else prompt
                f.write(json.dumps({"prompt_id": f"if_k{k}_{i}", "prompt": prompt,
                                    "k": k, "domain": "if"}) + "\n")
                written += 1
    return {"path": str(out_path), "n": written, "buckets": list(buckets)}


def k_histogram(name: str, split: str = "train") -> dict[int, int]:
    """Count rows per k (for dry-run reporting / sanity)."""
    ds = load_dolci_if(name, split)
    hist: dict[int, int] = {}
    for row in ds:
        k = constraint_count(row.get("ground_truth"))
        hist[k] = hist.get(k, 0) + 1
    return dict(sorted(hist.items()))
