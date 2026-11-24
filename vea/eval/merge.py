"""Merge each trained LoRA adapter into its base → a standalone model vLLM can load.

WHY this exists: the eval stage loads a model with `LLM(model=path)`, which cannot
consume a bare LoRA adapter — vLLM needs a full model. The RL sweep saved only small
adapters (adapter_model.safetensors) under
    <output_dir>/ckpt/<label>/<exp>_<seed>_<ts>_checkpoints/step_<N>/
so before eval we fold each adapter into the frozen base and write a full merged model
to  <output_dir>/merged/<label>  (e.g. .../merged/k1). resolve_refs() then points eval
at these merged dirs instead of the un-loadable adapter checkpoint dirs.

Run ON THE NODE (needs torch+transformers+peft and one GPU for the 32B base), offline:
    HF_HUB_OFFLINE=1 python -m vea.eval.merge configs/breadth_32b_lora.yaml
    python -m vea.eval.merge configs/breadth_32b_lora.yaml --only-k 2 --force

Idempotent: a rung whose merged dir already holds a config.json + weights is skipped
(use --force to re-merge). One rung at a time, freeing the GPU between, so a 32B base
(~64GB bf16) fits comfortably on a single 97GB GH200.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

# import-light: only the config loader (no torch at module import, so model.py can
# import merged_model_dir() during planning / dry-run without a GPU stack present).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vea import config as cfgmod  # noqa: E402


def merged_model_dir(output_dir, label: str) -> Path:
    """Canonical location of the merged full model for a rung label (e.g. 'k=1' → .../merged/k1)."""
    safe = label.replace("=", "").replace(":", "_")
    return Path(output_dir).resolve() / "merged" / safe


def _rung_label(ms, rung: dict) -> str:
    """Mirror of model._rung_label, reimplemented here to avoid importing the train stack."""
    axes = ms.dataset.sweep_axes() if ms.dataset else []
    if axes:
        key = axes[0][0]
        return f"{key}={rung.get(key)}"
    return "trained"


def _ckpt_out_dir(output_dir, label: str) -> Path:
    """Where the training stage wrote this rung's checkpoints (mirror of plan_training)."""
    safe = label.replace("=", "").replace(":", "_")
    return Path(output_dir).resolve() / "ckpt" / safe


def _latest_adapter(ckpt_out_dir: Path) -> str | None:
    """Return the dir holding the highest-step adapter_config.json under a rung's ckpt dir."""
    cands = glob.glob(str(ckpt_out_dir / "*" / "step_*" / "adapter_config.json"))
    if not cands:
        cands = glob.glob(str(ckpt_out_dir / "**" / "adapter_config.json"), recursive=True)
    if not cands:
        return None

    def stepnum(p):
        m = re.search(r"step_(\d+)", p)
        return int(m.group(1)) if m else -1

    cands.sort(key=stepnum)
    return str(Path(cands[-1]).parent)


def _is_merged(d: Path) -> bool:
    return (d / "config.json").exists() and (
        any(d.glob("*.safetensors")) or (d / "pytorch_model.bin").exists())


def merge_one(base_id: str, adapter_dir: str, out_dir: Path, device: str = "cuda:0"):
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[merge] base={base_id}\n        adapter={adapter_dir}\n        -> {out_dir}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True)
    base = base.to(device)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_id, trust_remote_code=True).save_pretrained(out_dir)
    del model, base
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"[merge] done -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(prog="vea.eval.merge")
    ap.add_argument("config")
    ap.add_argument("--only-k", type=int, default=None,
                    help="merge only this one rung (by k value)")
    ap.add_argument("--force", action="store_true", help="re-merge even if the merged dir exists")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = cfgmod.load(args.config)
    ms = cfg.model
    if ms is None or ms.source != "train":
        sys.exit("merge: config has no trained model stage")

    rc = 0
    for rung in ms.rungs():
        label = _rung_label(ms, rung)
        if args.only_k is not None and rung.get("k") != args.only_k:
            continue
        out_dir = merged_model_dir(cfg.output_dir, label)
        if _is_merged(out_dir) and not args.force:
            print(f"[merge] skip (already merged): {label} -> {out_dir}", flush=True)
            continue
        adapter = _latest_adapter(_ckpt_out_dir(cfg.output_dir, label))
        if adapter is None:
            print(f"[merge] !! no adapter found for {label} under "
                  f"{_ckpt_out_dir(cfg.output_dir, label)} — did this rung train?", flush=True)
            rc = 1
            continue
        try:
            merge_one(ms.base, adapter, out_dir, device=args.device)
        except Exception as e:  # noqa: BLE001 — report + continue so one bad rung doesn't block the rest
            print(f"[merge] !! FAILED {label}: {type(e).__name__}: {e}", flush=True)
            rc = 1

    print("\n[merge] === audit ===", flush=True)
    for rung in ms.rungs():
        label = _rung_label(ms, rung)
        if args.only_k is not None and rung.get("k") != args.only_k:
            continue
        d = merged_model_dir(cfg.output_dir, label)
        print(f"  {label}: {'OK ' if _is_merged(d) else 'MISSING'} {d}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
