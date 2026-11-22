#!/usr/bin/env python
"""Preflight for the GH200 node: check the stack before launching a 32B e2e run.

Checks (non-fatal, reported): torch+CUDA, vLLM, Ray, DeepSpeed, transformers,
the vendored grpo_fast import, and that vLLM is new enough for the Nemotron-3
hybrid-MoE judge. Exits 0 if the essentials import, 1 otherwise.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

OI = Path(__file__).resolve().parents[1] / "third_party" / "open-instruct"


def check(label, fn):
    try:
        info = fn()
        print(f"  [ok]  {label}: {info}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {type(e).__name__}: {e}")
        return False


def _torch():
    import torch
    cuda = torch.cuda.is_available()
    n = torch.cuda.device_count() if cuda else 0
    name = torch.cuda.get_device_name(0) if cuda else "no-cuda"
    return f"torch {torch.__version__}, cuda={cuda}, gpus={n} ({name})"


def _vllm():
    import vllm
    return f"vllm {vllm.__version__}"


def _vllm_nemotron():
    # Nemotron-3 Super is a hybrid Mamba-Transformer MoE; needs a recent vLLM.
    import vllm
    v = tuple(int(x) for x in vllm.__version__.split(".")[:3] if x.isdigit())
    ok = v >= (0, 11, 0)
    return f"vllm {vllm.__version__} {'>=0.11 (likely has Nemotron-3/Mamba2)' if ok else '<0.11 — may lack Nemotron-3 hybrid-MoE support; upgrade'}"


def main():
    print("preflight: training-for-vea @ node")
    essentials = []
    essentials.append(check("torch + CUDA", _torch))
    essentials.append(check("vLLM", _vllm))
    check("vLLM Nemotron-3 readiness", _vllm_nemotron)
    for mod in ("ray", "deepspeed", "transformers", "datasets"):
        check(mod, lambda m=mod: importlib.import_module(m).__version__)
    # vendored grpo_fast import (the actual training entrypoint)
    sys.path.insert(0, str(OI))
    essentials.append(check("open_instruct.grpo_fast import",
                            lambda: __import__("open_instruct.grpo_fast",
                                               fromlist=["x"]).__name__))
    # our pipeline
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    check("vea pipeline import", lambda: __import__("vea").__version__)

    ok = all(essentials)
    print("\nPREFLIGHT:", "PASS (essentials import)" if ok else "PROBLEMS — fix the [FAIL] lines above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
