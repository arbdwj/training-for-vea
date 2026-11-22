"""Modal backend — x86 H100/H200 containers for the eval and train stages.

Deploy:  modal deploy vea/backends/modal_app.py
Then the pipeline (backend: modal) invokes `generate` / `train` by name, and the UI
polls `vea-progress` (a modal.Dict) for live bars. This mirrors the proven pattern in
complexity_vs_vea/modal_app.py (modal.Dict progress, volumes for weights/outputs,
gpu="H200:N" for multi-GPU).

Validate the open-instruct training path HERE (x86) before porting to the GH200 node,
since open-instruct's aarch64 support is unverified (flash-attn is skipped on aarch64).
"""
from __future__ import annotations

import os

try:
    import modal
except Exception:  # lets the module be imported on a machine without modal
    modal = None

if modal is not None:
    app = modal.App("vea-pipeline")

    # eval image: vLLM for rollout generation
    gen_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("vllm>=0.6.0", "transformers>=4.57", "huggingface-hub>=0.24")
    )
    # train image: install the vendored open-instruct deps (built from its pyproject)
    train_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .pip_install("torch>=2.10", "vllm>=0.19.1", "ray[default]>=2.49.2",
                     "deepspeed>=0.18.3", "transformers>=5.4.0", "accelerate>=1.10.1",
                     "datasets>=4.8.4,<5")
    )

    out_vol = modal.Volume.from_name("vea-outputs", create_if_missing=True)
    ckpt_vol = modal.Volume.from_name("vea-checkpoints", create_if_missing=True)
    progress = modal.Dict.from_name("vea-progress", create_if_missing=True)
    SECRETS = [modal.Secret.from_name("vea-secrets")]   # HF_TOKEN, WANDB, etc.

    # --- EVAL: generate rollouts -------------------------------------------
    @app.function(image=gen_image, gpu="H200:4", volumes={"/out": out_vol},
                  secrets=SECRETS, timeout=24 * 3600)
    def generate(model: str, cell: str, prompts: list[dict], gen_cfg: dict,
                 tensor_parallel: int = 4):
        """Generate one (model, cell) rollout file into /out; update progress Dict."""
        import json
        from vllm import LLM, SamplingParams
        key = f"gen:{model}:{cell}"
        progress[key] = {"done": 0, "total": len(prompts) * gen_cfg["rollouts"], "status": "running"}
        llm = LLM(model=model, tensor_parallel_size=tensor_parallel,
                  max_model_len=gen_cfg.get("max_len", gen_cfg["max_tokens"] + 2048),
                  trust_remote_code=True)
        tmpl = "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>"
        sp = SamplingParams(n=gen_cfg["rollouts"], temperature=gen_cfg["temperature"],
                            top_p=gen_cfg["top_p"], max_tokens=gen_cfg["max_tokens"],
                            seed=gen_cfg["seed"])
        outs = llm.generate([tmpl.format(p=p["prompt"]) for p in prompts], sp)
        path = f"/out/{cell}.jsonl"
        n = 0
        with open(path, "w") as f:
            for p, o in zip(prompts, outs):
                for comp in o.outputs:
                    f.write(json.dumps({"prompt_id": p.get("prompt_id"), "k": p.get("k"),
                                        "completion": comp.text}) + "\n")
                    n += 1
                progress[key] = {"done": n, "total": len(prompts) * gen_cfg["rollouts"],
                                 "status": "running"}
        out_vol.commit()
        progress[key] = {"done": n, "total": n, "status": "done"}
        return {"path": path, "n": n}

    # --- TRAIN: run open-instruct grpo_fast.py -----------------------------
    @app.function(image=train_image, gpu="H200:8",
                  volumes={"/ckpt": ckpt_vol, "/out": out_vol}, secrets=SECRETS,
                  timeout=48 * 3600, mounts=[modal.Mount.from_local_dir(
                      os.path.join(os.path.dirname(__file__), "..", "..", "third_party",
                                   "open-instruct"), remote_path="/open-instruct")])
    def train(argv: list[str], exp_name: str):
        """Run the rendered grpo_fast.py argv inside the vendored open-instruct."""
        import subprocess
        key = f"train:{exp_name}"
        progress[key] = {"done": 0, "total": 0, "status": "running"}
        # start a single-node ray head, then the trainer
        subprocess.run(["ray", "start", "--head", "--disable-usage-stats"], cwd="/open-instruct")
        proc = subprocess.Popen(argv, cwd="/open-instruct",
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        from ..backends.base import parse_train_log_line  # noqa (available if package mounted)
        for line in proc.stdout:        # stream + parse progress
            print(line, end="")
            try:
                step, reward = parse_train_log_line(line)
            except Exception:
                step, reward = None, None
            cur = progress.get(key, {})
            if step is not None:
                cur["done"] = step
            if reward is not None:
                cur["reward"] = reward
            cur["status"] = "running"
            progress[key] = cur
        proc.wait()
        ckpt_vol.commit()
        progress[key] = {**progress.get(key, {}), "status": "done" if proc.returncode == 0 else "error"}
        return {"returncode": proc.returncode, "exp_name": exp_name}


def poll_progress_into_store(store, keys, interval: float = 5.0, stop=None):
    """Client-side: poll the modal.Dict and mirror into a ProgressStore for the UI."""
    import time
    d = modal.Dict.from_name("vea-progress", create_if_missing=True)
    while stop is None or not stop.is_set():
        for k in keys:
            v = d.get(k)
            if v:
                store.update(k, done=v.get("done", 0), total=v.get("total", 0) or None,
                             metric=v.get("reward"),
                             status=v.get("status", "running"))
        if store.all_done():
            break
        time.sleep(interval)
