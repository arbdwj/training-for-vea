"""Render an open-instruct grpo_fast.py launch command from our YAML.

Maps `model.params` + `model.cluster` onto the grpo_fast.py flags confirmed from
scripts/train/olmo3/32b_think_rl.sh. We DON'T reimplement GRPO — we build the argv
and let the vendored engine run it. The renderer returns (argv, warnings); warnings
flag anything that should be checked against the vendored trainer before a real run.

Confirmed flags (from 7b_think_rl.sh / docs/algorithms/grpo.md):
  --learning_rate --lr_scheduler_type --num_samples_per_prompt_rollout
  --num_unique_prompts_rollout --response_length --max_token_length
  --max_prompt_token_length --pack_length --temperature --beta --loss_fn
  --total_episodes --num_epochs --num_mini_batches --deepspeed_stage
  --num_learners_per_node --vllm_num_engines --vllm_tensor_parallel_size
  --gradient_checkpointing --model_name_or_path --chat_template_name
  --dataset_mixer_list --dataset_mixer_list_splits --ground_truths_key
  --sft_messages_key --apply_verifiable_reward --output_dir --exp_name --seed --save_freq
"""
from __future__ import annotations

from pathlib import Path

from ..config import ModelStage

ENTRYPOINT = "open_instruct/grpo_fast.py"

# our YAML param key -> grpo_fast.py flag (value passed through)
_PARAM_FLAG = {
    "learning_rate": "--learning_rate",
    "lr_scheduler_type": "--lr_scheduler_type",
    "num_samples_per_prompt_rollout": "--num_samples_per_prompt_rollout",
    "num_unique_prompts_rollout": "--num_unique_prompts_rollout",
    "response_length": "--response_length",
    "max_token_length": "--max_token_length",
    "max_prompt_token_length": "--max_prompt_token_length",
    "pack_length": "--pack_length",
    "temperature": "--temperature",
    "beta": "--beta",
    "loss_fn": "--loss_fn",
    "total_episodes": "--total_episodes",
    "num_epochs": "--num_epochs",
    "num_mini_batches": "--num_mini_batches",
    "seed": "--seed",
    "kl_estimator": "--kl_estimator",
    "clip_lower": "--clip_lower",
    "clip_higher": "--clip_higher",
}
_CLUSTER_FLAG = {
    "deepspeed_stage": "--deepspeed_stage",
    "vllm_num_engines": "--vllm_num_engines",
    "vllm_tensor_parallel_size": "--vllm_tensor_parallel_size",
}


def derive_total_episodes(params: dict) -> int | None:
    """grpo_fast.py counts episodes, not steps. Convert max_steps if that's what's set."""
    if params.get("total_episodes"):
        return int(params["total_episodes"])
    steps = params.get("max_steps")
    if steps:
        g = int(params.get("num_samples_per_prompt_rollout", 8))
        u = int(params.get("num_unique_prompts_rollout", 32))
        return int(steps) * g * u
    return None


def render(ms: ModelStage, *, data_path: str, output_dir: str, exp_name: str,
           chat_template: str = "olmo_thinker") -> tuple[list[str], list[str]]:
    assert ms.source == "train" and ms.framework == "open_instruct"
    p = dict(ms.params)
    c = dict(ms.cluster)
    warnings: list[str] = []

    argv: list[str] = ["python", ENTRYPOINT,
                       "--exp_name", exp_name,
                       "--model_name_or_path", ms.base,
                       "--chat_template_name", chat_template,
                       "--dataset_mixer_list", data_path, "1.0",
                       "--dataset_mixer_list_splits", "train",
                       "--sft_messages_key", (ms.dataset.messages_key if ms.dataset else "messages"),
                       "--ground_truths_key", (ms.dataset.ground_truths_key if ms.dataset else "ground_truth"),
                       "--apply_verifiable_reward", "true",
                       "--output_dir", output_dir]

    # total_episodes (converted from max_steps if needed)
    ep = derive_total_episodes(p)
    if ep is not None:
        argv += ["--total_episodes", str(ep)]
        if "total_episodes" not in p and p.get("max_steps"):
            warnings.append(f"converted max_steps={p['max_steps']} -> total_episodes={ep} "
                            f"(= steps × group × unique_prompts)")
    p.pop("total_episodes", None)
    p.pop("max_steps", None)

    # save_steps -> --save_freq
    if p.get("save_steps"):
        argv += ["--save_freq", str(p.pop("save_steps"))]

    # pack_length default (response + prompt + slack) if not given
    if "pack_length" not in p:
        rl = int(p.get("response_length", 8192))
        pl = int(p.get("max_prompt_token_length", 2048))
        argv += ["--pack_length", str(rl + pl + 1024)]

    for key, flag in _PARAM_FLAG.items():
        if key in p and p[key] is not None:
            argv += [flag, str(p[key])]

    for key, flag in _CLUSTER_FLAG.items():
        if key in c and c[key] is not None:
            argv += [flag, str(c[key])]

    # num_learners_per_node takes a per-node list; single node -> one int
    if c.get("num_learners") is not None:
        argv += ["--num_learners_per_node", str(int(c["num_learners"]))]

    # boolean flags
    if c.get("gradient_checkpointing"):
        argv += ["--gradient_checkpointing"]

    # vllm_enforce_eager: skip vLLM's torch.compile + CUDA-graph capture at engine init.
    # On aarch64/GH200 the 32B compile+capture can exceed open-instruct's 1200s init timeout
    # (TimeoutError: Initializing vLLM engines). Eager trades a little rollout throughput for a
    # fast, reliable startup — the right call for short/plumbing runs.
    if c.get("vllm_enforce_eager"):
        argv += ["--vllm_enforce_eager"]

    # attn: flash-attn is unavailable on aarch64. open-instruct's choices are
    # {torch, flash_2, flash_3, flash_4, te}; "torch" IS PyTorch SDPA (the eager/non-flash
    # path we want on GH200). Translate HF-style aliases (sdpa/eager) to the accepted name.
    if c.get("attn"):
        _ATTN_ALIAS = {"sdpa": "torch", "eager": "torch"}
        attn = _ATTN_ALIAS.get(str(c["attn"]).lower(), str(c["attn"]))
        argv += ["--attn_implementation", attn]

    # arbitrary passthrough for anything else the user puts under params.extra_flags / cluster.extra_flags
    for src in (p.get("extra_flags"), c.get("extra_flags")):
        if isinstance(src, dict):
            for k, v in src.items():
                flag = k if k.startswith("--") else f"--{k}"
                if isinstance(v, bool):
                    if v:
                        argv.append(flag)
                else:
                    argv += [flag, str(v)]

    if ms.finetune == "lora":
        argv += ["--use_lora"]
        warnings.append("finetune: lora -> added --use_lora (verify grpo_fast.py LoRA support/flags)")

    return argv, warnings


def command_string(argv: list[str]) -> str:
    """Pretty, copy-pasteable one-liner (line-continued), grouping flag+value pairs."""
    s = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            # collect flag + following non-flag values
            vals = [tok]
            j = i + 1
            while j < len(argv) and not argv[j].startswith("--"):
                vals.append(argv[j]); j += 1
            s.append(" ".join(vals)); i = j
        else:
            s.append(tok); i += 1
    return " \\\n  ".join(s)
