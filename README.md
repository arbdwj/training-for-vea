# training-for-vea

A config-driven pipeline for **VEA (verbalized eval-awareness)** experiments:
**train → eval → judge**, from one YAML, on **Modal** or a **4×GH200 SLURM node**, with a
live Inspect-evals-style terminal dashboard.

The training stage wraps Ai2's **[open-instruct](https://github.com/allenai/open-instruct)**
(vendored in `third_party/open-instruct`, the *actual* Olmo-3 RLVR engine,
`open_instruct/grpo_fast.py`) — so you get real OlmoRL parity, not a reimplementation.

```
         STAGE 1 model            STAGE 2 eval              STAGE 3 judge
  train grpo_fast.py  ──►  vLLM rollouts on datasets  ──►  LLM judge (VEA) + verifiers
  or use pretrained/closed   (forced <think>)              per-rollout rate + Wilson CI
```

## Quick start

```bash
pip install -e .                      # core deps (pyyaml, rich, anthropic, openai)

# preview the whole pipeline + rough ETA (no GPU, no API):
python scripts/run.py run configs/if_complexity_32b.yaml --dry-run

# watch the live dashboard against a fake feed (UI smoke test):
python scripts/run.py run configs/if_complexity_32b.yaml --simulate

# run just the judge over existing rollouts (API; --limit for a cheap check):
python scripts/run.py run configs/judge_7bif_repro.yaml --stages judge --limit 60

# emit the open-instruct command + sbatch for the training rungs (no GPU needed):
python scripts/run.py run configs/if_complexity_32b.yaml --stages model
```

## The YAML (three stages)

One file configures all three stages; see `configs/if_complexity_32b.yaml` (the k=1..5
instruction-complexity sweep on Olmo-3-32B) and `configs/eval_only_closed.yaml`.

- **`model`** — `source: train | pretrained | closed`.
  - `train`: `framework: open_instruct` (or `trl`), a `base` model, a `dataset` with a
    `filter` (a **list value sweeps** — `{k: [1,2,3,4,5]}` trains 5 checkpoints), OlmoRL
    `params`, and a `cluster` block (GPU split, deepspeed stage, attn). `params` map to
    `grpo_fast.py` flags; `max_steps` is converted to `--total_episodes`.
  - `pretrained`/`closed`: just a `model` id; training is skipped.
- **`eval`** — `datasets` (with `bucket_by`/`n_per_bucket`), `generation` (rollouts, temp,
  top_p, max_tokens, seed, template). Writes `{model}__{domain}_k{N}.jsonl` rollouts.
- **`judge`** — a list of `judges` (`kind: llm` with a model + prompt, or `kind: verifier`),
  `group_by` (e.g. `[model, k]`). Reports the **per-rollout VEA rate** with Wilson CIs.

## Backends

Any stage picks `backend: local | modal | slurm` (default set at top level, overridable
per stage or with `--backend`).

| backend | model (train) | eval (gen) | judge |
|---|---|---|---|
| **local** | prints the command | `evaluate.generate_cell()` on a local GPU | runs (API) |
| **modal** | `modal_app.train` (x86 H100/H200) | `modal_app.generate` | runs (API) |
| **slurm** | emits an `sbatch` (Ray single-node + DeepSpeed ZeRO-3) | — | runs (API) |

- **Modal**: `modal deploy vea/backends/modal_app.py`; progress flows through the
  `vea-progress` `modal.Dict` and the UI mirrors it.
- **SLURM (4×GH200)**: `--stages model` writes `runs/<name>/slurm/*.sbatch` + a DeepSpeed
  config; submit with `sbatch …` on the node. The script starts a single-node Ray head and
  runs `grpo_fast.py` with `--attn_implementation sdpa` (no flash-attn on aarch64).

## Layout

```
third_party/open-instruct/   vendored OlmoRL engine (see VENDORED.md)
vea/
  config.py                  YAML -> typed, validated config
  plan.py                    config -> task graph + rough ETA (dry-run)
  stages/                    model.py · evaluate.py · judge.py
  train/open_instruct.py     render grpo_fast.py flags from YAML
  data/prepare.py            Dolci IF -> open-instruct format, filtered by k
  judges/clients.py          batch LLM judge + <score>/<quote> parse + Wilson CI
  backends/                  base · local · modal_app · slurm
  ui/progress.py             rich Live dashboard + ProgressStore
configs/                     worked examples
prompts/                     VEA judge prompts (scored rubric)
scripts/run.py               CLI
```

## What's validated

- **Config / UI / dry-run / sbatch rendering** — on CPU (this repo's smoke runs).
- **Judge** — reproduces the existing 7B-IF VEA numbers through the new pipeline
  (per-rollout rate, by-k, Wilson CI; logic is byte-identical to the original script).
- **k-filter** — `data/prepare.constraint_count` matches the known Dolci-IF distribution
  (k1=2862 … k5=1039; agrees with the `constraint` column on all 13,179 rows).
- **open-instruct command** — flags match `scripts/train/olmo3/32b_think_rl.sh`.

## Known risks / things to verify on the node

- **Not a bit-exact Olmo-3 reproduction.** Ai2's `olmo3/*_rl.sh` are Beaker snapshots with
  internal paths and pre-release dataset names, "not a verified reproduction recipe." We
  adapt the *algorithm + recipe args* and map to public inputs (`Olmo-3-32B-Think-DPO`,
  `Dolci-RL-Zero-IF-7B`). Pin the vendored commit (`VENDORED.md`); the `olmo_thinker`
  template name may have changed upstream (open-instruct #1899 / #1216).
- **GH200 / aarch64 is unverified for open-instruct.** flash-attn is skipped on aarch64;
  we pass `--attn_implementation sdpa` (verify `grpo_fast.py` accepts it on the pinned
  commit). **Validate the open-instruct path on Modal (x86) first**, then port to the node;
  keep `framework: trl` as a fallback if open-instruct won't build on ARM.
- **Per-row verifier routing.** `data/prepare.py` emits `messages` + `ground_truth` +
  a `dataset` field; confirm the exact routing column against the vendored
  `open_instruct/ground_truth_utils.py` before the first real run.
- **28-node → 4-GPU scale-down.** We match per-step hyperparameters (lr, group, response
  length, beta, loss_fn, temp), not the parallelism; throughput differs from the paper.
- GPU stages don't run on a laptop — only config, dry-run, the UI, and the API judge do.
