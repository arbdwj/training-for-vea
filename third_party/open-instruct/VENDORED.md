# Vendored: allenai/open-instruct

This directory is a **vendored fork** of Ai2's open-instruct, pulled into this repo so we can
**fork on top of it** (patch `grpo_fast.py`, verifiers, launch scripts) while keeping the training
engine versioned alongside the `training-for-vea` pipeline that drives it.

| | |
|---|---|
| **Upstream** | https://github.com/allenai/open-instruct |
| **Vendored commit** | `a4cd966a9e202843baba653115045e6fd6f473db` (branch `main`) |
| **Vendored on** | 2026-11-22 |
| **License** | upstream's own (`LICENSE` in this dir) — unchanged |

## Why vendored (not a submodule)
We modify open-instruct directly (recipe scripts, flags, small patches for single-node 4×GH200 /
aarch64). A submodule would point at upstream and make local edits awkward; vendoring makes the code
first-class, editable, and committed in `training-for-vea`.

## What we use
- `open_instruct/grpo_fast.py` — the OlmoRL/GRPO entrypoint our `vea/train/open_instruct.py` launches.
- `open_instruct/ground_truth_utils.py` + `IFEvalG/` — the IFEval verifier (reused in the judge stage).
- `scripts/train/olmo3/32b_think_rl.sh` — the recipe we adapt (map internal paths → public Dolci data).

## Updating from upstream (no git-subtree on this machine)
Re-vendor manually:
```bash
git clone --depth 1 https://github.com/allenai/open-instruct.git /tmp/oi && \
  (cd /tmp/oi && git rev-parse HEAD) && rm -rf /tmp/oi/.git && \
  rsync -a --delete --exclude='LOCAL_PATCHES.md' /tmp/oi/ third_party/open-instruct/
```
Record any local patches in `LOCAL_PATCHES.md` so they can be re-applied after an update.
