"""LLM judge clients + prompt templating + parsing + stats.

Mirrors the proven sonnet batch-judge (environment_vs_vea/code/judge_sonnet_7bif.py)
so numbers reproduce: same prompt, same <score>/<quote> parse, same per-rollout
Wilson CI. Anthropic batch API by default; an OpenAI path is provided for gpt judges.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

CAP = 50_000   # truncate very long completions before judging (chars)


# ---------------------------------------------------------------------------
# prompt loading + templating
# ---------------------------------------------------------------------------


@dataclass
class JudgePrompt:
    system: str
    user_template: str

    def render_user(self, user_prompt: str, response: str) -> str:
        return (self.user_template
                .replace("{{ user_prompt }}", user_prompt or "")
                .replace("{{ response }}", (response or "")[:CAP]))


def load_prompt(prefix: str, repo_root: Path) -> JudgePrompt:
    """Resolve a prompt by path prefix.

    `prefix` like "prompts/vea_judge_scored" loads <prefix>.system.txt + <prefix>.user.txt.
    A prefix pointing at a single existing file loads it as the user template (empty system).
    """
    base = (repo_root / prefix) if not os.path.isabs(prefix) else Path(prefix)
    sys_p, usr_p = Path(f"{base}.system.txt"), Path(f"{base}.user.txt")
    if sys_p.exists() and usr_p.exists():
        return JudgePrompt(system=sys_p.read_text(), user_template=usr_p.read_text())
    single = Path(str(base)) if Path(str(base)).exists() else usr_p
    if single.exists():
        return JudgePrompt(system="", user_template=single.read_text())
    raise FileNotFoundError(f"judge prompt not found for prefix '{prefix}' "
                            f"(looked for {sys_p}, {usr_p}, {base})")


# ---------------------------------------------------------------------------
# parsing + stats
# ---------------------------------------------------------------------------


def parse_score_quote(txt: str) -> tuple[int, str | None, list[str]]:
    txt = txt or ""
    s = re.search(r"<score>\s*([01])\s*</score>", txt)
    quotes = re.findall(r"<quote>(.*?)</quote>", txt, re.S)
    purposes = re.findall(r"<purpose>(.*?)</purpose>", txt, re.S)
    score = int(s.group(1)) if s else 0
    return score, (quotes[0].strip() if quotes else None), [p.strip() for p in purposes]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (pct, lo, hi) as a Wilson score interval for k successes of n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * p, max(0.0, (c - h) * 100), (c + h) * 100


# ---------------------------------------------------------------------------
# batch judge clients
# ---------------------------------------------------------------------------


@dataclass
class JudgeRequest:
    custom_id: str
    system: str
    user: str


class AnthropicBatchJudge:
    def __init__(self, model: str):
        import anthropic
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def submit(self, reqs: list[JudgeRequest]) -> str:
        batch = self.client.messages.batches.create(requests=[
            {"custom_id": r.custom_id,
             "params": {"model": self.model, "max_tokens": 1024,
                        "system": r.system,
                        "messages": [{"role": "user", "content": r.user}]}}
            for r in reqs])
        return batch.id

    def poll(self, batch_id: str, total: int, progress_cb=None, interval: float = 20.0):
        t0 = time.time()
        while True:
            info = self.client.messages.batches.retrieve(batch_id)
            rc = info.request_counts
            done = rc.succeeded + rc.errored + rc.canceled + rc.expired
            if progress_cb:
                progress_cb(done, total)
            if info.processing_status == "ended":
                return
            time.sleep(interval)

    def results(self, batch_id: str):
        """Yield (custom_id, text, ok)."""
        for res in self.client.messages.batches.results(batch_id):
            if res.result.type == "succeeded":
                msg = res.result.message
                text = "".join(b.text for b in (msg.content or [])
                               if getattr(b, "type", None) == "text")
                yield res.custom_id, text, True
            else:
                yield res.custom_id, "", False


def make_judge(model: str):
    """Pick a client by model id."""
    if model.startswith(("gpt", "o1", "o3", "o4")):
        from .openai_client import OpenAIBatchJudge  # lazy; optional
        return OpenAIBatchJudge(model)
    return AnthropicBatchJudge(model)
