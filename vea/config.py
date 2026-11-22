"""YAML -> typed config for the three-stage VEA pipeline.

A pipeline config has up to three stages:
  model   (STAGE 1): train a model, or point at a pretrained / closed one  -> ModelRef
  eval    (STAGE 2): generate rollouts on datasets                          -> rollouts
  judge   (STAGE 3): score rollouts with LLM judges + verifiers             -> summary

Validation is pragmatic: we fail fast with a readable message naming the offending
key, rather than silently accepting a typo'd field.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Raised with a human-readable, key-qualified message."""


def _req(d: dict, key: str, where: str):
    if key not in d:
        raise ConfigError(f"{where}: missing required key '{key}'")
    return d[key]


def _one_of(val, allowed, where: str):
    if val not in allowed:
        raise ConfigError(f"{where}: '{val}' is not one of {sorted(allowed)}")
    return val


def _unknown(d: dict, allowed: set[str], where: str):
    extra = set(d) - allowed
    if extra:
        raise ConfigError(f"{where}: unknown key(s) {sorted(extra)}; allowed {sorted(allowed)}")


# ---------------------------------------------------------------------------
# stage dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BackendCfg:
    type: str = "local"           # local | modal | slurm
    gpus: int = 1
    nodes: int = 1
    gpu: str | None = None        # modal gpu string, e.g. "H200" / "H200:4"
    extra: dict = field(default_factory=dict)

    @staticmethod
    def parse(v, default_type: str, where: str) -> "BackendCfg":
        if v is None:
            return BackendCfg(type=default_type)
        if isinstance(v, str):
            return BackendCfg(type=_one_of(v, {"local", "modal", "slurm"}, where))
        if not isinstance(v, dict):
            raise ConfigError(f"{where}: backend must be a string or mapping")
        t = _one_of(v.get("type", default_type), {"local", "modal", "slurm"}, where)
        known = {"type", "gpus", "nodes", "gpu"}
        return BackendCfg(
            type=t,
            gpus=int(v.get("gpus", 1)),
            nodes=int(v.get("nodes", 1)),
            gpu=v.get("gpu"),
            extra={k: val for k, val in v.items() if k not in known},
        )


@dataclass
class DatasetCfg:
    name: str                              # HF id or local path
    filter: dict = field(default_factory=dict)   # {k: 5} or {k: [1,2,3,4,5]} (sweep)
    messages_key: str = "messages"
    ground_truths_key: str = "ground_truth"
    verifier: str = "ifeval"
    split: str = "train"

    def sweep_axes(self) -> list[tuple[str, list]]:
        """Filter keys whose value is a list define a sweep; return (key, values)."""
        return [(k, v) for k, v in self.filter.items() if isinstance(v, list)]

    def expand(self) -> list[dict]:
        """Expand list-valued filters into the cartesian product of concrete filters.

        {k: [1,2,3]}            -> [{k:1},{k:2},{k:3}]
        {k: [1,2], hard: [T,F]} -> 4 combos.  No list-valued filter -> [self.filter].
        """
        axes = self.sweep_axes()
        if not axes:
            return [dict(self.filter)]
        base = {k: v for k, v in self.filter.items() if not isinstance(v, list)}
        keys = [k for k, _ in axes]
        out = []
        for combo in itertools.product(*[vs for _, vs in axes]):
            out.append({**base, **dict(zip(keys, combo))})
        return out


@dataclass
class ModelStage:
    source: str                            # train | pretrained | closed
    # train:
    framework: str = "open_instruct"       # open_instruct | trl
    recipe_ref: str | None = None
    base: str | None = None
    dataset: DatasetCfg | None = None
    params: dict = field(default_factory=dict)
    cluster: dict = field(default_factory=dict)
    finetune: str = "full_ft"              # full_ft | lora
    # pretrained / closed:
    model: str | None = None
    backend: BackendCfg = field(default_factory=BackendCfg)

    @staticmethod
    def parse(d: dict, default_backend: str) -> "ModelStage":
        where = "model"
        _unknown(d, {"source", "framework", "recipe_ref", "base", "dataset", "params",
                     "cluster", "finetune", "model", "backend"}, where)
        source = _one_of(_req(d, "source", where), {"train", "pretrained", "closed"}, f"{where}.source")
        ms = ModelStage(source=source)
        ms.backend = BackendCfg.parse(d.get("backend"), default_backend, f"{where}.backend")
        if source == "train":
            ms.framework = _one_of(d.get("framework", "open_instruct"),
                                   {"open_instruct", "trl"}, f"{where}.framework")
            ms.recipe_ref = d.get("recipe_ref")
            ms.base = _req(d, "base", where)
            ms.finetune = _one_of(d.get("finetune", "full_ft"), {"full_ft", "lora"}, f"{where}.finetune")
            ms.params = dict(d.get("params", {}))
            ms.cluster = dict(d.get("cluster", {}))
            dd = _req(d, "dataset", where)
            if not isinstance(dd, dict):
                raise ConfigError(f"{where}.dataset: must be a mapping")
            ms.dataset = DatasetCfg(
                name=_req(dd, "name", f"{where}.dataset"),
                filter=dict(dd.get("filter", {})),
                messages_key=dd.get("messages_key", "messages"),
                ground_truths_key=dd.get("ground_truths_key", "ground_truth"),
                verifier=dd.get("verifier", "ifeval"),
                split=dd.get("split", "train"),
            )
        else:  # pretrained | closed
            ms.model = _req(d, "model", where)
        return ms

    def rungs(self) -> list[dict]:
        """One training rung per expanded filter combo (sweep); [] if not training."""
        if self.source != "train" or self.dataset is None:
            return []
        return self.dataset.expand()


@dataclass
class GenCfg:
    rollouts: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 8192
    seed: int = 42
    template: str = "olmo_think"


@dataclass
class EvalDatasetCfg:
    name: str
    path: str
    bucket_by: str | None = None
    n_per_bucket: int | None = None
    n: int | None = None
    filter: dict = field(default_factory=dict)


@dataclass
class EvalStage:
    datasets: list[EvalDatasetCfg]
    generation: GenCfg = field(default_factory=GenCfg)
    backend: BackendCfg = field(default_factory=BackendCfg)

    @staticmethod
    def parse(d: dict, default_backend: str) -> "EvalStage":
        where = "eval"
        _unknown(d, {"datasets", "generation", "backend"}, where)
        raw = _req(d, "datasets", where)
        if not isinstance(raw, list) or not raw:
            raise ConfigError(f"{where}.datasets: must be a non-empty list")
        dsets = []
        for i, x in enumerate(raw):
            w = f"{where}.datasets[{i}]"
            dsets.append(EvalDatasetCfg(
                name=_req(x, "name", w),
                path=_req(x, "path", w),
                bucket_by=x.get("bucket_by"),
                n_per_bucket=x.get("n_per_bucket"),
                n=x.get("n"),
                filter=dict(x.get("filter", {})),
            ))
        g = dict(d.get("generation", {}))
        _unknown(g, {"rollouts", "temperature", "top_p", "max_tokens", "seed", "template"},
                 f"{where}.generation")
        gen = GenCfg(
            rollouts=int(g.get("rollouts", 8)),
            temperature=float(g.get("temperature", 1.0)),
            top_p=float(g.get("top_p", 1.0)),
            max_tokens=int(g.get("max_tokens", 8192)),
            seed=int(g.get("seed", 42)),
            template=g.get("template", "olmo_think"),
        )
        return EvalStage(datasets=dsets, generation=gen,
                         backend=BackendCfg.parse(d.get("backend"), default_backend, f"{where}.backend"))


@dataclass
class JudgeCfg:
    name: str
    kind: str                              # llm | verifier
    model: str | None = None
    prompt: str | None = None              # path prefix (loads .system.txt/.user.txt) or single file
    parse: str = "score_quote"


@dataclass
class JudgeStage:
    judges: list[JudgeCfg]
    aggregate: str = "per_rollout"
    rollouts_dir: str | None = None        # dir of {model}__{dataset}_{bucket}.jsonl rollouts
    prompts: str | None = None             # path(s) to prompt jsonl for prompt_id -> text lookup
    group_by: list[str] = field(default_factory=lambda: ["model"])  # report breakdown keys

    @staticmethod
    def parse(d: dict) -> "JudgeStage":
        where = "judge"
        _unknown(d, {"judges", "aggregate", "rollouts_dir", "prompts", "group_by"}, where)
        raw = _req(d, "judges", where)
        if not isinstance(raw, list) or not raw:
            raise ConfigError(f"{where}.judges: must be a non-empty list")
        js = []
        for i, x in enumerate(raw):
            w = f"{where}.judges[{i}]"
            kind = _one_of(_req(x, "kind", w), {"llm", "verifier"}, f"{w}.kind")
            if kind == "llm":
                _req(x, "model", w)
                _req(x, "prompt", w)
            js.append(JudgeCfg(
                name=_req(x, "name", w),
                kind=kind,
                model=x.get("model"),
                prompt=x.get("prompt"),
                parse=x.get("parse", "score_quote"),
            ))
        gb = d.get("group_by", ["model"])
        if isinstance(gb, str):
            gb = [gb]
        return JudgeStage(judges=js, aggregate=d.get("aggregate", "per_rollout"),
                          rollouts_dir=d.get("rollouts_dir"), prompts=d.get("prompts"),
                          group_by=list(gb))


@dataclass
class PipelineCfg:
    name: str
    output_dir: str = "runs"
    backend: str = "local"
    model: ModelStage | None = None
    eval: EvalStage | None = None
    judge: JudgeStage | None = None
    config_path: Path | None = None

    @property
    def stages(self) -> list[str]:
        out = []
        if self.model:
            out.append("model")
        if self.eval:
            out.append("eval")
        if self.judge:
            out.append("judge")
        return out


def load(path: str | Path) -> PipelineCfg:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping")
    _unknown(d, {"name", "output_dir", "backend", "model", "eval", "judge"}, "<top-level>")
    default_backend = _one_of(d.get("backend", "local"), {"local", "modal", "slurm"}, "backend")
    cfg = PipelineCfg(
        name=_req(d, "name", "<top-level>"),
        output_dir=d.get("output_dir", "runs"),
        backend=default_backend,
        config_path=path.resolve(),
    )
    if "model" in d and d["model"]:
        cfg.model = ModelStage.parse(d["model"], default_backend)
    if "eval" in d and d["eval"]:
        cfg.eval = EvalStage.parse(d["eval"], default_backend)
    if "judge" in d and d["judge"]:
        cfg.judge = JudgeStage.parse(d["judge"])
    if not cfg.stages:
        raise ConfigError(f"{path}: config has no stages (need at least one of model/eval/judge)")
    return cfg
