"""Inspect-evals-style terminal progress.

A `ProgressStore` is a thread-safe bag of `Task` rows that any backend writes to
(a local progress_cb, a polled Modal Dict, or a tailed SLURM/open-instruct log).
The `Dashboard` renders it live with `rich`; if `rich` is missing it degrades to a
periodic plaintext dump, so dry-runs and headless runs still work.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

STATUS_ORDER = {"running": 0, "pending": 1, "done": 2, "error": 3}
STAGE_ORDER = {"model": 0, "eval": 1, "judge": 2}
_STATUS_MARK = {"pending": "·", "running": "▸", "done": "✓", "error": "✗"}


@dataclass
class Task:
    id: str
    stage: str
    backend: str = "local"
    status: str = "pending"            # pending | running | done | error
    done: int = 0
    total: int = 0
    metric_name: str | None = None     # "reward" | "VEA%" | ...
    metric_value: float | None = None
    note: str = ""
    t_start: float | None = None
    t_end: float | None = None

    @property
    def elapsed(self) -> float:
        if self.t_start is None:
            return 0.0
        return (self.t_end or time.time()) - self.t_start

    @property
    def frac(self) -> float:
        return (self.done / self.total) if self.total else 0.0

    @property
    def eta(self) -> float | None:
        if self.status != "running" or not self.done or not self.total:
            return None
        rate = self.done / max(self.elapsed, 1e-6)
        return (self.total - self.done) / rate if rate else None


class ProgressStore:
    def __init__(self, title: str = "", subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self._tasks: dict[str, Task] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    def add(self, task_id: str, stage: str, total: int = 0, backend: str = "local",
            metric_name: str | None = None, note: str = "") -> Task:
        with self._lock:
            if task_id not in self._tasks:
                self._order.append(task_id)
            t = Task(id=task_id, stage=stage, backend=backend, total=total,
                     metric_name=metric_name, note=note)
            self._tasks[task_id] = t
            return t

    def update(self, task_id: str, *, status: str | None = None, done: int | None = None,
               total: int | None = None, metric: float | None = None, note: str | None = None):
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None:
                t = Task(id=task_id, stage="?")
                self._tasks[task_id] = t
                self._order.append(task_id)
            if status is not None:
                if status == "running" and t.t_start is None:
                    t.t_start = time.time()
                if status in ("done", "error") and t.t_end is None:
                    t.t_end = time.time()
                t.status = status
            if done is not None:
                t.done = done
                if t.t_start is None:
                    t.t_start = time.time()
            if total is not None:
                t.total = total
            if metric is not None:
                t.metric_value = metric
            if note is not None:
                t.note = note

    def snapshot(self) -> list[Task]:
        with self._lock:
            return [self._tasks[i] for i in self._order]

    def all_done(self) -> bool:
        with self._lock:
            ts = list(self._tasks.values())
            return bool(ts) and all(t.status in ("done", "error") for t in ts)

    # -- rendering ---------------------------------------------------------

    def rows(self):
        """(stage, backend, status_mark, bar, done/total, metric, time) tuples."""
        ts = sorted(self.snapshot(),
                    key=lambda t: (STAGE_ORDER.get(t.stage, 9), STATUS_ORDER.get(t.status, 9), t.id))
        for t in ts:
            dt = f"{t.done}/{t.total}" if t.total else (str(t.done) if t.done else "")
            metric = ""
            if t.metric_value is not None:
                metric = f"{t.metric_name or ''} {t.metric_value:.3g}".strip()
            tm = _fmt_time(t.elapsed)
            if t.eta is not None:
                tm += f" ~{_fmt_time(t.eta)}"
            yield t, dt, _bar(t.frac, t.status), metric, tm


def _bar(frac: float, status: str, width: int = 22) -> str:
    frac = max(0.0, min(1.0, frac))
    fill = int(round(frac * width))
    ch = "━" if status != "error" else "╳"
    return ch * fill + "·" * (width - fill)


def _fmt_time(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich import box
    _HAS_RICH = True
except Exception:  # pragma: no cover
    _HAS_RICH = False


def _rich_table(store: ProgressStore) -> "Table":
    done = sum(1 for t in store.snapshot() if t.status == "done")
    err = sum(1 for t in store.snapshot() if t.status == "error")
    n = len(store.snapshot())
    cap = store.title or "pipeline"
    sub = store.subtitle
    title = f"[bold]{cap}[/]   {done}/{n} done" + (f"  [red]{err} failed[/]" if err else "")
    if sub:
        title += f"\n[dim]{sub}[/]"
    tbl = Table(title=title, box=box.SIMPLE_HEAVY, expand=True, title_justify="left", pad_edge=False)
    tbl.add_column("", width=1)
    tbl.add_column("task", style="bold", no_wrap=True)
    tbl.add_column("stage", style="cyan", no_wrap=True)
    tbl.add_column("backend", style="dim", no_wrap=True)
    tbl.add_column("progress", no_wrap=True)
    tbl.add_column("n", justify="right", no_wrap=True)
    tbl.add_column("metric", justify="right", style="yellow", no_wrap=True)
    tbl.add_column("time", justify="right", style="dim", no_wrap=True)
    color = {"pending": "dim", "running": "white", "done": "green", "error": "red"}
    for t, dt, bar, metric, tm in store.rows():
        c = color.get(t.status, "white")
        barcol = {"done": "green", "error": "red", "running": "blue"}.get(t.status, "dim")
        tbl.add_row(f"[{c}]{_STATUS_MARK.get(t.status,'?')}[/]", t.id, t.stage, t.backend,
                    f"[{barcol}]{bar}[/]", dt, metric, tm)
    return tbl


def render_static(store: ProgressStore) -> str:
    """Plaintext one-shot render (used for --dry-run and the no-rich fallback)."""
    lines = []
    done = sum(1 for t in store.snapshot() if t.status == "done")
    lines.append(f"{store.title}   ({done}/{len(store.snapshot())} done)")
    if store.subtitle:
        lines.append(f"  {store.subtitle}")
    lines.append("")
    hdr = f"  {'':1} {'task':<26} {'stage':<8} {'backend':<8} {'progress':<22} {'n':>9} {'metric':>14} {'time':>10}"
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for t, dt, bar, metric, tm in store.rows():
        lines.append(f"  {_STATUS_MARK.get(t.status,'?'):1} {t.id:<26.26} {t.stage:<8} "
                     f"{t.backend:<8} {bar:<22} {dt:>9} {metric:>14} {tm:>10}")
    return "\n".join(lines)


class Dashboard:
    """Context manager that live-renders a store until `.stop()` or all tasks finish.

    with Dashboard(store) as dash:
        ... backends write to store from other threads ...
        dash.wait()      # blocks until all_done()
    """

    def __init__(self, store: ProgressStore, refresh_hz: float = 4.0, enable: bool = True):
        self.store = store
        self.refresh_hz = refresh_hz
        self.enable = enable
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if self.enable and _HAS_RICH:
            self._thread = threading.Thread(target=self._run_rich, daemon=True)
            self._thread.start()
        elif self.enable:
            self._thread = threading.Thread(target=self._run_plain, daemon=True)
            self._thread.start()
        return self

    def _run_rich(self):
        console = Console()
        with Live(_rich_table(self.store), console=console, refresh_per_second=self.refresh_hz,
                  screen=False) as live:
            while not self._stop.is_set():
                live.update(_rich_table(self.store))
                time.sleep(1.0 / self.refresh_hz)
            live.update(_rich_table(self.store))

    def _run_plain(self):
        last = ""
        while not self._stop.is_set():
            cur = render_static(self.store)
            if cur != last:
                print(cur, flush=True)
                last = cur
            time.sleep(max(1.0, 1.0 / self.refresh_hz))

    def wait(self, timeout: float | None = None):
        start = time.time()
        while not self.store.all_done():
            if timeout and time.time() - start > timeout:
                break
            time.sleep(0.2)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def __exit__(self, *exc):
        self.stop()
        # final frame for the non-rich path
        if not (_HAS_RICH and self.enable):
            print(render_static(self.store), flush=True)
        return False
