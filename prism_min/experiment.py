"""Experiment directory: the files the human, the agent and the runner share.

    program.md     human-written goal, constraints, FOM, budget     (agent: read-only)
    sim_config.py  locked SimConfig as CONFIG                       (agent: never sees it for editing)
    design.py      `build() -> Geometry`                            (agent edits)
    model.py       `predict(geom) -> {fom, sigma, breakdown?, notes}` (agent edits in revise mode)
    laws.md        agent-maintained lessons, append-only via the loop
    journal.jsonl  one record per iteration (+ journal.md rendering)
    runs/          per-iteration snapshots, sim data, plots
    best/          design.py of the best kept design
"""
from __future__ import annotations

import importlib.util
import shutil
import types
from pathlib import Path
from typing import Optional

from .config import SimConfig
from .geometry import Geometry


class DesignError(RuntimeError):
    pass


class Experiment:
    FILES = ("program.md", "sim_config.py", "design.py", "model.py", "laws.md")

    def __init__(self, path):
        self.dir = Path(path).resolve()
        if not self.dir.exists():
            raise FileNotFoundError(self.dir)
        self.runs = self.dir / "runs"
        self.best_dir = self.dir / "best"
        self._cfg: Optional[SimConfig] = None

    # -- files ---------------------------------------------------------------
    def path(self, name: str) -> Path:
        return self.dir / name

    def read(self, name: str) -> str:
        p = self.path(name)
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def write(self, name: str, text: str) -> None:
        if name == "sim_config.py":
            raise PermissionError("sim_config.py is locked")
        self.path(name).write_text(text, encoding="utf-8")

    def run_dir(self, iteration: int) -> Path:
        d = self.runs / f"iter_{iteration:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def snapshot(self, iteration: int) -> Path:
        d = self.run_dir(iteration)
        for name in ("design.py", "model.py"):
            if self.path(name).exists():
                shutil.copy(self.path(name), d / name)
        return d

    def save_best(self) -> None:
        self.best_dir.mkdir(exist_ok=True)
        shutil.copy(self.path("design.py"), self.best_dir / "design.py")

    def restore_best(self) -> bool:
        b = self.best_dir / "design.py"
        if b.exists():
            shutil.copy(b, self.path("design.py"))
            return True
        return False

    # -- config ------------------------------------------------------------------
    @property
    def config(self) -> SimConfig:
        if self._cfg is None:
            mod = _load_module(self.path("sim_config.py"), "sim_config")
            cfg = getattr(mod, "CONFIG", None)
            if not isinstance(cfg, SimConfig):
                raise DesignError("sim_config.py must define CONFIG = SimConfig(...)")
            self._cfg = cfg
        return self._cfg

    # -- design / model -----------------------------------------------------------
    def build_design(self, source: Optional[str] = None) -> Geometry:
        """Execute design.py (or the given source) and return build()."""
        try:
            if source is None:
                mod = _load_module(self.path("design.py"), "design")
            else:
                mod = _exec_source(source, "design_candidate")
        except Exception as e:  # syntax / import / runtime errors in agent code
            raise DesignError(f"design.py failed to import: {type(e).__name__}: {e}") from e
        build = getattr(mod, "build", None)
        if build is None:
            raise DesignError("design.py must define build() -> Geometry")
        try:
            geom = build()
        except Exception as e:
            raise DesignError(f"build() raised {type(e).__name__}: {e}") from e
        if not isinstance(geom, Geometry):
            raise DesignError(f"build() must return a Geometry, got {type(geom).__name__}")
        if not geom.polygons:
            raise DesignError("build() returned a Geometry with no polygons")
        return geom

    def load_model(self, source: Optional[str] = None):
        try:
            mod = _load_module(self.path("model.py"), "model") if source is None else _exec_source(source, "model_candidate")
        except Exception as e:
            raise DesignError(f"model.py failed to import: {type(e).__name__}: {e}") from e
        if not hasattr(mod, "predict"):
            raise DesignError("model.py must define predict(geom) -> dict(fom=..., sigma=..., ...)")
        return mod

    def predict(self, geom: Geometry, model_source: Optional[str] = None) -> dict:
        mod = self.load_model(model_source)
        try:
            out = mod.predict(geom)
        except Exception as e:
            raise DesignError(f"model.predict raised {type(e).__name__}: {e}") from e
        if not isinstance(out, dict) or "fom" not in out or "sigma" not in out:
            raise DesignError("predict() must return a dict with at least 'fom' and 'sigma'")
        out["fom"] = float(out["fom"])
        out["sigma"] = float(out["sigma"])
        return out


def _load_module(path: Path, name: str) -> types.ModuleType:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"prism_exp_{name}_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _exec_source(source: str, name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    code = compile(source, f"<{name}>", "exec")
    exec(code, mod.__dict__)
    return mod
