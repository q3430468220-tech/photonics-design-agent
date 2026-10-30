"""evaluate(): the judge. Geometry in, scored breakdown out - or a refusal.

Order of operations (each step can stop the run before a simulation is spent):
  1. build design.py                      -> DesignError  (invalid, no sim)
  2. DRC + sim sanity                     -> invalid, no sim
  3. prediction guard (unless --manual)   -> GuardError   (invalid, no sim)
  4. backend.run                          -> SimResult
  5. decompose + closure                  -> closure fail = voided run (sim spent)
  6. FOM + constraints, prediction miss
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from .backends.base import Backend
from .checks import errors_only, format_violations, run_drc, sim_sanity
from .decompose import decompose, fom_and_constraints
from .experiment import DesignError, Experiment
from .geometry import Geometry
from .guard import GuardError, PredictionRegistry


@dataclass
class EvalResult:
    iteration: int
    valid: bool  # a scored result exists (sim ran and closure passed)
    sim_spent: bool  # a simulation was consumed (valid or voided)
    stage: str  # build | checks | guard | simulate | closure | scored
    reason: str = ""
    design_hash: str = ""
    config_hash: str = ""
    fom: Optional[float] = None
    constraints_ok: Optional[bool] = None
    constraint_violations: List[str] = field(default_factory=list)
    breakdown: Optional[dict] = None
    prediction: Optional[dict] = None
    miss_sigma: Optional[float] = None
    checks: List[str] = field(default_factory=list)
    backend: str = ""
    runtime_s: float = 0.0
    run_dir: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def short(self) -> str:
        if not self.valid:
            return f"INVALID at {self.stage}: {self.reason}"
        s = self.breakdown["summary"] if self.breakdown else {}
        ok = "ok" if self.constraints_ok else "VIOLATED: " + "; ".join(self.constraint_violations)
        miss = f", miss {self.miss_sigma:+.2f} sigma" if self.miss_sigma is not None else ""
        return (
            f"FOM(IL worst)={self.fom:.4f} dB @ {s.get('IL_worst_wl_um', 0):.3f} um; xtalk {s.get('crosstalk_worst_dB', 0):.1f} dB; "
            f"refl {s.get('reflection_worst_dB', 0):.1f} dB; closure err {s.get('closure_err', 0):.4f}; constraints {ok}{miss}"
        )


def evaluate(
    exp: Experiment,
    backend: Backend,
    iteration: int,
    geom: Optional[Geometry] = None,
    manual: bool = False,
    prediction: Optional[dict] = None,
    run_dir: Optional[Path] = None,
) -> EvalResult:
    cfg = exp.config
    t0 = time.time()
    if run_dir is None:
        run_dir = exp.run_dir(iteration) if not manual else exp.runs / time.strftime("manual_%Y%m%d_%H%M%S")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    res = EvalResult(iteration=iteration, valid=False, sim_spent=False, stage="build", config_hash=cfg.hash(), run_dir=str(run_dir))

    # 1. build
    try:
        geom = geom if geom is not None else exp.build_design()
    except DesignError as e:
        res.reason = str(e)
        return _finish(res, run_dir, t0)
    res.design_hash = geom.hash()

    # 2. checks
    res.stage = "checks"
    viols = run_drc(geom, cfg) + sim_sanity(geom, cfg)
    res.checks = [str(v) for v in viols]
    if errors_only(viols):
        res.reason = "DRC/sanity errors:\n" + format_violations(errors_only(viols))
        return _finish(res, run_dir, t0)

    # 3. guard
    res.stage = "guard"
    registry = PredictionRegistry(exp.dir)
    if manual:
        res.prediction = prediction or {"fom": None, "sigma": None, "source": "manual (guard bypassed)"}
    else:
        try:
            res.prediction = registry.consume(geom.hash())
        except GuardError as e:
            res.reason = str(e)
            return _finish(res, run_dir, t0)

    # 4. simulate
    res.stage = "simulate"
    try:
        sim = backend.run(geom, cfg, run_dir)
    except Exception as e:
        res.reason = f"backend error: {type(e).__name__}: {e}"
        res.sim_spent = True
        return _finish(res, run_dir, t0)
    res.sim_spent = True
    res.backend = sim.backend
    res.extra = {k: v for k, v in sim.extra.items() if not k.startswith("_")}
    (run_dir / "sim_result.json").write_text(json.dumps(sim.to_dict(), indent=1))

    # 5. decompose + closure
    res.stage = "closure"
    bd = decompose(sim, cfg)
    res.breakdown = bd.to_dict()
    if not bd.closure_ok:
        res.reason = (
            f"closure failed: max|box_total-1| = {bd.closure_err:.4f} > {cfg.closure_tol}"
            + ("" if sim.converged else "; backend reports unconverged")
            + " -> measurement untrustworthy, run voided (not scored)"
        )
        return _finish(res, run_dir, t0)

    # 6. score
    res.stage = "scored"
    fc = fom_and_constraints(bd, cfg)
    res.fom = fc["fom"]
    res.constraints_ok = fc["constraints_ok"]
    res.constraint_violations = fc["constraint_violations"]
    res.valid = True
    p = res.prediction
    if p and p.get("fom") is not None and p.get("sigma"):
        res.miss_sigma = (res.fom - float(p["fom"])) / float(p["sigma"])
    return _finish(res, run_dir, t0)


def _finish(res: EvalResult, run_dir: Path, t0: float) -> EvalResult:
    res.runtime_s = time.time() - t0
    (run_dir / "eval.json").write_text(json.dumps(res.to_dict(), indent=1, default=str))
    return res
