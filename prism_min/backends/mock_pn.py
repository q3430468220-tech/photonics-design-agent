"""Analytic mock backend for the pn_phase_shifter family (toy physics, see families/pn_phase_shifter.py).

Deterministic per geometry (seeded by its hash) with 0.8 % multiplicative noise on the
full-solve quantities, so the closure test against the perturbation terms is exercised.
Reports an unconverged run when the depletion width is below two CHARGE mesh cells.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..families.pn_phase_shifter import PNResult, toy_physics
from ..geometry import Geometry
from .base import Backend, register


@register("mock_pn")
class MockPN(Backend):
    def __init__(self, noise: float = 0.008, **_):
        self.noise = noise

    def run(self, geom: Geometry, cfg, run_dir: Optional[Path] = None) -> PNResult:
        t0 = time.time()
        rng = np.random.default_rng(int(geom.hash(), 16) % (2**32))
        res = toy_physics(geom, cfg, noise=self.noise, rng=rng)
        res.backend = "mock_pn (toy 1-D depletion + Soref model - not physics)"
        res.runtime_s = time.time() - t0
        if run_dir is not None:
            try:
                from ..viz import plot_geometry

                p = Path(run_dir) / "geometry.png"
                plot_geometry(geom, p, title="mock_pn run (x-z cross-section)")
                res.extra["geometry_png"] = str(p)
            except Exception as e:  # noqa: BLE001
                res.extra["plot_error"] = str(e)
        return res
