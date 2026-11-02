"""Analytic mock backend for the pn_ring_modulator family: mock_pn cross-section physics + toy
bend loss alpha_bend(R) = A exp(-R/R0) and toy point-coupler kappa^2(gap, R)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..families.pn_phase_shifter import toy_physics
from ..families.pn_ring_modulator import RingResult, ring_params, toy_bend_loss_db_cm, toy_coupling
from ..geometry import Geometry
from .base import Backend, register


@register("mock_ring")
class MockRing(Backend):
    def __init__(self, noise: float = 0.008, **_):
        self.noise = noise

    def run(self, geom: Geometry, cfg, run_dir: Optional[Path] = None) -> RingResult:
        t0 = time.time()
        rng = np.random.default_rng(int(geom.hash(), 16) % (2**32))
        pn = toy_physics(geom, cfg, noise=self.noise, rng=rng)
        p = ring_params(geom, cfg)
        res = RingResult(
            **{k: getattr(pn, k) for k in pn.__dataclass_fields__},
            alpha_bend_db_cm=toy_bend_loss_db_cm(p["radius_um"], cfg) * float(rng.normal(1.0, self.noise)),
            kappa2=toy_coupling(p["gap_um"], p["radius_um"], cfg),
            radius_um=p["radius_um"],
            gap_um=p["gap_um"],
            doped_fraction=p["doped_fraction"],
            coupling_model="toy",
        )
        res.backend = "mock_ring (toy PN + toy bend/coupler - not physics)"
        res.runtime_s = time.time() - t0
        if run_dir is not None:
            try:
                from ..viz import plot_geometry

                out = Path(run_dir) / "geometry.png"
                plot_geometry(geom, out, title=f"mock_ring: R {p['radius_um']:.1f} um, gap {p['gap_um']:.2f} um, f {p['doped_fraction']:.2f}")
                res.extra["geometry_png"] = str(out)
            except Exception as e:  # noqa: BLE001
                res.extra["plot_error"] = str(e)
        return res
