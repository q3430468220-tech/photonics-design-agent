"""Analytic mock backend for a 4-port waveguide crossing.

Purpose: exercise the whole agent loop (checks, guard, decomposition, closure,
keep/revert, journal) without a real solver. It is a *toy* physical model and is
labelled as such in every result. Do not draw physics conclusions from it.

What it does with the geometry (measured from the raster, not from meta hints):
  w_j   : width of the perpendicular arm at the junction (the unguided gap the
          through-beam must cross)
  w_c   : width of the through arm where it meets the junction
  a_max : worst local taper adiabaticity parameter |dw/dx| * w / (lambda/n)
  w_port: width at the through port

Toy terms (fractions of injected power):
  diffraction across the gap  F = w_j * (lambda/n_slab) / w_c^2
      radiation  = 0.18 F^2 / (0.5 + F^2)
      crosstalk  = 0.05 F^3 / (0.1 + F^3)   per side port
  taper mismatch               = 0.015 a^2 / (2 + a^2)
  reflection                   = 0.0002 + 0.005 (a/(1+a))^2
With the default 40 um bbox limit the toy optimum is a ~4.4 um parabolic-taper
crossing at roughly 0.06 dB worst-case IL and -30 dB crosstalk.
Everything is deterministic for a given geometry (seeded by its hash) with a
2 % multiplicative "mesh noise", and the closure box reads 1 +- 0.3 % unless the
port straights are too short (then the run is reported unconverged).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import SimConfig
from ..features import N_SLAB, crossing_features
from ..geometry import Geometry
from .base import Backend, SimResult, register

def measure_crossing(geom: Geometry, cfg: SimConfig, dx: float = 0.01) -> dict:
    """Toy-model inputs = the shared raster features (see prism_min.features)."""
    return crossing_features(geom, cfg, dx)


@register("mock")
class MockCrossing(Backend):
    def __init__(self, noise: float = 0.02, **_):
        self.noise = noise

    def run(self, geom: Geometry, cfg: SimConfig, run_dir: Optional[Path] = None) -> SimResult:
        t0 = time.time()
        feat = measure_crossing(geom, cfg)
        wl = cfg.wavelengths()
        rng = np.random.default_rng(int(geom.hash(), 16) % (2**32))

        w_j, w_c, a, w_port = feat["w_j"], max(feat["w_c"], 0.05), feat["a_max"], feat["w_port"]
        lam_eff = wl / N_SLAB
        F = w_j * lam_eff / w_c**2 if w_j > 0 else np.zeros_like(wl)

        rad = 0.18 * F**2 / (0.5 + F**2)
        xt = 0.05 * F**3 / (0.1 + F**3)
        mis = 0.015 * a**2 / (2.0 + a**2) * np.ones_like(wl)
        refl = (0.0002 + 0.005 * (a / (1 + a)) ** 2) * np.ones_like(wl)
        # slight wavelength dependence of the taper term (longer wavelength = more adiabatic)
        mis = mis * (1.55 / wl) ** 2

        def jitter(v):
            return v * rng.normal(1.0, self.noise, size=v.shape)

        rad, xt, mis, refl = jitter(rad), jitter(xt), jitter(mis), jitter(refl)
        through_fund = 1.0 - rad - 2 * xt - mis - refl
        higher = 0.5 * mis if w_port > 0.6 else 1e-4 * np.ones_like(wl)
        through_fund = through_fund - higher
        through_total = through_fund + higher

        converged = feat["straight_len"] >= 0.8 * cfg.port_straight_len
        # everything not guided out of a port leaves through the flux box as radiation
        box = through_total + 2 * xt + refl + rad + mis + rng.normal(0.0, 0.003, size=wl.shape)
        if not converged:
            box = box - 0.06  # source overlaps the taper: injected power mis-normalised

        res = SimResult(
            wavelengths=wl,
            through_total=through_total,
            through_fund=through_fund,
            side={name: xt.copy() for name in cfg.side_ports},
            reflection=refl,
            box_total=box,
            converged=bool(converged),
            backend="mock (toy analytic model - not physics)",
            runtime_s=time.time() - t0,
            extra={
                "features": {k: float(v) for k, v in feat.items() if k != "profile"},
                # hidden ground truth of the toy (for debugging the loop; never shown to the agent)
                "_toy_terms": {"junction_radiation": rad.tolist(), "taper_mismatch": mis.tolist()},
            },
        )
        if run_dir is not None:
            try:
                from ..viz import plot_geometry

                p = Path(run_dir) / "geometry.png"
                plot_geometry(geom, p, title="mock run")
                res.extra["geometry_png"] = str(p)
            except Exception as e:  # plotting is best-effort
                res.extra["plot_error"] = str(e)
        return res
