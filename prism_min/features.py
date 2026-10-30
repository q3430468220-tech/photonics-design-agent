"""Geometry features measured from the raster (shared by mock backend, agent prompts and model.py).

These are *descriptive* measurements, not physics. model.py may use them to build
predictions; the agent sees them in the journal so it can relate designs to results.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .config import SimConfig
from .geometry import Geometry, rasterize

N_SLAB = 2.8  # effective slab index used for the adiabaticity length scale


def crossing_features(geom: Geometry, cfg: SimConfig, dx: float = 0.01) -> Dict[str, float]:
    """Features of a 4-port crossing seen along the through arm (y = 0).

    w_j          width of the perpendicular arm at the junction (gap the beam crosses)
    w_c          width of the through arm where it meets the junction
    w_max        maximum width of the through arm
    taper_len    length over which the arm width changes (from port straight to junction)
    a_max        worst local adiabaticity parameter |dw/dx| * w / (lambda/n_slab)
                 (<~1 adiabatic, >>1 abrupt)
    w_port       port width
    straight_len uniform-width section behind the through port
    arm_len      port plane to junction edge
    bbox         footprint side length
    """
    r = rasterize(geom, dx, pad=0.2)
    through = geom.port(cfg.through_port)
    x, wx = r.run_profile_x(y0=through.center[1])

    # walk inward from the through port: outside the junction the y=0 run is the
    # through arm alone (run < 2x); the first column with run >= 2x is the junction edge
    order = np.argsort(-x)
    x_c = 0.0
    for i in order:
        if x[i] <= 0:
            break
        if wx[i] >= 2 * x[i] and wx[i] > 0:
            x_c = float(x[i])
            break
    w_j = 2 * x_c if x_c > 0 else 0.0

    sel = (x > x_c + dx) & (x <= through.center[0] - 2 * dx)
    xs_arm, ws_arm = x[sel], wx[sel]
    xmin, ymin, xmax, ymax = geom.bbox()
    base = {
        "w_j": w_j,
        "w_port": float(through.width),
        "arm_len": float(through.center[0] - x_c),
        "bbox": float(max(xmax - xmin, ymax - ymin)),
    }
    if len(xs_arm) < 5:
        return {**base, "w_c": 0.0, "w_max": 0.0, "taper_len": 0.0, "a_max": 10.0, "straight_len": 0.0}

    w_c = float(ws_arm[:3].mean())
    w_port = float(through.width)
    w_max = float(ws_arm.max())

    lam_eff = 0.5 * (cfg.wl_min + cfg.wl_max) / N_SLAB
    nb = max(3, int(round(0.1 / dx)))
    m = (len(ws_arm) // nb) * nb
    if m >= 3 * nb:
        ws_b = ws_arm[:m].reshape(-1, nb).mean(axis=1)
        slope = np.abs(np.gradient(ws_b, nb * dx))
        a_local = slope * ws_b / lam_eff
        a_max = float(a_local[1:-1].max()) if len(a_local) > 2 else float(a_local.max())
    else:
        a_max = 0.0

    tol = 2.5 * dx
    uniform = np.abs(ws_arm[::-1] - w_port) < tol
    n_uni = int(np.argmin(uniform)) if not uniform.all() else len(uniform)
    straight_len = n_uni * dx + 2 * dx
    taper_len = max(0.0, float(xs_arm[-1] - xs_arm[0]) - straight_len)

    return {
        **base,
        "w_c": w_c,
        "w_max": w_max,
        "taper_len": taper_len,
        "a_max": a_max,
        "straight_len": straight_len,
    }


def format_features(feat: Dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.3f}" for k, v in feat.items())
