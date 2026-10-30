"""Design-rule checks (DRC) and pre-simulation sanity checks.

Both return a list of `Violation`. DRC answers "can this be fabricated?";
sanity answers "will a simulation of this be trustworthy?". They are kept apart
because their failures mean different things to the agent: a DRC failure is a bad
design, a sanity failure is a bad measurement (the run is voided, not scored).

Implementation is raster-based (numpy only) so it runs anywhere. A square
structuring element is used, so rules are exact for axis-aligned features and
slightly conservative for diagonal ones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import SimConfig
from .geometry import Geometry, Raster, closing, label_components, opening, rasterize


@dataclass
class Violation:
    rule: str
    message: str
    where: Optional[Tuple[float, float]] = None
    severity: str = "error"  # error | warning

    def __str__(self) -> str:
        loc = f" @({self.where[0]:.2f},{self.where[1]:.2f})" if self.where else ""
        return f"[{self.severity}] {self.rule}: {self.message}{loc}"


def _centroid(mask: np.ndarray, r: Raster) -> Optional[Tuple[float, float]]:
    iy, ix = np.nonzero(mask)
    if len(ix) == 0:
        return None
    return (float(r.x[ix].mean()), float(r.y[iy].mean()))


# ----------------------------------------------------------------------------
# DRC
# ----------------------------------------------------------------------------


def run_drc(geom: Geometry, cfg: SimConfig) -> List[Violation]:
    v: List[Violation] = []
    if not geom.polygons:
        return [Violation("geometry", "no polygons")]

    xmin, ymin, xmax, ymax = geom.bbox()
    if (xmax - xmin) > cfg.max_extent or (ymax - ymin) > cfg.max_extent:
        v.append(Violation("extent", f"bbox {xmax-xmin:.1f} x {ymax-ymin:.1f} um exceeds {cfg.max_extent} um"))
        return v  # rasterising a huge geometry is pointless

    r = rasterize(geom, cfg.raster_dx, pad=max(cfg.min_gap, cfg.min_width) + 0.1)
    px_area = r.dx * r.dx

    # min width: pixels removed by an opening of half-width min_width/2
    k_w = int(math.ceil(cfg.min_width / 2 / r.dx))
    narrow = r.mask & ~opening(r.mask, k_w)
    narrow_area = narrow.sum() * px_area
    if narrow_area > 0.25 * cfg.min_width**2:
        v.append(Violation("min_width", f"features narrower than {cfg.min_width} um (area {narrow_area:.4f} um^2)", _centroid(narrow, r)))

    # min gap: pixels added by a closing of half-width min_gap/2
    k_g = int(math.ceil(cfg.min_gap / 2 / r.dx))
    gap = closing(r.mask, k_g) & ~r.mask
    gap_area = gap.sum() * px_area
    if gap_area > 0.25 * cfg.min_gap**2:
        v.append(Violation("min_gap", f"gaps narrower than {cfg.min_gap} um (area {gap_area:.4f} um^2)", _centroid(gap, r)))

    # min area per connected island / connectivity (coarse grid: labelling is the slow part)
    rc = rasterize(geom, max(cfg.raster_dx * 5, 0.05), pad=0.1)
    labels, n = label_components(rc.mask)
    for i in range(1, n + 1):
        comp = labels == i
        a = comp.sum() * rc.dx * rc.dx
        if a < cfg.min_area:
            v.append(Violation("min_area", f"island of {a:.4f} um^2 < {cfg.min_area} um^2", _centroid(comp, rc)))
    if n > 1:
        v.append(Violation("connectivity", f"{n} disconnected islands (expected 1 for a crossing)", None, "warning"))

    # ports: must sit on the bbox boundary, on material, with matching width
    for p in geom.ports:
        ux, uy = p.unit()
        cx, cy = p.center
        edge_dist = min(abs(cx - xmin), abs(cx - xmax), abs(cy - ymin), abs(cy - ymax))
        if edge_dist > 2 * r.dx:
            v.append(Violation("port_on_edge", f"port {p.name} is {edge_dist:.3f} um from the bbox edge", (cx, cy)))
        inside = (cx - ux * 3 * r.dx, cy - uy * 3 * r.dx)
        if not r.filled_at(*inside):
            v.append(Violation("port_on_material", f"port {p.name} is not on a waveguide", (cx, cy)))
        w_meas = _width_at(r, p, 3 * r.dx)
        if w_meas is not None and abs(w_meas - p.width) > max(0.05 * p.width, 2 * r.dx):
            v.append(Violation("port_width", f"port {p.name} width {p.width:.3f} um but waveguide there is {w_meas:.3f} um", (cx, cy)))
    return v


def _width_at(r: Raster, p, inset: float) -> Optional[float]:
    """Waveguide width in the port cross-section plane, `inset` inside the port
    (contiguous filled run through the port centre line)."""
    ux, uy = p.unit()
    cx, cy = p.center[0] - ux * inset, p.center[1] - uy * inset
    if abs(ux) > 0.5:  # port faces +-x -> measure run along y at column x=cx
        ix = int(round((cx - r.x[0]) / r.dx))
        if not (0 <= ix < r.mask.shape[1]):
            return None
        _, prof = r.run_profile_x(y0=cy)
        return float(prof[ix])
    iy = int(round((cy - r.y[0]) / r.dx))
    if not (0 <= iy < r.mask.shape[0]):
        return None
    _, prof = r.run_profile_y(x0=cx)
    return float(prof[iy])


# ----------------------------------------------------------------------------
# Pre-simulation sanity
# ----------------------------------------------------------------------------


def sim_sanity(geom: Geometry, cfg: SimConfig) -> List[Violation]:
    v: List[Violation] = []
    needed = {cfg.input_port, cfg.through_port, *cfg.side_ports}
    have = {p.name for p in geom.ports}
    missing = needed - have
    if missing:
        v.append(Violation("ports", f"missing ports {sorted(missing)}; have {sorted(have)}"))
        return v

    if abs(geom.thickness - cfg.thickness) > 1e-6:
        v.append(Violation("thickness", f"design thickness {geom.thickness} != locked {cfg.thickness}"))

    r = rasterize(geom, cfg.raster_dx, pad=0.3)
    px_area = r.dx * r.dx

    # smallest feature must span >= N mesh cells
    feat = cfg.min_mesh_cells_per_feature * cfg.mesh_dx
    k = int(math.ceil(feat / 2 / r.dx))
    fine = r.mask & ~opening(r.mask, k)
    fine_area = fine.sum() * px_area
    if fine_area > 0.25 * feat**2:
        v.append(Violation("mesh_resolution", f"features < {feat:.3f} um ({cfg.min_mesh_cells_per_feature} mesh cells of {cfg.mesh_dx} um) cannot be resolved", _centroid(fine, r)))

    # ports must have a uniform straight section inside them (mode source / expansion validity)
    for p in geom.ports:
        ws = _port_profile(r, p, cfg.port_straight_len)
        if len(ws) and (ws.max() - ws.min()) > 2.5 * r.dx:
            v.append(Violation("port_straight", f"port {p.name}: width varies {ws.min():.3f}-{ws.max():.3f} um within {cfg.port_straight_len} um of the port plane", p.center))

    # 4-fold symmetry (only the W->E path is simulated; the device must be identical from every port)
    if cfg.require_fourfold_symmetry:
        xmin, ymin, xmax, ymax = geom.bbox()
        half = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax)) + 0.1
        rs = rasterize(geom, 0.02, bounds=(-half, -half, half, half))
        m = rs.mask
        n = min(m.shape)
        m = m[:n, :n]
        mismatch = (m ^ np.rot90(m)).sum() / max(m.sum(), 1)
        if mismatch > cfg.symmetry_tol:
            v.append(Violation("symmetry", f"device is not 4-fold symmetric about the origin ({100*mismatch:.1f}% of pixels differ under 90-degree rotation)"))
    return v


def _port_profile(r: Raster, p, depth: float) -> np.ndarray:
    """Waveguide width sampled from the port plane inward over `depth` (one profile call)."""
    ux, uy = p.unit()
    cx, cy = p.center
    if abs(ux) > 0.5:
        xs, prof = r.run_profile_x(y0=cy)
        d = (cx - xs) * ux  # distance inside the port along -u
        sel = (d > 2 * r.dx) & (d < depth)
        return prof[sel]
    ys, prof = r.run_profile_y(x0=cx)
    d = (cy - ys) * uy
    sel = (d > 2 * r.dx) & (d < depth)
    return prof[sel]


def errors_only(viols: List[Violation]) -> List[Violation]:
    return [x for x in viols if x.severity == "error"]


def format_violations(viols: List[Violation]) -> str:
    return "\n".join(str(x) for x in viols) if viols else "none"
