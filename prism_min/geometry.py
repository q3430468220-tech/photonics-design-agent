"""Geometry intermediate representation (IR).

design.py returns a `Geometry` (polygons + ports). Nothing in design.py needs to
know which simulator is used; backends translate the IR themselves.

Units: micrometres everywhere. Single device layer (thickness in `Geometry.thickness`).
Ports: `orientation` is the direction light LEAVES the device through that port
(0 = +x, 90 = +y, 180 = -x, 270 = -y). The input port therefore has orientation 180
for a device excited from the west.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.path import Path as _MplPath

# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------


@dataclass
class Port:
    name: str
    center: Tuple[float, float]
    width: float
    orientation: float  # degrees, outward

    def unit(self) -> Tuple[float, float]:
        a = math.radians(self.orientation)
        return (math.cos(a), math.sin(a))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "center": [float(self.center[0]), float(self.center[1])],
            "width": float(self.width),
            "orientation": float(self.orientation),
        }

    @staticmethod
    def from_dict(d: dict) -> "Port":
        return Port(d["name"], (d["center"][0], d["center"][1]), d["width"], d["orientation"])


# layer-name prefixes used to assign a polygon to a design "domain" (for change classification)
DOMAIN_PREFIXES = {
    "doping": ("P", "N", "PP", "NN", "P+", "N+", "P++", "N++", "DOP", "IMPL"),
    "electrode": ("M", "METAL", "VIA", "HEATER", "TIN", "AL", "AU", "CONTACT", "ELEC"),
    "stack": ("SLAB", "RIB", "ETCH", "PARTIAL", "CLAD", "BOX"),
}


def layer_domain(layer: str) -> str:
    """geometry | doping | electrode | stack, from the layer name (case-insensitive prefix match)."""
    L = layer.upper()
    if re.match(r"^M\d+", L) or re.match(r"^MH\b", L):  # M1, M2, MH ... metal levels
        return "electrode"
    for dom, prefixes in DOMAIN_PREFIXES.items():
        for pre in prefixes:
            if L == pre or L.startswith(pre + "_") or L.startswith(pre + "-") or (dom == "doping" and L in ("P", "N")):
                return dom
    return "geometry"


@dataclass
class Polygon:
    """A polygon on a named layer with optional physical properties.

    `props` carries non-geometric parameters of the region, e.g.
        {"doping": 1e18, "dopant": "p"}         for an implant window
        {"etch_depth": 0.07}                     for a partial-etch region
        {"material": "TiN", "thickness": 0.11}   for a heater
    They are part of the geometry hash and of the change classification.
    """

    points: np.ndarray  # (N, 2)
    layer: str = "WG"
    props: Dict = field(default_factory=dict)

    def __post_init__(self):
        pts = np.asarray(self.points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
            raise ValueError("Polygon needs an (N>=3, 2) array of points")
        if not np.all(np.isfinite(pts)):
            raise ValueError("Polygon contains non-finite coordinates")
        self.points = pts

    def area(self) -> float:
        x, y = self.points[:, 0], self.points[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def bbox(self) -> Tuple[float, float, float, float]:
        return (
            float(self.points[:, 0].min()),
            float(self.points[:, 1].min()),
            float(self.points[:, 0].max()),
            float(self.points[:, 1].max()),
        )

    @property
    def domain(self) -> str:
        return layer_domain(self.layer)

    def to_dict(self) -> dict:
        d = {"layer": self.layer, "points": np.round(self.points, 6).tolist()}
        if self.props:
            d["props"] = self.props
        return d

    @staticmethod
    def from_dict(d: dict) -> "Polygon":
        return Polygon(np.asarray(d["points"], dtype=float), d.get("layer", "WG"), dict(d.get("props", {})))


@dataclass
class Geometry:
    """A device as polygons + ports in one plane.

    plane = "xy": layout top view (passive devices); polygons are extruded to `thickness`.
    plane = "xz": vertical cross-section (active phase shifters); the second coordinate
                  is height above the BOX, polygons are extruded along the propagation
                  direction, `ports` is empty and contacts are polygons on a metal layer
                  with props {"terminal": "anode" | "cathode"}.
    """

    polygons: List[Polygon]
    ports: List[Port]
    thickness: float = 0.22
    meta: Dict = field(default_factory=dict)
    plane: str = "xy"

    # -- basic queries ------------------------------------------------------
    def bbox(self) -> Tuple[float, float, float, float]:
        if not self.polygons:
            raise ValueError("Geometry has no polygons")
        bs = np.array([p.bbox() for p in self.polygons])
        return (float(bs[:, 0].min()), float(bs[:, 1].min()), float(bs[:, 2].max()), float(bs[:, 3].max()))

    def port(self, name: str) -> Port:
        for p in self.ports:
            if p.name == name:
                return p
        raise KeyError(f"no port named {name!r}; have {[p.name for p in self.ports]}")

    def area(self) -> float:
        return float(sum(p.area() for p in self.polygons))

    # -- serialisation / identity -----------------------------------------
    def by_layer(self, layer: str) -> List[Polygon]:
        return [p for p in self.polygons if p.layer.upper() == layer.upper()]

    def by_domain(self, domain: str) -> List[Polygon]:
        return [p for p in self.polygons if p.domain == domain]

    def to_dict(self) -> dict:
        return {
            "plane": self.plane,
            "thickness": float(self.thickness),
            "polygons": [p.to_dict() for p in self.polygons],
            "ports": [p.to_dict() for p in self.ports],
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "Geometry":
        return Geometry(
            [Polygon.from_dict(p) for p in d["polygons"]],
            [Port.from_dict(p) for p in d["ports"]],
            d.get("thickness", 0.22),
            d.get("meta", {}),
            d.get("plane", "xy"),
        )

    def hash(self) -> str:
        """Stable identity of the geometry (coordinates rounded to 0.1 nm)."""
        payload = {
            "plane": self.plane,
            "t": round(float(self.thickness), 4),
            "polys": [[p.layer, np.round(p.points, 4).tolist(), json.dumps(p.props, sort_keys=True, default=str)] for p in self.polygons],
            "ports": [p.to_dict() for p in self.ports],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# Polygon construction helpers (agent-facing)
# ----------------------------------------------------------------------------


def rect(x0: float, y0: float, x1: float, y1: float, layer: str = "WG", props: Optional[Dict] = None) -> Polygon:
    return Polygon(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), layer, dict(props or {}))


def xz_rect(x0: float, x1: float, z0: float, z1: float, layer: str, **props) -> Polygon:
    """Axis-aligned rectangle of a cross-section (x lateral, z height above the BOX), e.g.
    xz_rect(-0.25, 0.25, 0, 0.22, "SI")                       the rib
    xz_rect(-3, 0, 0, 0.22, "P", doping=5e17, dopant="p")     an implant window (cm^-3)
    xz_rect(1.5, 2.0, 0.22, 0.5, "METAL", terminal="cathode") a contact"""
    return Polygon(np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]]), layer, dict(props))


def region(points, layer: str, **props) -> Polygon:
    """Polygon on a non-waveguide layer with physical properties, e.g.
    region(pts, "P", doping=1e18, dopant="p")   or   region(pts, "M1", material="Al")."""
    return Polygon(np.asarray(points, float), layer, dict(props))


def straight(length: float, width: float, center=(0.0, 0.0), angle: float = 0.0, layer: str = "WG") -> Polygon:
    """Rectangle of given length (along `angle`) and width, centred at `center`."""
    center = np.asarray(center, dtype=float)
    if center.shape != (2,):
        raise ValueError("center must be an (x, y) pair")
    L, W = length / 2, width / 2
    pts = np.array([[-L, -W], [L, -W], [L, W], [-L, W]])
    return Polygon(transform(pts, angle, center), layer)


def taper(x0: float, w0: float, x1: float, w1: float, y: float = 0.0, layer: str = "WG") -> Polygon:
    """Linear taper along +x from (x0, width w0) to (x1, width w1)."""
    return Polygon(np.array([[x0, y - w0 / 2], [x1, y - w1 / 2], [x1, y + w1 / 2], [x0, y + w0 / 2]]), layer)


def profile_taper(xs: Sequence[float], ws: Sequence[float], y: float = 0.0, layer: str = "WG") -> Polygon:
    """Taper defined by a sampled width profile w(x). xs must be increasing."""
    xs = np.asarray(xs, float)
    ws = np.asarray(ws, float)
    if xs.shape != ws.shape or len(xs) < 2:
        raise ValueError("xs and ws must be same length >= 2")
    if np.any(np.diff(xs) <= 0):
        raise ValueError("xs must be strictly increasing")
    top = np.column_stack([xs, y + ws / 2])
    bot = np.column_stack([xs[::-1], y - ws[::-1] / 2])
    return Polygon(np.vstack([top, bot]), layer)


def transform(points: np.ndarray, angle_deg: float = 0.0, translate=(0.0, 0.0)) -> np.ndarray:
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return np.asarray(points, float) @ R.T + np.asarray(translate, float)


def rotated(poly: Polygon, angle_deg: float, about=(0.0, 0.0)) -> Polygon:
    about = np.asarray(about, float)
    return Polygon(transform(poly.points - about, angle_deg) + about, poly.layer)


def four_fold(polys: Sequence[Polygon]) -> List[Polygon]:
    """Replicate polygons under 90-degree rotations about the origin (crossing symmetry)."""
    out: List[Polygon] = []
    for p in polys:
        for ang in (0, 90, 180, 270):
            out.append(rotated(p, ang))
    return out


def extend_ports(geom: Geometry, length: float) -> Geometry:
    """Return a copy with every port waveguide extended outward by `length`.

    Backends use this so waveguides run through the PML instead of ending inside
    the simulation region (an end facet would reflect). Ports are unchanged.
    """
    polys = list(geom.polygons)
    for p in geom.ports:
        ux, uy = p.unit()
        cx, cy = p.center
        # rectangle from the port plane outward, slightly overlapping inward to avoid a seam
        polys.append(straight(length + 0.05, p.width, (cx + ux * (length - 0.05) / 2, cy + uy * (length - 0.05) / 2), p.orientation))
    return Geometry(polys, list(geom.ports), geom.thickness, dict(geom.meta))


# ----------------------------------------------------------------------------
# Rasterisation (shared by DRC, sanity checks and the mock backend)
# ----------------------------------------------------------------------------


@dataclass
class Raster:
    mask: np.ndarray  # (ny, nx) bool
    x: np.ndarray  # pixel centres
    y: np.ndarray
    dx: float

    def width_profile_x(self) -> Tuple[np.ndarray, np.ndarray]:
        """Filled extent along y for every x column -> (x, w(x))."""
        return self.x, self.mask.sum(axis=0) * self.dx

    def width_profile_y(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.y, self.mask.sum(axis=1) * self.dx

    def run_profile_x(self, y0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Width of the contiguous filled run that contains the line y=y0, per x column.

        Unlike `width_profile_x`, this ignores other structures crossing the column
        (e.g. the perpendicular arm's taper), so it tracks one waveguide's width.
        """
        iy = int(round((y0 - self.y[0]) / self.dx))
        iy = min(max(iy, 0), self.mask.shape[0] - 1)
        up = np.cumprod(self.mask[iy:, :], axis=0).sum(axis=0)
        down = np.cumprod(self.mask[:iy, :][::-1], axis=0).sum(axis=0)
        return self.x, (up + down) * self.dx

    def run_profile_y(self, x0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        ix = int(round((x0 - self.x[0]) / self.dx))
        ix = min(max(ix, 0), self.mask.shape[1] - 1)
        right = np.cumprod(self.mask[:, ix:], axis=1).sum(axis=1)
        left = np.cumprod(self.mask[:, :ix][:, ::-1], axis=1).sum(axis=1)
        return self.y, (right + left) * self.dx

    def filled_at(self, px: float, py: float) -> bool:
        ix = int(round((px - self.x[0]) / self.dx))
        iy = int(round((py - self.y[0]) / self.dx))
        if 0 <= iy < self.mask.shape[0] and 0 <= ix < self.mask.shape[1]:
            return bool(self.mask[iy, ix])
        return False


_RASTER_CACHE: Dict[Tuple[str, float, float, Optional[str]], "Raster"] = {}


def rasterize(geom: Geometry, dx: float = 0.01, pad: float = 0.2, layer: Optional[str] = None, bounds: Optional[Tuple[float, float, float, float]] = None) -> Raster:
    """Boolean occupancy grid of the union of all polygons (even-odd fill per polygon, OR across).

    Each polygon is only tested on the sub-grid covering its own bbox, so memory
    scales with polygon area rather than device area. Results are cached per
    geometry hash (the same geometry is rasterised by DRC, sanity and the mock).
    `bounds` = (xmin, ymin, xmax, ymax) overrides the bbox+pad grid.
    """
    key = (geom.hash(), float(dx), float(pad), layer, bounds)
    hit = _RASTER_CACHE.get(key)
    if hit is not None:
        return hit
    if bounds is None:
        xmin, ymin, xmax, ymax = geom.bbox()
        xmin, ymin, xmax, ymax = xmin - pad, ymin - pad, xmax + pad, ymax + pad
    else:
        xmin, ymin, xmax, ymax = bounds
    x = np.arange(xmin, xmax + dx / 2, dx)
    y = np.arange(ymin, ymax + dx / 2, dx)
    mask = np.zeros((len(y), len(x)), dtype=bool)
    for poly in geom.polygons:
        if layer is not None and poly.layer != layer:
            continue
        bx0, by0, bx1, by1 = poly.bbox()
        ix0 = max(0, int(math.floor((bx0 - x[0]) / dx)) - 1)
        ix1 = min(len(x), int(math.ceil((bx1 - x[0]) / dx)) + 2)
        iy0 = max(0, int(math.floor((by0 - y[0]) / dx)) - 1)
        iy1 = min(len(y), int(math.ceil((by1 - y[0]) / dx)) + 2)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        X, Y = np.meshgrid(x[ix0:ix1], y[iy0:iy1])
        inside = _MplPath(poly.points).contains_points(np.column_stack([X.ravel(), Y.ravel()]))
        mask[iy0:iy1, ix0:ix1] |= inside.reshape(X.shape)
    r = Raster(mask, x, y, dx)
    if len(_RASTER_CACHE) > 16:
        _RASTER_CACHE.pop(next(iter(_RASTER_CACHE)))
    _RASTER_CACHE[key] = r
    return r


# -- 1-D / separable morphology on boolean arrays (no scipy needed) ----------


def _min_filter_1d(a: np.ndarray, k: int, axis: int) -> np.ndarray:
    """Erode along one axis by k pixels on each side (edges treated as empty)."""
    out = a.copy()
    for s in range(1, k + 1):
        fwd = np.roll(a, s, axis=axis)
        bwd = np.roll(a, -s, axis=axis)
        # roll wraps around; kill wrapped region
        idx_f = [slice(None)] * a.ndim
        idx_b = [slice(None)] * a.ndim
        idx_f[axis] = slice(0, s)
        idx_b[axis] = slice(-s, None)
        fwd[tuple(idx_f)] = False
        bwd[tuple(idx_b)] = False
        out &= fwd
        out &= bwd
    return out


def _max_filter_1d(a: np.ndarray, k: int, axis: int) -> np.ndarray:
    return ~_min_filter_1d(~a, k, axis)


def erode(mask: np.ndarray, k: int) -> np.ndarray:
    """Erosion with a (2k+1)x(2k+1) square structuring element."""
    if k <= 0:
        return mask.copy()
    return _min_filter_1d(_min_filter_1d(mask, k, 0), k, 1)


def dilate(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask.copy()
    return _max_filter_1d(_max_filter_1d(mask, k, 0), k, 1)


def opening(mask: np.ndarray, k: int) -> np.ndarray:
    return dilate(erode(mask, k), k)


def closing(mask: np.ndarray, k: int) -> np.ndarray:
    return erode(dilate(mask, k), k)


def label_components(mask: np.ndarray, max_components: int = 64) -> Tuple[np.ndarray, int]:
    """4-connected component labelling by vectorised region growing (numpy only).

    Each component is grown from a seed by repeated 1-pixel dilation masked by the
    occupancy grid. Cost ~ (component diameter in pixels) vectorised passes, so call
    it on a coarse grid (e.g. 50 nm) rather than the DRC grid.
    """
    labels = np.zeros(mask.shape, dtype=np.int32)
    remaining = mask.copy()
    current = 0
    while remaining.any() and current < max_components:
        current += 1
        seed = np.argwhere(remaining)[0]
        comp = np.zeros(mask.shape, dtype=bool)
        comp[seed[0], seed[1]] = True
        while True:
            grown = dilate(comp, 1) & remaining
            if grown.sum() == comp.sum():
                break
            comp = grown
        labels[comp] = current
        remaining &= ~comp
    return labels, current


# ----------------------------------------------------------------------------
# Optional: gdsfactory import
# ----------------------------------------------------------------------------


def from_gdsfactory(component, layer=(1, 0), port_names: Optional[Sequence[str]] = None, thickness: float = 0.22) -> Geometry:
    """Convert a gdsfactory Component to the IR (optional dependency)."""
    polys: List[Polygon] = []
    polygons = component.get_polygons(by="tuple") if hasattr(component, "get_polygons") else {}
    pts_list = polygons.get(layer, []) if isinstance(polygons, dict) else polygons
    for pts in pts_list:
        polys.append(Polygon(np.asarray(pts, float), "WG"))
    ports: List[Port] = []
    for p in component.ports:
        if port_names and p.name not in port_names:
            continue
        ports.append(Port(p.name, (float(p.center[0]), float(p.center[1])), float(p.width), float(p.orientation)))
    return Geometry(polys, ports, thickness)
