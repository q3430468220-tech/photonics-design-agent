"""Curved and composite polygon builders for layouts (plane 'xy').

All functions return `Polygon`s in micrometres. Curves are sampled finely enough that
the 10 nm DRC raster does not see facets (`pts_per_um` points per micrometre of arc).

    arc(radius, angle, width, ...)          circular bend (annular sector)
    euler_bend(radius, angle, width, ...)   Euler / clothoid bend: curvature 0 -> 1/radius -> 0
    sbend(length, offset, width, ...)       cosine S-bend
    bezier_path(control, width)             waveguide along a cubic Bezier
    path_to_polygon(points, widths)         any centreline + width(s) -> polygon
    annulus(radius, width, center)          full ring (keyhole polygon, even-odd fill)
    periodic_array(poly, pitch, n, angle)   n copies along a direction (gratings, photonic crystals)
    mirrored(poly, axis)                    mirror about 'x' or 'y' axis
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union

import numpy as np

from .geometry import Polygon, rotated, transform

PTS_PER_UM = 40


# ----------------------------------------------------------------------------
# Centreline -> polygon
# ----------------------------------------------------------------------------


def path_to_polygon(points: np.ndarray, widths: Union[float, Sequence[float]], layer: str = "WG") -> Polygon:
    """Offset a polyline by +-width/2 along its normals (miter joins, clipped)."""
    pts = np.asarray(points, float)
    if pts.ndim != 2 or len(pts) < 2:
        raise ValueError("need an (N>=2, 2) array of centreline points")
    w = np.full(len(pts), float(widths)) if np.isscalar(widths) else np.asarray(widths, float)
    if len(w) != len(pts):
        raise ValueError("widths must be a scalar or one value per point")
    d = np.gradient(pts, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    normals = np.column_stack([-d[:, 1], d[:, 0]])
    left = pts + normals * (w / 2)[:, None]
    right = pts - normals * (w / 2)[:, None]
    return Polygon(np.vstack([left, right[::-1]]), layer)


# ----------------------------------------------------------------------------
# Bends
# ----------------------------------------------------------------------------


def arc(radius: float, angle: float, width: float, start=(0.0, 0.0), start_angle: float = 0.0, layer: str = "WG", pts_per_um: int = PTS_PER_UM) -> Polygon:
    """Circular bend of `angle` degrees (positive = counter-clockwise) starting at `start`
    heading `start_angle`. The centreline radius is `radius`."""
    n = max(8, int(abs(math.radians(angle)) * radius * pts_per_um))
    t = np.linspace(0, math.radians(angle), n)
    sgn = 1.0 if angle >= 0 else -1.0
    # centre of curvature is to the left (ccw) or right (cw) of the start heading
    a0 = math.radians(start_angle)
    cx = start[0] - sgn * radius * math.sin(a0)
    cy = start[1] + sgn * radius * math.cos(a0)
    ang = a0 - sgn * math.pi / 2 + t
    pts = np.column_stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang)])
    return path_to_polygon(pts, width, layer)


def euler_centerline(radius: float, angle: float = 90.0, p: float = 1.0, n: int = 400) -> np.ndarray:
    """Euler (clothoid) bend centreline starting at the origin heading +x.

    radius : minimum radius of curvature (reached at the middle of the bend)
    angle  : total turning angle in degrees (positive = ccw)
    p      : fraction of the bend that is clothoid (1.0 = pure Euler, 0 -> circular arc);
             the remainder is a circular arc of `radius` in the middle
    Numerically integrated (no Fresnel special functions needed).
    """
    th = math.radians(abs(angle))
    sgn = 1.0 if angle >= 0 else -1.0
    th_e = p * th / 2  # turning in each clothoid half
    th_c = th - 2 * th_e  # turning in the central arc
    # clothoid: curvature grows linearly with arc length s from 0 to 1/R over length Le
    Le = 2 * th_e * radius
    Lc = th_c * radius
    s_tot = 2 * Le + Lc
    s = np.linspace(0, s_tot, n)
    kappa = np.where(s < Le, s / (Le * radius) if Le > 0 else 0.0, np.where(s < Le + Lc, 1 / radius, (s_tot - s) / (Le * radius) if Le > 0 else 0.0))
    heading = np.concatenate([[0.0], np.cumsum(0.5 * (kappa[1:] + kappa[:-1]) * np.diff(s))]) * sgn
    x = np.concatenate([[0.0], np.cumsum(np.cos(heading[:-1]) * np.diff(s))])
    y = np.concatenate([[0.0], np.cumsum(np.sin(heading[:-1]) * np.diff(s))])
    return np.column_stack([x, y])


def euler_bend(radius: float, angle: float, width: float, start=(0.0, 0.0), start_angle: float = 0.0, p: float = 1.0, layer: str = "WG") -> Polygon:
    """Euler bend polygon; see `euler_centerline`. Effective (end-to-end) radius is larger
    than `radius` - use `euler_centerline` to read the end point if you need to route."""
    cl = euler_centerline(radius, angle, p)
    return Polygon(transform(path_to_polygon(cl, width, layer).points, start_angle, start), layer)


def sbend(length: float, offset: float, width: float, start=(0.0, 0.0), layer: str = "WG", n: int = 200) -> Polygon:
    """Cosine S-bend from `start` heading +x: ends `length` further in x and `offset` in y."""
    x = np.linspace(0, length, n)
    y = offset * 0.5 * (1 - np.cos(np.pi * x / length))
    pts = np.column_stack([x + start[0], y + start[1]])
    return path_to_polygon(pts, width, layer)


def bezier_path(control: Sequence[Sequence[float]], width: Union[float, Sequence[float]], layer: str = "WG", n: int = 200) -> Polygon:
    """Waveguide along a Bezier curve of any order given its control points."""
    P = np.asarray(control, float)
    m = len(P) - 1
    t = np.linspace(0, 1, n)[:, None]
    pts = sum(math.comb(m, i) * (1 - t) ** (m - i) * t**i * P[i] for i in range(m + 1))
    w = width if np.isscalar(width) else np.interp(t.ravel(), np.linspace(0, 1, len(width)), np.asarray(width, float))
    return path_to_polygon(pts, w, layer)


# ----------------------------------------------------------------------------
# Rings, arrays, mirrors
# ----------------------------------------------------------------------------


def annulus(radius: float, width: float, center=(0.0, 0.0), layer: str = "WG", pts_per_um: int = PTS_PER_UM) -> Polygon:
    """Full ring as one 'keyhole' polygon (outer ccw, inner cw); even-odd fill leaves the hole."""
    n = max(32, int(2 * math.pi * radius * pts_per_um))
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ro, ri = radius + width / 2, radius - width / 2
    outer = np.column_stack([center[0] + ro * np.cos(t), center[1] + ro * np.sin(t)])
    inner = np.column_stack([center[0] + ri * np.cos(t[::-1]), center[1] + ri * np.sin(t[::-1])])
    pts = np.vstack([outer, outer[:1], inner[-1:], inner, inner[-1:], outer[:1]])
    return Polygon(pts, layer)


def periodic_array(poly: Polygon, pitch: float, n: int, angle: float = 0.0) -> List[Polygon]:
    """n copies of `poly` shifted by `pitch` along direction `angle` (degrees)."""
    ux, uy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return [Polygon(poly.points + np.array([i * pitch * ux, i * pitch * uy]), poly.layer, dict(poly.props)) for i in range(n)]


def mirrored(poly: Polygon, axis: str = "x") -> Polygon:
    """Mirror about the x axis (y -> -y) or the y axis (x -> -x)."""
    pts = poly.points.copy()
    if axis == "x":
        pts[:, 1] *= -1
    elif axis == "y":
        pts[:, 0] *= -1
    else:
        raise ValueError("axis must be 'x' or 'y'")
    return Polygon(pts[::-1], poly.layer, dict(poly.props))


__all__ = ["path_to_polygon", "arc", "euler_centerline", "euler_bend", "sbend", "bezier_path", "annulus", "periodic_array", "mirrored", "rotated"]
