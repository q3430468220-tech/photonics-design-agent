"""Geometry plotting (matplotlib, best-effort)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from .geometry import Geometry  # noqa: E402


def plot_geometry(geom: Geometry, path: Path, title: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    for p in geom.polygons:
        ax.add_patch(MplPolygon(p.points, closed=True, facecolor="#9fb3c8", edgecolor="#1f3b5a", lw=0.6))
    for pt in geom.ports:
        ux, uy = pt.unit()
        ax.annotate(pt.name, pt.center, xytext=(pt.center[0] + 0.6 * ux, pt.center[1] + 0.6 * uy), fontsize=8, ha="center", va="center", color="#b03a2e")
    xmin, ymin, xmax, ymax = geom.bbox()
    pad = 0.1 * max(xmax - xmin, ymax - ymin) + 0.5
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
