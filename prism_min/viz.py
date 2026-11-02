"""Plotting (matplotlib, best-effort): geometry previews and the evolution chart."""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

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


# ----------------------------------------------------------------------------
# Evolution chart
# ----------------------------------------------------------------------------

# marker = KIND of change (parameter / shape / topology); fill colour = outcome (kept / discarded);
# edge colour of kept points = DOMAIN (geometry / doping / stack / electrode); non-geometry
# domains are also written as a prefix in the label ("doping | raise p to 1e18").
MARKERS = {"baseline": "o", "parameter": "o", "shape": "s", "topology": "D", "unspecified": "o", "none": "o"}
DOMAIN_EDGE = {"geometry": "black", "doping": "#6a1b9a", "stack": "#e65100", "electrode": "#795548", "other": "#555555"}
KEEP_COLOR = "#1b7f5a"
DISCARD_COLOR = "#c0392b"
VOID_COLOR = "#8e8e8e"
MODEL_COLOR = "#2b6cb0"


def plot_evolution(entries: List[dict], path: Path, cfg=None, title: Optional[str] = None, log_y: Optional[bool] = None, label_discarded: bool = False) -> Optional[Path]:
    """FOM vs experiment number with the agent's *method* annotated.

    x axis   : experiment index = simulations spent (checks / model turns cost nothing)
    y axis   : FOM (worst-case IL, dB; lower is better)
    markers  : o parameter change, s shape change, D topology change
    colours  : green kept, red discarded, grey voided (closure failed)
    labels   : the agent's own 3-8 word description of what it did and why, on every kept
               point (and on discarded points if label_discarded)
    vertical : blue dashed lines = model.py revised between experiments
    """
    sims = [e for e in entries if e.get("sim_spent")]
    if not sims:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    from .agent import split_change_type

    xs, ys, kinds, domains, decisions, labels, iters = [], [], [], [], [], [], []
    for i, e in enumerate(sims, start=1):
        dom, kind = split_change_type(e.get("change_type") or "unspecified")
        xs.append(i)
        ys.append(e.get("fom"))
        kinds.append(kind)
        domains.append(dom)
        decisions.append(e.get("decision") if e.get("valid") else "void")
        lab = e.get("label") or (e.get("change") or "")[:40]
        if dom not in ("geometry",) and kind not in ("baseline", "unspecified"):
            lab = f"{dom} | {lab}"
        labels.append(lab)
        iters.append(e["iteration"])

    # model turns: position between the experiments they fall between
    model_x = []
    count = 0
    for e in entries:
        if e.get("sim_spent"):
            count += 1
        elif e.get("mode") == "model" and e.get("decision") == "model updated":
            model_x.append(count + 0.5)

    # best-so-far staircase
    best_x, best_y, best = [], [], None
    for x, y, d in zip(xs, ys, decisions):
        if d == "keep":
            best = y
        if best is not None:
            best_x.append(x)
            best_y.append(best)

    valid_y = [y for y in ys if y is not None]
    ymin, ymax = min(valid_y), max(valid_y)
    if log_y is None:  # auto: a log axis when the FOM spans more than ~5x (typical 1 dB -> 0.06 dB runs)
        log_y = ymin > 0 and ymax / max(ymin, 1e-9) > 5
    void_y = ymax * 1.08 if not log_y else ymax * 1.3

    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=130)
    if best_x:
        ax.step(best_x, best_y, where="post", color=KEEP_COLOR, lw=2, zorder=2, label="best so far")

    for x, y, k, dom, d in zip(xs, ys, kinds, domains, decisions):
        m = MARKERS.get(k, "o")
        if d == "void":
            ax.scatter([x], [void_y], marker="x", s=46, color=VOID_COLOR, zorder=3)
            continue
        if d == "keep":
            edge = DOMAIN_EDGE.get(dom, "black")
            ax.scatter([x], [y], marker=m, s=58, facecolor=KEEP_COLOR, edgecolor=edge, lw=0.8 if edge == "black" else 1.8, zorder=4)
        else:
            ax.scatter([x], [y], marker=m, s=46, facecolor="none", edgecolor=DISCARD_COLOR, lw=1.3, zorder=3)

    for mx in model_x:
        ax.axvline(mx, color=MODEL_COLOR, ls="--", lw=1, alpha=0.7, zorder=1)
        ax.text(mx, ymax if not log_y else ymax * 1.15, "model\nrevised", color=MODEL_COLOR, fontsize=7, ha="center", va="bottom")

    # annotations: every kept point (alternate above/below to limit overlap), big jumps in bold
    prev_best = None
    toggle = 0
    span = (ymax - ymin) if not log_y else None
    for x, y, d, lab, k in zip(xs, ys, decisions, labels, kinds):
        if d == "void":
            continue
        show = d == "keep" or label_discarded
        if not show:
            continue
        gain = (prev_best - y) if (d == "keep" and prev_best is not None) else 0.0
        big = prev_best is not None and gain > 0.15 * max(prev_best, 1e-9)
        if d == "keep":
            prev_best = y
        txt = "\n".join(textwrap.wrap(lab, 28)[:2])
        if big:
            txt = f"{txt}\n(−{gain:.3f} dB)"
        dy = (22 if toggle % 2 == 0 else -26)
        toggle += 1
        ax.annotate(
            txt,
            (x, y),
            xytext=(6, dy),
            textcoords="offset points",
            fontsize=7.2,
            fontweight="bold" if big else "normal",
            color=KEEP_COLOR if d == "keep" else DISCARD_COLOR,
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.5},
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.75},
            zorder=5,
        )

    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(ymin * 0.6, (void_y if decisions.count("void") else ymax) * 1.6)
    else:
        ax.set_ylim(max(0.0, ymin - 0.08 * (ymax - ymin + 1e-9)), void_y * 1.12 if decisions.count("void") else ymax * 1.12)
    ax.set_xlabel("Experiment (simulations spent)")
    fom_name = getattr(cfg, "fom_description", "FOM") if cfg is not None else "FOM"
    ax.set_ylabel("FOM: " + fom_name.split("(")[0].strip() + " (dB, lower is better)", fontsize=9)
    ax.set_title(title or "Design evolution - what the agent changed and what it bought", fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_xlim(0.3, max(xs) + 0.7)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    handles = [
        Line2D([], [], color=KEEP_COLOR, lw=2, label="best so far"),
        Line2D([], [], marker="o", ls="", markerfacecolor=KEEP_COLOR, markeredgecolor="black", label="kept"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=DISCARD_COLOR, label="discarded"),
        Line2D([], [], marker="x", ls="", color=VOID_COLOR, label="voided (closure failed)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="#dddddd", markeredgecolor="black", label="parameter change"),
        Line2D([], [], marker="s", ls="", markerfacecolor="#dddddd", markeredgecolor="black", label="shape change (same elements)"),
        Line2D([], [], marker="D", ls="", markerfacecolor="#dddddd", markeredgecolor="black", label="topology change (elements added/replaced)"),
        Line2D([], [], color=MODEL_COLOR, ls="--", label="model.py revised"),
    ]
    used_domains = {dom for dom, d in zip(domains, decisions) if d == "keep" and dom != "geometry"}
    for dom in ("doping", "stack", "electrode", "other"):
        if dom in used_domains:
            handles.append(Line2D([], [], marker="o", ls="", markerfacecolor=KEEP_COLOR, markeredgecolor=DOMAIN_EDGE[dom], markeredgewidth=1.8, label=f"{dom} change (edge colour)"))
    ax.legend(handles=handles, fontsize=7.5, loc="upper right", ncol=2, framealpha=0.9).set_zorder(6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
