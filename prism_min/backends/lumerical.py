"""Lumerical FDTD backend (lumapi).

STATUS: written against the lumapi conventions of FDTD 2023-2025 but NOT executed
in the environment where this package was authored (no Lumerical there). Validate
once on your machine with

    prism eval experiments/soi_crossing --backend lumerical --manual

and compare the printed breakdown + closure with expectations before letting the
agent loop consume simulations. Things most likely to need a one-line fix are the
material names and the mode-expansion result keys (T_forward / T_backward).

Simulation layout for a 4-port crossing excited from the input port (W):

    - all port waveguides are extended through the PML (`extend_ports`)
    - mode source at W, `port_offset` inside the port plane, forward injection
    - 'back' power monitor at W between source and PML -> reflection
    - power + mode-expansion monitors just inside E (through), N, S (crosstalk)
    - 6 power monitors forming a closed box around the device (+ `box_margin`)
      -> box_total (closure) and radiation = box_total - guided terms
    - z-normal field monitor at z=0 for the field plot

Everything numerical (mesh, PML margin, time, shutoff, band) comes from SimConfig.
The agent never touches this file.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import SimConfig
from ..geometry import Geometry, extend_ports
from .base import Backend, SimResult, register

UM = 1e-6


def _import_lumapi():
    root = os.environ.get("LUMERICAL_ROOT", r"D:/Program Files/ANSYS Inc/v251")
    api = os.path.join(root, "api", "python")
    if api not in sys.path:
        sys.path.append(api)
    try:
        import lumapi  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(f"lumapi not importable from {api}. Set LUMERICAL_ROOT to the folder containing api/python/lumapi.py") from e
    return lumapi


def _check_ascii(path: Path) -> None:
    s = str(path)
    if any(ord(c) > 127 for c in s):
        raise ValueError(f"Lumerical cannot handle non-ASCII paths: {s}")


@register("lumerical")
class LumericalFDTD(Backend):
    """Long-lived FDTD session; call `close()` (or use the CLI, which does) when done."""

    def __init__(self, hide: bool = True, **_):
        self.hide = hide
        self._fdtd = None

    # -- session ---------------------------------------------------------------
    def session(self):
        if self._fdtd is None:
            lumapi = _import_lumapi()
            self._fdtd = lumapi.FDTD(hide=self.hide)
        return self._fdtd

    def close(self) -> None:
        if self._fdtd is not None:
            try:
                self._fdtd.close()
            except Exception:
                pass
            self._fdtd = None

    # -- build ----------------------------------------------------------------
    def _build(self, f, geom: Geometry, cfg: SimConfig) -> dict:
        f.switchtolayout()
        f.selectall()
        f.delete()

        xmin, ymin, xmax, ymax = geom.bbox()  # device bbox (ports on its edges)
        ext = extend_ports(geom, cfg.pml_margin + 1.0)
        t = geom.thickness * UM

        # cladding block (SiO2 everywhere, lowest priority)
        f.addrect(name="clad")
        f.set("x min", (xmin - cfg.pml_margin - 2) * UM)
        f.set("x max", (xmax + cfg.pml_margin + 2) * UM)
        f.set("y min", (ymin - cfg.pml_margin - 2) * UM)
        f.set("y max", (ymax + cfg.pml_margin + 2) * UM)
        f.set("z", 0.0)
        f.set("z span", (cfg.z_span + 2) * UM)
        f.set("material", cfg.clad_material)
        f.set("override mesh order from material database", 1)
        f.set("mesh order", 3)
        f.set("alpha", 0.2)

        # device polygons
        for i, poly in enumerate(ext.polygons):
            f.addpoly(name=f"wg_{i}")
            f.set("x", 0.0)
            f.set("y", 0.0)
            f.set("z", 0.0)
            f.set("z span", t)
            f.set("vertices", np.asarray(poly.points, float) * UM)
            f.set("material", cfg.core_material)

        # FDTD region
        f.addfdtd()
        f.set("x min", (xmin - cfg.pml_margin) * UM)
        f.set("x max", (xmax + cfg.pml_margin) * UM)
        f.set("y min", (ymin - cfg.pml_margin) * UM)
        f.set("y max", (ymax + cfg.pml_margin) * UM)
        f.set("z", 0.0)
        f.set("z span", cfg.z_span * UM)
        f.set("mesh accuracy", cfg.mesh_accuracy)
        f.set("simulation time", cfg.sim_time_fs * 1e-15)
        f.set("auto shutoff min", cfg.auto_shutoff)

        # mesh override over the device
        f.addmesh(name="mesh_device")
        f.set("x min", (xmin - 0.2) * UM)
        f.set("x max", (xmax + 0.2) * UM)
        f.set("y min", (ymin - 0.2) * UM)
        f.set("y max", (ymax + 0.2) * UM)
        f.set("z", 0.0)
        f.set("z span", (geom.thickness + 0.4) * UM)
        f.set("dx", cfg.mesh_dx * UM)
        f.set("dy", cfg.mesh_dx * UM)
        f.set("dz", cfg.mesh_dz * UM)

        # global monitor / source settings
        f.setglobalsource("wavelength start", cfg.wl_min * UM)
        f.setglobalsource("wavelength stop", cfg.wl_max * UM)
        f.setglobalmonitor("use source limits", 1)
        f.setglobalmonitor("frequency points", cfg.n_wl)

        pin = geom.port(cfg.input_port)
        ux, uy = pin.unit()  # outward; light travels along -u
        mon_span = (pin.width + 2.0) * UM
        z_mon = (cfg.z_span - 0.4) * UM

        # mode source, `port_offset` inside the input port
        f.addmode(name="src")
        self._place_plane(f, pin, inset=cfg.port_offset, span=mon_span, zspan=z_mon)
        f.set("direction", "Forward" if (ux < 0 or uy < 0) else "Backward")  # into the device
        f.set("mode selection", f"fundamental {cfg.polarization} mode")

        # reflection monitor between source and PML
        f.addpower(name="mon_back")
        self._place_plane(f, pin, inset=cfg.port_offset / 2, span=mon_span, zspan=z_mon)

        # through + side port monitors with mode expansion
        names = {}
        for pname in [cfg.through_port, *cfg.side_ports]:
            p = geom.port(pname)
            mon = f"mon_{pname}"
            f.addpower(name=mon)
            self._place_plane(f, p, inset=cfg.port_offset, span=(p.width + 2.0) * UM, zspan=z_mon)
            exp = f"exp_{pname}"
            f.addmodeexpansion(name=exp)
            self._place_plane(f, p, inset=cfg.port_offset, span=(p.width + 2.0) * UM, zspan=z_mon)
            f.set("mode selection", f"fundamental {cfg.polarization} mode")
            f.setexpansion(pname, mon)
            names[pname] = (mon, exp)

        # closed flux box
        bm = cfg.box_margin
        bx0, bx1 = (xmin - bm) * UM, (xmax + bm) * UM
        by0, by1 = (ymin - bm) * UM, (ymax + bm) * UM
        bz = (cfg.z_span - 0.6) * UM
        for name, mtype, pos in [
            ("box_xmin", "2D X-normal", ("x", bx0)),
            ("box_xmax", "2D X-normal", ("x", bx1)),
            ("box_ymin", "2D Y-normal", ("y", by0)),
            ("box_ymax", "2D Y-normal", ("y", by1)),
            ("box_zmin", "2D Z-normal", ("z", -bz / 2)),
            ("box_zmax", "2D Z-normal", ("z", +bz / 2)),
        ]:
            f.addpower(name=name)
            f.set("monitor type", mtype)
            f.set(pos[0], pos[1])
            if mtype != "2D X-normal":
                f.set("x min", bx0)
                f.set("x max", bx1)
            if mtype != "2D Y-normal":
                f.set("y min", by0)
                f.set("y max", by1)
            if mtype != "2D Z-normal":
                f.set("z", 0.0)
                f.set("z span", bz)

        # field plot
        f.addpower(name="field_xy")
        f.set("monitor type", "2D Z-normal")
        f.set("x min", xmin * UM)
        f.set("x max", xmax * UM)
        f.set("y min", ymin * UM)
        f.set("y max", ymax * UM)
        f.set("z", 0.0)
        f.set("override global monitor settings", 1)
        f.set("frequency points", 1)
        return names

    @staticmethod
    def _place_plane(f, port, inset: float, span: float, zspan: float) -> None:
        """Place the currently selected planar object `inset` inside `port`, normal to it."""
        ux, uy = port.unit()
        cx, cy = port.center[0] - ux * inset, port.center[1] - uy * inset
        if abs(ux) > 0.5:
            try:
                f.set("injection axis", "x-axis")
            except Exception:
                f.set("monitor type", "2D X-normal")
            f.set("x", cx * UM)
            f.set("y", cy * UM)
            f.set("y span", span)
        else:
            try:
                f.set("injection axis", "y-axis")
            except Exception:
                f.set("monitor type", "2D Y-normal")
            f.set("x", cx * UM)
            f.set("y", cy * UM)
            f.set("x span", span)
        f.set("z", 0.0)
        f.set("z span", zspan)

    # -- run ------------------------------------------------------------------
    def run(self, geom: Geometry, cfg: SimConfig, run_dir: Optional[Path] = None) -> SimResult:
        t0 = time.time()
        f = self.session()
        names = self._build(f, geom, cfg)
        if run_dir is not None:
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            _check_ascii(run_dir.resolve())
            f.save(str((run_dir / "sim.fsp").resolve()))
        f.run()

        wl = np.asarray(f.getresult("mon_" + cfg.through_port, "T")["lambda"]).ravel() / UM
        order = np.argsort(wl)
        wl = wl[order]

        def T(mon: str) -> np.ndarray:  # net transmission normalised to source power, along +axis
            return np.asarray(f.transmission(mon)).ravel()[order]

        def outward(mon: str, port) -> np.ndarray:
            ux, uy = port.unit()
            sign = 1.0 if (ux > 0 or uy > 0) else -1.0
            return sign * T(mon)

        pin = geom.port(cfg.input_port)
        # reflection: power travelling outward (away from device) at the input side
        reflection = outward("mon_back", pin)

        pth = geom.port(cfg.through_port)
        through_total = outward(names[cfg.through_port][0], pth)
        exp = f.getresult(names[cfg.through_port][1], "expansion for " + cfg.through_port)
        key = "T_forward" if (pth.unit()[0] > 0 or pth.unit()[1] > 0) else "T_backward"
        through_fund = np.abs(np.asarray(exp[key]).ravel()[order])

        side = {}
        for pname in cfg.side_ports:
            side[pname] = outward(names[pname][0], geom.port(pname))

        box = (
            -T("box_xmin") + T("box_xmax") - T("box_ymin") + T("box_ymax") - T("box_zmin") + T("box_zmax")
        )

        extra = {"fsp": str(run_dir / "sim.fsp") if run_dir else None}
        if run_dir is not None:
            try:
                extra["field_png"] = str(self._save_field_png(f, run_dir, cfg))
            except Exception as e:  # pragma: no cover
                extra["field_png_error"] = str(e)

        return SimResult(
            wavelengths=wl,
            through_total=through_total,
            through_fund=through_fund,
            side=side,
            reflection=reflection,
            box_total=box,
            converged=True,  # evaluate() decides from closure
            backend="lumerical-fdtd",
            runtime_s=time.time() - t0,
            extra=extra,
        )

    @staticmethod
    def _save_field_png(f, run_dir: Path, cfg: SimConfig) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        res = f.getresult("field_xy", "E")
        E = np.asarray(res["E"])  # (nx, ny, nz, nf, 3)
        I = np.sum(np.abs(E[:, :, 0, 0, :]) ** 2, axis=-1)
        x = np.asarray(res["x"]).ravel() / UM
        y = np.asarray(res["y"]).ravel() / UM
        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        ax.pcolormesh(x, y, I.T, shading="auto", cmap="inferno")
        ax.set_aspect("equal")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_title("|E|^2 at z=0 (band centre)", fontsize=9)
        p = run_dir / "field_xy.png"
        fig.tight_layout()
        fig.savefig(p)
        plt.close(fig)
        return p
