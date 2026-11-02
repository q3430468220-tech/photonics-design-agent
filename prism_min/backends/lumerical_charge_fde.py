"""Lumerical CHARGE + MODE(FDE) backend for the pn_phase_shifter family (lumapi).

STATUS: ported from the user's working .lsf scripts (01_MRM_PN_CHARGE.lsf, 02_MRM_PN_MODE_FDE.lsf)
to lumapi, but NOT executed where this package was authored. Validate with

    prism eval experiments/soi_pn_phase_shifter --manual --real

Chain for one geometry (x-z cross-section, propagation along y):
  CHARGE  : Si/SiO2/Al CT materials with Masetti + high-field mobility, Slotboom BGN, SRH/Auger;
            rib/slab/metal rectangles, doping boxes from the implant windows, 2D y-normal region,
            anode swept 0 -> v_min (cathode grounded), charge monitor saved to charge.mat,
            total-charge monitor over the core for C = dQ/dV.
  FDE     : same structure; silicon as an "Index perturbation" material fed by the np-density
            grid attribute imported from charge.mat; TE0 found at every bias index ->
            n_eff(V), loss(V). One extra solve with plain silicon -> loss of the undoped
            structure (metal + substrate leakage).
  python  : first-order perturbation integrals of the CHARGE carriers against |E|^2 of the
            0 V mode give dneff_core / dneff_slab and alpha_fca_p / alpha_fca_n; closure compares
            them with the full FDE solve. R x L from the carrier densities along the slab path.

Everything numerical (meshes, bias list, spans, materials) comes from SimConfig.active.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

from ..config import SimConfig
from ..families.pn_phase_shifter import (
    ALPHA_BACKGROUND_DB_CM,
    Q_E,
    PNResult,
    mobility_cm2,
    parse_xsection,
    soref_dalpha_cm,
    soref_dn,
)
from ..geometry import Geometry
from .base import Backend, register

UM = 1e-6
NM = 1e-9


def _import_lumapi():
    root = os.environ.get("LUMERICAL_ROOT", r"D:/Program Files/ANSYS Inc/v251")
    api = os.path.join(root, "api", "python")
    if api not in sys.path:
        sys.path.append(api)
    import lumapi  # type: ignore

    return lumapi


def _try(f, *calls):
    """Run the first of several (method, args) that does not raise; return its result or None."""
    for fn, args in calls:
        try:
            return getattr(f, fn)(*args)
        except Exception:  # noqa: BLE001 - property names differ between versions
            continue
    return None


@register("lumerical_charge_fde")
class LumericalChargeFDE(Backend):
    def __init__(self, hide: bool = True, **_):
        self.hide = hide
        self._charge = None
        self._mode = None

    # -- sessions -----------------------------------------------------------------
    def charge(self):
        if self._charge is None:
            self._charge = _import_lumapi().DEVICE(hide=self.hide)
        return self._charge

    def mode(self):
        if self._mode is None:
            self._mode = _import_lumapi().MODE(hide=self.hide)
        return self._mode

    def close(self) -> None:
        for s in (self._charge, self._mode):
            try:
                if s is not None:
                    s.close()
            except Exception:  # noqa: BLE001
                pass
        self._charge = self._mode = None

    # -- main ---------------------------------------------------------------------
    def run(self, geom: Geometry, cfg: SimConfig, run_dir: Optional[Path] = None) -> PNResult:
        t0 = time.time()
        a = cfg.active
        run_dir = Path(run_dir) if run_dir is not None else Path.cwd() / "prism_charge_fde_tmp"
        run_dir.mkdir(parents=True, exist_ok=True)
        if any(ord(c) > 127 for c in str(run_dir.resolve())):
            raise ValueError(f"Lumerical cannot handle non-ASCII paths: {run_dir}")
        charge_mat = (run_dir / "charge.mat").resolve()
        xs = parse_xsection(geom, cfg)

        # ---------------- CHARGE ----------------
        V_sweep, Cq, carriers = self._run_charge(geom, cfg, xs, charge_mat, run_dir)

        # ---------------- FDE ----------------
        idx = [int(np.argmin(np.abs(V_sweep - v))) for v in a.v_list]
        V = V_sweep[idx]
        neff, loss_db_m, field0, loss_undoped = self._run_fde(geom, cfg, xs, charge_mat, idx, run_dir)

        # ---------------- perturbation terms (python) ----------------
        pert = self._perturbation_terms(xs, cfg, carriers, V_sweep, idx, field0, float(np.real(neff[0])))

        dneff_total = np.real(neff) - np.real(neff[0])
        alpha_total = np.asarray(loss_db_m) / 100.0 + ALPHA_BACKGROUND_DB_CM  # dB/m -> dB/cm, plus scattering
        C_fF_um = np.interp(V, V_sweep[::-1], Cq[::-1]) * 1e15 / 1e6 if len(Cq) == len(V_sweep) else np.full_like(V, np.nan)
        R = self._series_resistance(xs, carriers, V_sweep, idx[int(np.argmin(np.abs(V - a.v_ref)))])

        res = PNResult(
            voltages=V,
            dneff_total=dneff_total,
            alpha_total=alpha_total,
            dneff_core=pert["dneff_core"],
            dneff_slab=pert["dneff_slab"],
            alpha_fca_p=pert["alpha_p"],
            alpha_fca_n=pert["alpha_n"],
            alpha_contacts=np.full_like(V, loss_undoped / 100.0),
            alpha_background=np.full_like(V, ALPHA_BACKGROUND_DB_CM),
            C_fF_um=C_fF_um,
            R_ohm_m=R,
            W_dep_um=pert["W_dep"],
            converged=True,
            backend="lumerical CHARGE + MODE FDE",
            runtime_s=time.time() - t0,
            extra={"charge_mat": str(charge_mat), "ldev": str(run_dir / "charge.ldev"), "lms": str(run_dir / "fde.lms"), "neff0": float(np.real(neff[0]))},
        )
        return res

    # -- CHARGE -------------------------------------------------------------------
    def _run_charge(self, geom, cfg, xs, charge_mat: Path, run_dir: Path):
        a = cfg.active
        d = self.charge()
        d.switchtolayout()
        d.deleteall()

        # materials (CT)
        d.addmodelmaterial()
        d.set("name", "Si")
        d.addmaterialproperties("CT", "Si (Silicon)")
        for prop, val in [
            ("electronic.gamma.mun.high field.active model", "monotonic"),
            ("electronic.gamma.mup.high field.active model", "monotonic"),
            ("electronic.gamma.mun.high field.driving field", "grad Phi"),
            ("electronic.gamma.mup.high field.driving field", "grad Phi"),
            ("electronic.gamma.mun.low field.active model", "masetti"),
            ("electronic.gamma.mup.low field.active model", "masetti"),
            ("recombination.band gap narrowing.active model", "slotboom"),
            ("recombination.trap assisted.enabled", 1),
            ("recombination.auger.enabled", 1),
        ]:
            _try(d, ("setnamed", ("Si", prop, val)), ("setnamed", ("Si", prop.replace(".gamma.", ".l."), val)))
        d.addmodelmaterial()
        d.set("name", "SiO2")
        d.addmaterialproperties("CT", "SiO2 (Glass) - Sze")
        d.addmodelmaterial()
        d.set("name", "Al")
        _try(d, ("addmaterialproperties", ("CT", "Al (Aluminium) - CRC")), ("addmaterialproperties", ("CT", "Al (Aluminium)")))

        X = max(abs(xs.si_x0), abs(xs.si_x1))
        h = a.core_height
        self._rect(d, "BOX", "SiO2", -X - 1, X + 1, -1.0, 0.0, 4)
        self._rect(d, "clad", "SiO2", -X - 1, X + 1, 0.0, h + 1.0, 4)
        for i, p in enumerate(geom.by_layer("SI")):
            x0, z0, x1, z1 = p.bbox()
            self._rect(d, f"si_{i}", "Si", x0, x1, z0, z1, 2)
        metal_names = {}
        for i, p in enumerate(geom.by_domain("electrode")):
            x0, z0, x1, z1 = p.bbox()
            name = f"metal_{p.props.get('terminal', i)}"
            self._rect(d, name, "Al", x0, x1, z0, z1, 1)
            metal_names[str(p.props.get("terminal", ""))] = name

        # doping boxes (additive, like the design contract)
        for i, p in enumerate(geom.by_domain("doping")):
            x0, z0, x1, z1 = p.bbox()
            typ = str(p.props.get("dopant", "p" if p.layer.upper().startswith("P") else "n")).lower()[0]
            d.adddope()
            d.set("name", f"dope_{p.layer}_{i}")
            d.set("dopant type", typ)
            d.set("concentration", float(p.props["doping"]) * 1e6)
            d.set("x", 0.5 * (x0 + x1) * UM)
            d.set("x span", (x1 - x0) * UM)
            d.set("y", 0.0)
            d.set("y span", 1 * UM)
            d.set("z", 0.5 * (z0 + z1) * UM)
            d.set("z span", (z1 - z0) * UM)

        # region, solver, mesh
        d.addsimulationregion()
        d.set("name", "sim_region")
        _try(d, ("set", ("dimension", "2D Y-Normal")), ("set", ("dimension", 2)))
        d.set("x", 0.0)
        d.set("x span", (2 * X + 0.2) * UM)
        d.set("y", 0.0)
        d.set("z", 0.5 * (h + 1.0 - 1.0) * UM)
        d.set("z span", (h + 2.0) * UM)
        d.addchargesolver()
        d.set("simulation region", "sim_region")
        d.set("solver mode", "steady state")
        d.set("norm length", a.norm_length_um * UM)
        d.set("min edge length", 0.5 * NM)
        d.set("max edge length", 40 * NM)
        _try(d, ("set", ("temperature dependence", "isothermal")))
        _try(d, ("set", ("simulation temperature", 300)))
        xj = xs.junction_x() or 0.0
        self._mesh_box(d, "mesh_junction", 2 * NM, xj - 0.08, xj + 0.08, 0.0, h)
        self._mesh_box(d, "mesh_core", a.charge_mesh_edge_um * UM / 2, xs.rib_x0 - 0.3, xs.rib_x1 + 0.3, 0.0, h)
        self._mesh_box(d, "mesh_all", a.charge_mesh_edge_um * UM, -X, X, 0.0, h)

        # contacts: anode swept, cathode grounded
        v_min = float(min(cfg.active.v_list))
        n_pts = int(round(abs(v_min) / 0.25)) + 1
        d.addelectricalcontact()
        d.set("name", "anode")
        d.set("bc mode", "steady state")
        d.set("sweep type", "range")
        d.set("range start", 0.0)
        d.set("range stop", v_min)
        d.set("range num points", n_pts)
        d.set("surface type", "solid")
        d.set("solid", metal_names.get("anode", "metal_anode"))
        d.addelectricalcontact()
        d.set("name", "cathode")
        d.set("bc mode", "steady state")
        d.set("sweep type", "single")
        d.set("voltage", 0.0)
        d.set("surface type", "solid")
        d.set("solid", metal_names.get("cathode", "metal_cathode"))

        # monitors
        d.addchargemonitor()
        d.set("name", "charge_full")
        d.set("monitor type", 6)  # 2D y-normal
        d.set("x", 0.0)
        d.set("x span", (2 * X + 0.02) * UM)
        d.set("y", 0.0)
        d.set("z", h / 2 * UM)
        d.set("z span", (h + 0.02) * UM)
        d.set("save data", 1)
        d.set("filename", str(charge_mat))
        d.addchargemonitor()
        d.set("name", "charge_core")
        d.set("monitor type", 6)
        d.set("x", 0.0)
        d.set("x span", (xs.rib_w + 1.0) * UM)
        d.set("y", 0.0)
        d.set("z", h / 2 * UM)
        d.set("z span", h * UM)
        _try(d, ("set", ("integrate total charge", 1)))

        d.save(str((run_dir / "charge.ldev").resolve()))
        d.run()

        iv = d.getresult("CHARGE", "anode")
        V_sweep = np.asarray(iv["V_anode"]).ravel()
        # C = dQ/dV from the integrated charge over the core
        Cq = np.array([])
        try:
            tq = d.getresult("CHARGE::charge_core", "total_charge")
            Vq = np.asarray(tq["V_anode"]).ravel()
            Nn = np.asarray(tq["n"]).ravel()
            Np = np.asarray(tq["p"]).ravel()
            Cq = 0.5 * (np.abs(np.gradient(Q_E * Nn, Vq)) + np.abs(np.gradient(Q_E * Np, Vq)))  # F/m
        except Exception:  # noqa: BLE001
            pass
        # carrier densities on the FE mesh for the perturbation integrals
        carriers = None
        try:
            ch = d.getresult("CHARGE::charge_full", "charge")
            carriers = {
                "x": np.asarray(ch["x"]).ravel() / UM,
                "z": np.asarray(ch["z"]).ravel() / UM,
                "n": np.asarray(ch["n"]),  # (nvertex, nV) m^-3
                "p": np.asarray(ch["p"]),
                "V": np.asarray(ch["V_anode"]).ravel(),
            }
        except Exception:  # noqa: BLE001
            carriers = None
        return V_sweep, Cq, carriers

    @staticmethod
    def _rect(d, name, mat, x0, x1, z0, z1, mesh_order):
        d.addrect()
        d.set("name", name)
        d.set("material", mat)
        d.set("x", 0.5 * (x0 + x1) * UM)
        d.set("x span", (x1 - x0) * UM)
        d.set("y", 0.0)
        d.set("y span", 1 * UM)
        d.set("z", 0.5 * (z0 + z1) * UM)
        d.set("z span", (z1 - z0) * UM)
        _try(d, ("set", ("override mesh order from material database", 1)))
        _try(d, ("set", ("mesh order", mesh_order)))

    @staticmethod
    def _mesh_box(d, name, edge, x0, x1, z0, z1):
        d.addchargemesh()
        d.set("name", name)
        d.set("max edge length", edge)
        d.set("x", 0.5 * (x0 + x1) * UM)
        d.set("x span", (x1 - x0) * UM)
        d.set("y", 0.0)
        d.set("y span", 1 * UM)
        d.set("z", 0.5 * (z0 + z1) * UM)
        d.set("z span", (z1 - z0) * UM)

    # -- FDE ----------------------------------------------------------------------
    def _run_fde(self, geom, cfg, xs, charge_mat: Path, idx, run_dir: Path):
        a = cfg.active
        m = self.mode()
        m.switchtolayout()
        m.deleteall()
        mat_ox, mat_si, mat_np = "SiO2 (Glass) - Palik", "Si (Silicon) - Palik", "Si_np"
        tmp = m.addmaterial("Index perturbation")
        m.setmaterial(tmp, "name", mat_np)
        _try(m, ("setmaterial", (mat_np, "base material", mat_si)))
        _try(m, ("setmaterial", (mat_np, "np density model", "Soref and Bennet")), ("setmaterial", (mat_np, "np density model", 1)))
        m.addgridattribute("np density")
        m.set("name", "np_density")
        m.importdataset(str(charge_mat))

        X = a.fde_span_x / 2 + 3.0
        h = a.core_height
        def rect(name, mat, x0, x1, z0, z1, mo, attr=None):
            m.addrect()
            m.set("name", name)
            m.set("material", mat)
            m.set("x", 0.5 * (x0 + x1) * UM)
            m.set("x span", (x1 - x0) * UM)
            m.set("y", 0.0)
            m.set("y span", 2 * UM)
            m.set("z", 0.5 * (z0 + z1) * UM)
            m.set("z span", (z1 - z0) * UM)
            _try(m, ("set", ("override mesh order from material database", 1)))
            _try(m, ("set", ("mesh order", mo)))
            if attr:
                _try(m, ("set", ("grid attribute name", attr)))

        rect("BOX", mat_ox, -X, X, -2.0, 0.0, 3)
        rect("clad", mat_ox, -X, X, 0.0, 2.0, 4)
        si_names = []
        for i, p in enumerate(geom.by_layer("SI")):
            x0, z0, x1, z1 = p.bbox()
            # extend outer silicon through the PML so no facet is seen
            x0e = -X if abs(x0 - xs.si_x0) < 1e-6 else x0
            x1e = X if abs(x1 - xs.si_x1) < 1e-6 else x1
            rect(f"si_{i}", mat_np, x0e, x1e, z0, z1, 2, "np_density")
            si_names.append(f"si_{i}")
        for i, p in enumerate(geom.by_domain("electrode")):
            x0, z0, x1, z1 = p.bbox()
            rect(f"metal_{i}", "Al (Aluminium) - Palik", x0, x1, z0, z1, 1)

        m.addfde()
        _try(m, ("set", ("solver type", "2D Y normal")))
        m.set("x", 0.0)
        m.set("x span", a.fde_span_x * UM)
        m.set("y", 0.0)
        m.set("z", (h / 2) * UM)
        m.set("z span", a.fde_span_z * UM)
        m.set("define x mesh by", "maximum mesh step")
        m.set("dx", a.fde_mesh_dx * UM)
        m.set("define z mesh by", "maximum mesh step")
        m.set("dz", a.fde_mesh_dz * UM)
        for bc in ("x min bc", "x max bc", "z min bc", "z max bc"):
            _try(m, ("set", (bc, "PML")))
        m.setanalysis("wavelength", a.wavelength * UM)
        m.setanalysis("number of trial modes", 4)
        m.setanalysis("search", "near n")
        m.setanalysis("use max index", 0)
        m.setanalysis("n", 2.5)
        m.save(str((run_dir / "fde.lms").resolve()))

        def solve_te0():
            n = int(m.findmodes())
            if n < 1:
                raise RuntimeError("FDE found no modes")
            best, best_score = 1, -1.0
            for k in range(1, n + 1):
                te = float(np.real(m.getdata(f"FDE::data::mode{k}", "TE polarization fraction")))
                ne = float(np.real(m.getdata(f"FDE::data::mode{k}", "neff")))
                score = te + 0.1 * ne
                if te > 0.6 and score > best_score:
                    best, best_score = k, score
            tag = f"FDE::data::mode{best}"
            return complex(m.getdata(tag, "neff")), float(np.real(m.getdata(tag, "loss"))), tag

        # undoped reference (plain silicon): metal + leakage loss
        for nm in si_names:
            m.setnamed(nm, "material", mat_si)
        _, loss_undoped, _ = solve_te0()
        m.switchtolayout()
        for nm in si_names:
            m.setnamed(nm, "material", mat_np)
            _try(m, ("setnamed", (nm, "grid attribute name", "np_density")))

        neff, loss, field0 = [], [], None
        for j, i_bias in enumerate(idx):
            m.switchtolayout()
            m.setnamed("np_density", "V_anode_index", int(i_bias) + 1)  # 1-based
            ne, lo, tag = solve_te0()
            neff.append(ne)
            loss.append(lo)
            if j == 0:
                x = np.asarray(m.getdata(tag, "x")).ravel() / UM
                z = np.asarray(m.getdata(tag, "z")).ravel() / UM
                E2 = sum(np.abs(np.squeeze(m.getdata(tag, c))) ** 2 for c in ("Ex", "Ey", "Ez"))
                field0 = (x, z, np.asarray(E2))
        return np.asarray(neff), np.asarray(loss), field0, loss_undoped

    # -- perturbation integrals ------------------------------------------------------------
    def _perturbation_terms(self, xs, cfg, carriers, V_sweep, idx, field0, neff0) -> Dict[str, np.ndarray]:
        nV = len(idx)
        out = {k: np.full(nV, np.nan) for k in ("dneff_core", "dneff_slab", "alpha_p", "alpha_n", "W_dep")}
        if carriers is None or field0 is None:
            return out
        from matplotlib.tri import LinearTriInterpolator, Triangulation

        x, z, E2 = field0
        Xg, Zg = np.meshgrid(x, z, indexing="ij")
        if E2.shape != Xg.shape:
            E2 = E2.T if E2.T.shape == Xg.shape else np.resize(E2, Xg.shape)
        tri = Triangulation(carriers["x"], carriers["z"])
        h_at = np.interp(Xg, xs.x, xs.h)
        in_si = (Zg >= 0) & (Zg <= h_at)
        norm = _trapz(_trapz(E2, z, axis=1), x)
        w = (3.476 / neff0) * E2 / max(norm, 1e-30)
        in_rib = (Xg >= xs.rib_x0) & (Xg <= xs.rib_x1)

        def grid(field_col):
            f = LinearTriInterpolator(tri, field_col)
            g = np.asarray(f(Xg, Zg))
            return np.where(np.isfinite(g), g, 0.0) * 1e-6  # m^-3 -> cm^-3

        n0, p0 = grid(carriers["n"][:, idx[0]]), grid(carriers["p"][:, idx[0]])
        for j, i_bias in enumerate(idx):
            n, p = grid(carriers["n"][:, i_bias]), grid(carriers["p"][:, i_bias])
            dn = -soref_dn(np.clip(n0 - n, 0, None), np.clip(p0 - p, 0, None)) * w * in_si
            out["dneff_core"][j] = _trapz(_trapz(dn * in_rib, z, axis=1), x)
            out["dneff_slab"][j] = _trapz(_trapz(dn * ~in_rib, z, axis=1), x)
            out["alpha_p"][j] = 4.343 * _trapz(_trapz(soref_dalpha_cm(0 * n, p) * w * in_si, z, axis=1), x)
            out["alpha_n"][j] = 4.343 * _trapz(_trapz(soref_dalpha_cm(n, 0 * p) * w * in_si, z, axis=1), x)
            # depletion width: extent along the slab mid-height where both n and p < 1e15
            zi = int(np.argmin(np.abs(z - xs.slab_h / 2)))
            dep = (n[:, zi] < 1e15) & (p[:, zi] < 1e15) & (Xg[:, zi] >= xs.rib_x0 - 0.5) & (Xg[:, zi] <= xs.rib_x1 + 0.5)
            out["W_dep"][j] = float(dep.sum() * (x[1] - x[0])) if len(x) > 1 else np.nan
        return out

    def _series_resistance(self, xs, carriers, V_sweep, i_bias) -> float:
        """R x L (ohm m) from the CHARGE carrier densities along the slab mid-height."""
        if carriers is None:
            return float("nan")
        from matplotlib.tri import LinearTriInterpolator, Triangulation

        tri = Triangulation(carriers["x"], carriers["z"])
        xg = xs.x
        zc = np.full_like(xg, max(xs.slab_h / 2, 0.02))
        n = np.asarray(LinearTriInterpolator(tri, carriers["n"][:, i_bias])(xg, zc)) * 1e-6
        p = np.asarray(LinearTriInterpolator(tri, carriers["p"][:, i_bias])(xg, zc)) * 1e-6
        n, p = np.nan_to_num(n), np.nan_to_num(p)
        sigma = Q_E * (mobility_cm2(n, "n") * n + mobility_cm2(p, "p") * p) * 1e2  # S/m
        lo = xs.contacts.get("anode", (xs.si_x0, xs.si_x0))[1]
        hi = xs.contacts.get("cathode", (xs.si_x1, xs.si_x1))[0]
        sel = (xg >= lo) & (xg <= hi) & (sigma > 1e-3)
        dx = float(xg[1] - xg[0]) * 1e-6
        R = float(np.sum(dx / (sigma[sel] * np.clip(xs.h[sel] * 1e-6, 1e-9, None))))
        return R + 2 * 1e-11 / 0.5e-6


@register("lumerical_ring")
class LumericalRing(LumericalChargeFDE):
    """pn_ring_modulator backend: the CHARGE + FDE chain above plus one bent-waveguide FDE solve of
    the undoped structure at the ring radius (alpha_bend = bent loss - straight loss). The bus
    coupler is still the toy CMT model (`coupling_model = "toy"`); an FDTD coupler run is a
    planned addition. UNTESTED on a real installation, like the parent class."""

    def run(self, geom: Geometry, cfg: SimConfig, run_dir: Optional[Path] = None):
        from ..families.pn_ring_modulator import RingResult, ring_params, toy_coupling

        pn = super().run(geom, cfg, run_dir)
        p = ring_params(geom, cfg)
        alpha_bend = float("nan")
        try:
            m = self.mode()
            m.switchtolayout()
            # plain silicon everywhere, bent solve at the ring radius
            mat_si = "Si (Silicon) - Palik"
            for i in range(len(geom.by_layer("SI"))):
                m.setnamed(f"si_{i}", "material", mat_si)
            m.setanalysis("bent waveguide", 1)
            m.setanalysis("bend radius", p["radius_um"] * UM)
            _try(m, ("setanalysis", ("bend location", "simulation center")))
            n = int(m.findmodes())
            best, best_te = 1, -1.0
            for k in range(1, n + 1):
                te = float(np.real(m.getdata(f"FDE::data::mode{k}", "TE polarization fraction")))
                if te > best_te:
                    best, best_te = k, te
            loss_bent = float(np.real(m.getdata(f"FDE::data::mode{best}", "loss"))) / 100.0  # dB/cm
            alpha_bend = max(loss_bent - float(pn.alpha_contacts[0]), 0.0)
            m.switchtolayout()
            m.setanalysis("bent waveguide", 0)
        except Exception as e:  # noqa: BLE001
            pn.extra["bend_solve_error"] = str(e)
        res = RingResult(
            **{k: getattr(pn, k) for k in pn.__dataclass_fields__},
            alpha_bend_db_cm=alpha_bend,
            kappa2=toy_coupling(p["gap_um"], p["radius_um"], cfg),
            radius_um=p["radius_um"],
            gap_um=p["gap_um"],
            doped_fraction=p["doped_fraction"],
            coupling_model="toy (CMT) - FDTD coupler run not implemented",
        )
        res.backend = "lumerical CHARGE + MODE FDE (+ bent FDE)"
        return res
