"""Active family: lateral PN-junction phase shifter, 2D x-z cross-section (CHARGE + FDE).

Geometry convention (plane = "xz", micrometres, z = 0 at the BOX top):
    layer SI              rib and slab rectangles (axis-aligned); core_height / slab_height from config
    layers P, N, P+, N+, P++, N++   implant windows, props {"doping": cm^-3, "dopant": "p"|"n"};
                          windows ADD (net = sum donors - sum acceptors), as CHARGE doping objects do
    layer METAL           contacts on top of the silicon, props {"terminal": "anode"|"cathode"}

What one simulation returns (PNResult), per bias V in cfg.active.v_list:
    dneff_total(V)    change of n_eff relative to 0 V from the full mode solve with carriers
    alpha_total(V)    propagation loss (dB/cm) from the full mode solve
    dneff_core/slab   first-order perturbation split of dneff: depletion inside the rib vs in the slab
    alpha_fca_p/n     free-carrier absorption from holes / electrons (Soref), alpha_contacts (metal +
                      heavily doped tails), alpha_background (sidewall scattering, fixed)
    C_fF_um(V)        junction capacitance per length, R_ohm_m  series resistance x length
Closure: sum of the perturbation terms must reproduce the full-solve dneff and alpha within
closure_tol. In the Lumerical backend both sides come from different computations (perturbation
integral over CHARGE carriers vs. FDE with imported carriers); in the mock they are the same
toy model plus noise, so the closure test only exercises the machinery.

FOM = VpiL x alpha at v_ref  (V dB), lower is better.  VpiL = lambda * |V| / (2 dneff(V)).
Constraints: alpha(v_ref) <= max_alpha_db_cm, VpiL <= max_vpil_vcm, f_RC = 1/(2 pi R C) >= min_f_rc_ghz.

The toy physics below is a 1-D abrupt-junction depletion model with Soref coefficients and an
analytic mode profile. It reproduces the standard trade-offs (doping up: efficiency up, loss up,
R down; junction offset: overlap; contacts closer: R down, loss up) but is not a substitute for
CHARGE + FDE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

from ..checks import Violation
from ..geometry import Geometry, Polygon
from . import Family, register

# physical constants / silicon numbers
Q_E = 1.602176634e-19
EPS0 = 8.8541878128e-12
EPS_SI = 11.7 * EPS0
KT_Q = 0.02585  # V at 300 K
NI_CM3 = 1.0e10
GAMMA_SI = 0.80  # fraction of modal power in silicon (TE0, 500 x 220 nm rib)
TAIL_LENGTH_UM = 0.15  # intensity decay length of the mode tail in the slab (toy)
ALPHA_BACKGROUND_DB_CM = 1.5  # sidewall scattering etc., bias independent
DOPING_LAYERS = ("P", "N", "P+", "N+", "P++", "N++", "PP", "NN", "PPP", "NNN")


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------


@dataclass
class PNResult:
    voltages: np.ndarray
    dneff_total: np.ndarray
    alpha_total: np.ndarray  # dB/cm
    dneff_core: np.ndarray
    dneff_slab: np.ndarray
    alpha_fca_p: np.ndarray
    alpha_fca_n: np.ndarray
    alpha_contacts: np.ndarray
    alpha_background: np.ndarray
    C_fF_um: np.ndarray
    R_ohm_m: float  # series resistance x device length (ohm m); R = R_ohm_m / L
    W_dep_um: np.ndarray
    converged: bool = True
    backend: str = ""
    runtime_s: float = 0.0
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items() if k != "extra"}
        d["extra"] = {k: (v if isinstance(v, (str, int, float, bool, list, dict)) else str(v)) for k, v in self.extra.items()}
        return d

    @staticmethod
    def from_dict(d: dict) -> "PNResult":
        arr = lambda k: np.asarray(d[k], float)  # noqa: E731
        return PNResult(
            arr("voltages"), arr("dneff_total"), arr("alpha_total"), arr("dneff_core"), arr("dneff_slab"),
            arr("alpha_fca_p"), arr("alpha_fca_n"), arr("alpha_contacts"), arr("alpha_background"),
            arr("C_fF_um"), float(d["R_ohm_m"]), arr("W_dep_um"), d.get("converged", True), d.get("backend", ""),
            d.get("runtime_s", 0.0), d.get("extra", {}),
        )


# ----------------------------------------------------------------------------
# Cross-section parsing
# ----------------------------------------------------------------------------


def _is_rect(p: Polygon) -> bool:
    pts = p.points
    if len(pts) != 4:
        return False
    xs, zs = sorted(set(np.round(pts[:, 0], 6))), sorted(set(np.round(pts[:, 1], 6)))
    return len(xs) == 2 and len(zs) == 2


def _rect(p: Polygon) -> Tuple[float, float, float, float]:
    return p.bbox()  # (x0, z0, x1, z1)


@dataclass
class XSection:
    rib_x0: float
    rib_x1: float
    rib_h: float
    slab_h: float
    si_x0: float
    si_x1: float
    x: np.ndarray  # sample grid along x (um)
    NA: np.ndarray  # acceptor concentration along x at slab level (cm^-3)
    ND: np.ndarray
    h: np.ndarray  # silicon height along x (um)
    contacts: Dict[str, Tuple[float, float]]  # terminal -> (x0, x1)
    z: np.ndarray = field(default_factory=lambda: np.zeros(0))  # row grid (um)
    NA2: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # (nx, nz) acceptors
    ND2: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    si2: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), bool))  # (nx, nz) silicon present

    @property
    def rib_w(self) -> float:
        return self.rib_x1 - self.rib_x0

    @property
    def net(self) -> np.ndarray:
        return self.ND - self.NA

    def junction_x(self) -> Optional[float]:
        """Metallurgical junction nearest the rib centre (sign change of net doping)."""
        s = np.sign(self.net)
        idx = np.nonzero(np.diff(s) != 0)[0]
        if len(idx) == 0:
            return None
        xc = 0.5 * (self.rib_x0 + self.rib_x1)
        xs = 0.5 * (self.x[idx] + self.x[idx + 1])
        return float(xs[np.argmin(np.abs(xs - xc))])


def parse_xsection(geom: Geometry, cfg, dx: float = 0.002) -> XSection:
    a = cfg.active
    si = geom.by_layer("SI")
    if not si:
        raise ValueError("no SI polygons")
    rects = [_rect(p) for p in si]
    si_x0 = min(r[0] for r in rects)
    si_x1 = max(r[2] for r in rects)
    x = np.arange(si_x0, si_x1 + dx / 2, dx)
    h = np.zeros_like(x)
    for x0, z0, x1, z1 in rects:
        sel = (x >= x0 - 1e-9) & (x <= x1 + 1e-9)
        h[sel] = np.maximum(h[sel], z1)
    rib = h >= a.core_height - 1e-6
    if not rib.any():
        raise ValueError("no SI polygon reaches core_height")
    ridx = np.nonzero(rib)[0]
    rib_x0, rib_x1 = float(x[ridx[0]]), float(x[ridx[-1]])
    slab = h[(h > 1e-6) & (h < a.core_height - 1e-6)]
    slab_h = float(np.median(slab)) if len(slab) else 0.0

    NA = np.zeros_like(x)
    ND = np.zeros_like(x)
    z_probe = max(slab_h / 2, 0.01)
    for p in geom.polygons:
        if p.layer.upper() not in DOPING_LAYERS and p.domain != "doping":
            continue
        x0, z0, x1, z1 = _rect(p)
        if not (z0 - 1e-9 <= z_probe <= z1 + 1e-9):
            continue
        conc = float(p.props.get("doping", 0.0))
        typ = str(p.props.get("dopant", "p" if p.layer.upper().startswith("P") else "n")).lower()
        sel = (x >= x0 - 1e-9) & (x <= x1 + 1e-9)
        if typ.startswith("p"):
            NA[sel] += conc
        else:
            ND[sel] += conc

    contacts = {}
    for p in geom.polygons:
        if p.domain == "electrode":
            x0, z0, x1, z1 = _rect(p)
            term = str(p.props.get("terminal", "")).lower()
            if term:
                contacts[term] = (x0, x1)

    # 2-D doping maps on a coarser grid (rows of dz) for the row-wise junction model
    dz = 0.01
    z = np.arange(dz / 2, a.core_height, dz)
    xc = np.arange(si_x0 + dx, si_x1, 4 * dx)  # 8 nm columns
    si2 = np.zeros((len(xc), len(z)), bool)
    for x0, z0, x1, z1 in rects:
        si2[np.ix_((xc >= x0) & (xc <= x1), (z >= z0) & (z <= z1))] = True
    NA2 = np.zeros((len(xc), len(z)))
    ND2 = np.zeros((len(xc), len(z)))
    for p in geom.polygons:
        if p.layer.upper() not in DOPING_LAYERS and p.domain != "doping":
            continue
        x0, z0, x1, z1 = _rect(p)
        conc = float(p.props.get("doping", 0.0))
        typ = str(p.props.get("dopant", "p" if p.layer.upper().startswith("P") else "n")).lower()
        blk = np.ix_((xc >= x0 - 1e-9) & (xc <= x1 + 1e-9), (z >= z0 - 1e-9) & (z <= z1 + 1e-9))
        if typ.startswith("p"):
            NA2[blk] += conc
        else:
            ND2[blk] += conc
    xs = XSection(rib_x0, rib_x1, a.core_height, slab_h, si_x0, si_x1, x, NA, ND, h, contacts, z, NA2, ND2, si2)
    xs.xc = xc  # type: ignore[attr-defined]
    return xs


# ----------------------------------------------------------------------------
# Toy physics
# ----------------------------------------------------------------------------


def soref_dn(dNe: np.ndarray, dNh: np.ndarray) -> np.ndarray:
    """Index change for ADDED carriers (cm^-3) at 1550 nm (Soref & Bennett 1987)."""
    return -(8.8e-22 * dNe + 8.5e-18 * np.power(np.clip(dNh, 0, None), 0.8))


def soref_dalpha_cm(Ne: np.ndarray, Nh: np.ndarray) -> np.ndarray:
    """Absorption (cm^-1) from carriers (cm^-3) at 1550 nm."""
    return 8.5e-18 * Ne + 6.0e-18 * Nh


def mobility_cm2(N: np.ndarray, carrier: str) -> np.ndarray:
    N = np.clip(N, 1e14, None)
    if carrier == "p":
        return 50 + 420 / (1 + (N / 1.6e17) ** 0.7)
    return 70 + 1360 / (1 + (N / 9.0e16) ** 0.7)


def mode_weight(xs: XSection) -> np.ndarray:
    """Normalised lateral intensity profile of the TE0 mode along x (integrates to 1 over um).

    Gaussian inside the rib, exponential evanescent tail in the slab, slab part scaled by the
    thickness ratio to mimic the weaker vertical confinement there.
    """
    x = xs.x
    xc = 0.5 * (xs.rib_x0 + xs.rib_x1)
    half = xs.rib_w / 2
    wx = half + 0.15
    I = np.exp(-2 * ((x - xc) / wx) ** 2)
    out = np.abs(x - xc) > half
    edge = np.exp(-2 * (half / wx) ** 2)
    I[out] = edge * np.exp(-(np.abs(x[out] - xc) - half) / TAIL_LENGTH_UM) * (max(xs.slab_h, 1e-3) / xs.rib_h)
    I[xs.h <= 1e-6] = 0.0
    return I / max(_trapz(I, x), 1e-12)


def depletion(xs: XSection, V: float) -> Tuple[float, float, float, float, float]:
    """1-D abrupt junction: returns (x_p, x_n, W_um, Na, Nd) at anode bias V (reverse < 0)."""
    xj = xs.junction_x()
    if xj is None:
        return (np.nan, np.nan, 0.0, 0.0, 0.0)
    i = int(np.argmin(np.abs(xs.x - xj)))
    Na = float(xs.NA[max(i - 10, 0)])
    Nd = float(xs.ND[min(i + 10, len(xs.x) - 1)])
    if Na <= 0 or Nd <= 0:
        return (xj, xj, 0.0, Na, Nd)
    vbi = KT_Q * np.log(Na * Nd / NI_CM3**2)
    vr = max(vbi - V, 0.05)  # anode negative = reverse bias
    W = np.sqrt(2 * EPS_SI * vr / Q_E * (Na + Nd) / (Na * Nd * 1e6)) * 1e6  # um; (Na+Nd)/(Na Nd) in m^3 from cm^-3
    xp = xj - W * Nd / (Na + Nd)
    xn = xj + W * Na / (Na + Nd)
    return (xp, xn, W, Na, Nd)


def _depletion_1d(Na: float, Nd: float, V: float) -> float:
    """Abrupt-junction depletion width (um) for anode bias V (reverse < 0)."""
    if Na <= 0 or Nd <= 0:
        return 0.0
    vbi = KT_Q * np.log(Na * Nd / NI_CM3**2)
    vr = max(vbi - V, 0.05)
    return float(np.sqrt(2 * EPS_SI * vr / Q_E * (Na + Nd) / (Na * Nd * 1e6)) * 1e6)


def mode_weight_2d(xs: XSection) -> np.ndarray:
    """Normalised TE0 intensity on the (xc, z) grid inside the silicon (integrates to 1 over um^2).

    Lateral: Gaussian in the rib, exponential tail outside. Vertical: Gaussian centred in the
    local silicon thickness; in the thin slab the field is mostly outside the silicon, so the
    in-silicon weight per unit height scales with (h_local / h_core).
    """
    xc, z = xs.xc, xs.z
    xm = 0.5 * (xs.rib_x0 + xs.rib_x1)
    half = xs.rib_w / 2
    wx = half + 0.15
    Ix = np.exp(-2 * ((xc - xm) / wx) ** 2)
    out = np.abs(xc - xm) > half
    Ix[out] = np.exp(-2 * (half / wx) ** 2) * np.exp(-(np.abs(xc[out] - xm) - half) / TAIL_LENGTH_UM)
    h_loc = np.interp(xc, xs.x, xs.h)
    I = np.zeros_like(xs.NA2)
    for i, hl in enumerate(h_loc):
        if hl <= 1e-6:
            continue
        Iz = np.exp(-2 * ((z - hl / 2) / (0.6 * hl)) ** 2) * (z <= hl + 1e-9)
        Iz *= (hl / xs.rib_h) / max(Iz.sum() * (z[1] - z[0]), 1e-12)  # in-silicon fraction per column
        I[i] = Ix[i] * Iz
    I[~xs.si2] = 0.0
    dA = (xc[1] - xc[0]) * (z[1] - z[0])
    return I / max(I.sum() * dA, 1e-30)


def _frac_inside(c: np.ndarray, dc: float, lo: float, hi: float) -> np.ndarray:
    """Fraction of each cell (centre c, width dc) lying inside [lo, hi] (sub-cell resolution)."""
    return np.clip((np.minimum(hi, c + dc / 2) - np.maximum(lo, c - dc / 2)) / dc, 0.0, 1.0)


def _carriers_2d(xs: XSection, V: float):
    """Row-wise (lateral) and column-wise (vertical) abrupt-junction depletion on the 2-D map.

    Handles lateral, offset, L-shaped and vertical junctions approximately: every row with a
    sign change of the net doping gets a lateral depletion zone, every column with a sign
    change along z a vertical one; the union is depleted. Depletion edges are resolved below
    the cell size with fractional occupancy. Returns (p, n, depleted_fraction, cap_sum) where
    cap_sum = sum(dz / W_row) + sum(dx / W_col) (-> C' = eps * cap_sum).
    """
    xc, z = xs.xc, xs.z
    dxc, dz = float(xc[1] - xc[0]), float(z[1] - z[0])
    net = xs.ND2 - xs.NA2
    p = np.where(net < 0, -net, 0.0) * xs.si2
    n = np.where(net > 0, net, 0.0) * xs.si2
    dep = np.zeros(xs.si2.shape, float)
    cap_sum = 0.0
    xm = 0.5 * (xs.rib_x0 + xs.rib_x1)
    # lateral junctions, row by row
    for k in range(len(z)):
        row = net[:, k] * xs.si2[:, k]
        sgn = np.sign(row)
        idx = np.nonzero((np.diff(sgn) != 0) & (sgn[:-1] != 0) & (sgn[1:] != 0))[0]
        for j in idx:
            Na = float(-row[max(j - 3, 0)]) if row[j] < 0 else float(-row[min(j + 4, len(row) - 1)])
            Nd = float(row[min(j + 4, len(row) - 1)]) if row[j] < 0 else float(row[max(j - 3, 0)])
            if Na <= 0 or Nd <= 0:
                continue
            W = _depletion_1d(Na, Nd, V)
            xj = 0.5 * (xc[j] + xc[j + 1])
            wp, wn = W * Nd / (Na + Nd), W * Na / (Na + Nd)
            lo, hi = (xj - wp, xj + wn) if row[j] < 0 else (xj - wn, xj + wp)
            dep[:, k] = np.maximum(dep[:, k], _frac_inside(xc, dxc, lo, hi))
            if abs(xj - xm) < 2.0:  # only junctions near the rib store the modulation charge
                cap_sum += dz / max(W, 1e-3)
    # vertical junctions, column by column
    for i in range(len(xc)):
        col = net[i, :] * xs.si2[i, :]
        sgn = np.sign(col)
        idx = np.nonzero((np.diff(sgn) != 0) & (sgn[:-1] != 0) & (sgn[1:] != 0))[0]
        for j in idx:
            Na = float(-min(col[j], col[j + 1]))
            Nd = float(max(col[j], col[j + 1]))
            if Na <= 0 or Nd <= 0:
                continue
            W = _depletion_1d(Na, Nd, V)
            zj = 0.5 * (z[j] + z[j + 1])
            wa = W * Nd / (Na + Nd)  # extent into the p side
            wd = W * Na / (Na + Nd)
            lo, hi = (zj - wa, zj + wd) if col[j] < 0 else (zj - wd, zj + wa)
            dep[i, :] = np.maximum(dep[i, :], _frac_inside(z, dz, lo, hi))
            if abs(xc[i] - xm) < xs.rib_w:
                cap_sum += dxc / max(W, 1e-3)
    dep *= xs.si2
    p = p * (1.0 - dep)
    n = n * (1.0 - dep)
    return p, n, dep, cap_sum


def toy_physics(geom: Geometry, cfg, noise: float = 0.0, rng: Optional[np.random.Generator] = None) -> PNResult:
    a = cfg.active
    xs = parse_xsection(geom, cfg)
    xc, z = xs.xc, xs.z
    dA = (xc[1] - xc[0]) * (z[1] - z[0])
    I = mode_weight_2d(xs)
    in_rib = ((xc >= xs.rib_x0) & (xc <= xs.rib_x1))[:, None]
    V = np.asarray(a.v_list, float)

    p0, n0, dep0, _ = _carriers_2d(xs, 0.0)
    dn_core, dn_slab, a_p, a_n, a_ct, Cs, Ws = [], [], [], [], [], [], []
    d_ct = _contact_distance(xs)
    for Vb in V:
        p, n, dep, cap_sum = _carriers_2d(xs, Vb)
        removed_h, removed_e = np.clip(p0 - p, 0, None), np.clip(n0 - n, 0, None)
        dn = -soref_dn(removed_e, removed_h) * I * GAMMA_SI  # positive: carriers removed
        dn_core.append(float((dn * in_rib).sum() * dA))
        dn_slab.append(float((dn * ~in_rib).sum() * dA))
        a_p.append(4.343 * GAMMA_SI * float((soref_dalpha_cm(0 * n, p) * I).sum() * dA))
        a_n.append(4.343 * GAMMA_SI * float((soref_dalpha_cm(n, 0 * p) * I).sum() * dA))
        a_ct.append(200.0 * np.exp(-d_ct / 0.12) if np.isfinite(d_ct) else 0.0)
        Cs.append(1.2 * EPS_SI * cap_sum * 1e9)  # F/m -> fF/um
        # effective depletion width: depleted silicon area near the rib / core height
        near = dep * (np.abs(xc - 0.5 * (xs.rib_x0 + xs.rib_x1)) < xs.rib_w)[:, None]
        Ws.append(float(near.sum() * dA / xs.rib_h))
    dn_core, dn_slab = np.array(dn_core), np.array(dn_slab)
    a_p, a_n, a_ct = np.array(a_p), np.array(a_n), np.array(a_ct)
    a_bg = np.full_like(V, ALPHA_BACKGROUND_DB_CM)

    # series resistance x length along the slab mid-height from the depletion edges to the contacts
    x, dx = xs.x, float(xs.x[1] - xs.x[0])
    xp, xn, _, _, _ = depletion(xs, a.v_ref)
    R = 0.0
    for side, (lo, hi) in (("p", (xs.contacts.get("anode", (xs.si_x0, xs.si_x0))[1], xp)), ("n", (xn, xs.contacts.get("cathode", (xs.si_x1, xs.si_x1))[0]))):
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            continue
        sel = (x >= min(lo, hi)) & (x <= max(lo, hi))
        N = (xs.NA if side == "p" else xs.ND)[sel]
        rho = 1.0 / (Q_E * mobility_cm2(N, side) * np.clip(N, 1e14, None)) * 1e-2  # ohm cm -> ohm m
        R += float(np.sum(rho / np.clip(xs.h[sel] * 1e-6, 1e-9, None)) * dx * 1e-6)
    R += 2 * 1e-11 / 0.5e-6  # contact resistance rho_c = 1e-7 ohm cm^2 over a 0.5 um wide contact, both sides

    total_dn = dn_core + dn_slab
    total_a = a_p + a_n + a_ct + a_bg
    if noise > 0 and rng is not None:
        total_dn = total_dn * rng.normal(1.0, noise, size=total_dn.shape)
        total_a = total_a * rng.normal(1.0, noise, size=total_a.shape)
    converged = True
    Wmin = float(np.min(Ws)) if len(Ws) else 0.0
    if Wmin < 2 * a.charge_mesh_edge_um:  # depletion region not resolved by the CHARGE mesh
        converged = False
        total_dn = total_dn * 0.9
    return PNResult(V, total_dn, total_a, dn_core, dn_slab, a_p, a_n, a_ct, a_bg, np.array(Cs), R, np.array(Ws), converged)


def _contact_distance(xs: XSection) -> float:
    d = np.inf
    for term, (x0, x1) in xs.contacts.items():
        if x1 <= xs.rib_x0:
            d = min(d, xs.rib_x0 - x1)
        elif x0 >= xs.rib_x1:
            d = min(d, x0 - xs.rib_x1)
        else:
            d = 0.0
    return d


# ----------------------------------------------------------------------------
# Breakdown
# ----------------------------------------------------------------------------


@dataclass
class PNBreakdown:
    res: PNResult
    cfg: object
    closure_err: float
    closure_ok: bool

    def _iref(self) -> int:
        return int(np.argmin(np.abs(self.res.voltages - self.cfg.active.v_ref)))

    def summary(self) -> dict:
        r, a, i = self.res, self.cfg.active, self._iref()
        dn = float(r.dneff_total[i])
        vpil = (a.wavelength * 1e-4) * abs(a.v_ref) / (2 * dn) if dn > 1e-9 else float("inf")  # V cm
        alpha = float(r.alpha_total[i])
        C = float(r.C_fF_um[i])
        f_rc = 1.0 / (2 * np.pi * r.R_ohm_m * C * 1e-15 / 1e-6) / 1e9 if (r.R_ohm_m > 0 and C > 0) else float("inf")
        tot_dn = max(float(r.dneff_core[i] + r.dneff_slab[i]), 1e-12)
        tot_a = max(float(r.alpha_fca_p[i] + r.alpha_fca_n[i] + r.alpha_contacts[i] + r.alpha_background[i]), 1e-12)
        return {
            "VpiL_Vcm": float(vpil),
            "alpha_dB_cm": alpha,
            "FOM_VdB": float(vpil * alpha),
            "f_rc_GHz": float(f_rc),
            "C_fF_um": C,
            "R_ohm_um": float(r.R_ohm_m * 1e6),
            "W_dep_um": float(r.W_dep_um[i]),
            "dneff": dn,
            "dneff_core_frac": float(r.dneff_core[i] / tot_dn),
            "dneff_slab_frac": float(r.dneff_slab[i] / tot_dn),
            "alpha_fca_p_dB_cm": float(r.alpha_fca_p[i]),
            "alpha_fca_n_dB_cm": float(r.alpha_fca_n[i]),
            "alpha_contacts_dB_cm": float(r.alpha_contacts[i]),
            "alpha_background_dB_cm": float(r.alpha_background[i]),
            "alpha_terms_sum_dB_cm": tot_a,
            "v_ref": float(a.v_ref),
            "closure_err": float(self.closure_err),
            "closure_ok": bool(self.closure_ok),
        }

    def table(self) -> str:
        return _table_rows(self.res, self.closure_err)

    def to_dict(self) -> dict:
        return {"family": "pn_phase_shifter", "result": self.res.to_dict(), "closure_err": float(self.closure_err), "closure_ok": bool(self.closure_ok), "summary": self.summary()}


def decompose_pn(res: PNResult, cfg) -> PNBreakdown:
    a = cfg.active
    i = int(np.argmin(np.abs(res.voltages - a.v_ref)))
    sum_dn = res.dneff_core[i] + res.dneff_slab[i]
    sum_a = res.alpha_fca_p[i] + res.alpha_fca_n[i] + res.alpha_contacts[i] + res.alpha_background[i]
    e_dn = abs(sum_dn - res.dneff_total[i]) / max(abs(res.dneff_total[i]), 1e-12)
    e_a = abs(sum_a - res.alpha_total[i]) / max(abs(res.alpha_total[i]), 1e-12)
    err = float(max(e_dn, e_a))
    return PNBreakdown(res, cfg, err, bool(err <= cfg.closure_tol and res.converged))


# ----------------------------------------------------------------------------
# Family
# ----------------------------------------------------------------------------


@register
class PNPhaseShifter(Family):
    name = "pn_phase_shifter"
    default_backends = ("lumerical_charge_fde", "mock_pn")
    fom_label = "VpiL x alpha at v_ref (V dB, lower is better)"
    fom_unit = "V dB"
    terms_help = (
        "Breakdown per bias V (anode, reverse < 0): dneff = n_eff(V) - n_eff(0) from the full mode solve, split by "
        "first-order perturbation into dneff_core (depletion inside the rib, where the mode is strong) and dneff_slab "
        "(depletion in the slab); alpha = loss in dB/cm split into fca_p (holes), fca_n (electrons), contacts "
        "(metal + heavily doped tails) and background (fixed scattering). C = junction capacitance per length, "
        "R x L = series resistance times length, f_RC = 1/(2 pi R C). VpiL = lambda |V| / (2 dneff(V)) at v_ref. "
        "Closure: the perturbation terms must reproduce the full-solve dneff and alpha within closure_tol."
    )

    # -- features -------------------------------------------------------------------
    def features(self, geom: Geometry, cfg) -> Dict[str, float]:
        xs = parse_xsection(geom, cfg)
        xj = xs.junction_x()
        xc = 0.5 * (xs.rib_x0 + xs.rib_x1)
        i = int(np.argmin(np.abs(xs.x - xj))) if xj is not None else None
        f = {
            "w_rib": xs.rib_w,
            "h_slab": xs.slab_h,
            "x_j_offset": (xj - xc) if xj is not None else float("nan"),
            "Na_core": float(xs.NA[max(i - 10, 0)]) if i is not None else 0.0,
            "Nd_core": float(xs.ND[min(i + 10, len(xs.x) - 1)]) if i is not None else 0.0,
        }
        for side, sign in (("p", -1), ("n", +1)):
            N = xs.NA if side == "p" else xs.ND
            edge = xs.rib_x0 if sign < 0 else xs.rib_x1
            outside = (xs.x < xs.rib_x0) if sign < 0 else (xs.x > xs.rib_x1)
            heavy = outside & (N >= 1e19)
            f[f"d_pp_{side}"] = float(np.min(np.abs(xs.x[heavy] - edge))) if heavy.any() else float("nan")
            mid = outside & (N > 0) & (N < 1e19)
            f[f"N_slab_{side}"] = float(np.max(N[mid])) if mid.any() else 0.0
            term = "anode" if side == "p" else "cathode"
            if term in xs.contacts:
                x0, x1 = xs.contacts[term]
                f[f"d_ct_{side}"] = float(abs((x1 if sign < 0 else x0) - edge))
            else:
                f[f"d_ct_{side}"] = float("nan")
        xp, xn, W, _, _ = depletion(xs, cfg.active.v_ref)
        f["W_dep_vref"] = W
        # junction shape: fraction of rib rows with a lateral sign change / rib columns with a vertical one
        net = (xs.ND2 - xs.NA2) * xs.si2
        rib_cols = (xs.xc >= xs.rib_x0) & (xs.xc <= xs.rib_x1)
        sub = net[rib_cols, :]
        lat = np.mean([(np.diff(np.sign(sub[:, k])) != 0).any() for k in range(sub.shape[1])]) if sub.size else 0.0
        ver = np.mean([(np.diff(np.sign(sub[i, :])) != 0).any() for i in range(sub.shape[0])]) if sub.size else 0.0
        f["lateral_junction_frac"] = float(lat)
        f["vertical_junction_frac"] = float(ver)
        return f

    # -- checks ---------------------------------------------------------------------
    def check(self, geom: Geometry, cfg) -> List[Violation]:
        a = cfg.active
        v: List[Violation] = []
        if geom.plane != "xz":
            return [Violation("plane", "pn_phase_shifter needs Geometry(plane='xz') cross-section polygons")]
        bad = [p.layer for p in geom.polygons if not _is_rect(p)]
        if bad:
            return [Violation("rectangles", f"all cross-section polygons must be axis-aligned rectangles; offending layers {bad[:5]}")]
        try:
            xs = parse_xsection(geom, cfg)
        except ValueError as e:
            return [Violation("silicon", str(e))]
        if abs(xs.slab_h - a.slab_height) > 0.005 and xs.slab_h > 0:
            v.append(Violation("stack", f"slab height {xs.slab_h:.3f} um != locked {a.slab_height} um"))
        if max(abs(xs.si_x0), abs(xs.si_x1)) > a.max_half_width:
            v.append(Violation("extent", f"cross-section half width {max(abs(xs.si_x0), abs(xs.si_x1)):.2f} um > {a.max_half_width} um"))
        if xs.rib_w < cfg.min_width:
            v.append(Violation("min_width", f"rib width {xs.rib_w:.3f} um < {cfg.min_width} um"))
        # doping windows inside the silicon
        for p in geom.by_domain("doping"):
            x0, z0, x1, z1 = _rect(p)
            if x0 < xs.si_x0 - 1e-6 or x1 > xs.si_x1 + 1e-6 or z1 > xs.rib_h + 1e-6 or z0 < -1e-6:
                v.append(Violation("doping_window", f"{p.layer} window [{x0:.2f},{x1:.2f}]x[{z0:.2f},{z1:.2f}] extends outside the silicon"))
            if float(p.props.get("doping", 0)) <= 0:
                v.append(Violation("doping_value", f"{p.layer} window has no positive 'doping' property (cm^-3)"))
        # junction inside the rib
        xj = xs.junction_x()
        if xj is None:
            v.append(Violation("junction", "no PN junction found (net doping never changes sign)"))
        elif not (xs.rib_x0 + 0.02 <= xj <= xs.rib_x1 - 0.02):
            v.append(Violation("junction", f"junction at x = {xj:.3f} um lies outside the rib [{xs.rib_x0:.3f},{xs.rib_x1:.3f}]", severity="warning"))
        # contacts
        for term, side in (("anode", "p"), ("cathode", "n")):
            if term not in xs.contacts:
                v.append(Violation("contacts", f"no METAL polygon with props terminal='{term}'"))
                continue
            x0, x1 = xs.contacts[term]
            d = (xs.rib_x0 - x1) if x1 <= xs.rib_x0 else (x0 - xs.rib_x1) if x0 >= xs.rib_x1 else 0.0
            if d < a.min_contact_to_rib:
                v.append(Violation("contact_distance", f"{term} contact is {d:.2f} um from the rib sidewall (< {a.min_contact_to_rib} um): metal absorption"))
            sel = (xs.x >= x0) & (xs.x <= x1)
            N = (xs.NA if side == "p" else xs.ND)[sel]
            if not sel.any() or N.max() < 1e19:
                v.append(Violation("contact_doping", f"{term} contact must sit on >= 1e19 cm^-3 {side}-type silicon (ohmic contact)"))
            if xs.h[sel].max() < xs.rib_h - 1e-6 and xs.h[sel].max() < xs.slab_h + 1e-6:
                pass  # contact on slab is allowed
        # heavy doping distance
        for side, sign in (("p", -1), ("n", +1)):
            N = xs.NA if side == "p" else xs.ND
            outside = (xs.x < xs.rib_x0) if sign < 0 else (xs.x > xs.rib_x1)
            heavy = outside & (N >= 1e19)
            if heavy.any():
                edge = xs.rib_x0 if sign < 0 else xs.rib_x1
                d = float(np.min(np.abs(xs.x[heavy] - edge)))
                if d < a.min_pp_to_rib:
                    v.append(Violation("pp_distance", f"{side}++ doping is {d:.2f} um from the rib sidewall (< {a.min_pp_to_rib} um): free-carrier loss"))
            inside = ~outside & (N >= 1e19)
            if inside.any():
                v.append(Violation("pp_in_rib", f">= 1e19 cm^-3 {side}-type doping inside the rib"))
        return v

    # -- judge pieces ---------------------------------------------------------------
    def decompose(self, result: PNResult, cfg) -> PNBreakdown:
        return decompose_pn(result, cfg)

    def fom_and_constraints(self, bd: PNBreakdown, cfg) -> dict:
        a = cfg.active
        s = bd.summary()
        viol = []
        score = 0.0
        if s["alpha_dB_cm"] > a.max_alpha_db_cm:
            viol.append(f"alpha {s['alpha_dB_cm']:.1f} dB/cm > {a.max_alpha_db_cm}")
            score += (s["alpha_dB_cm"] - a.max_alpha_db_cm) / a.max_alpha_db_cm
        if s["VpiL_Vcm"] > a.max_vpil_vcm:
            viol.append(f"VpiL {s['VpiL_Vcm']:.2f} V cm > {a.max_vpil_vcm}")
            score += (s["VpiL_Vcm"] - a.max_vpil_vcm) / a.max_vpil_vcm
        if s["f_rc_GHz"] < a.min_f_rc_ghz:
            viol.append(f"f_RC {s['f_rc_GHz']:.1f} GHz < {a.min_f_rc_ghz}")
            score += (a.min_f_rc_ghz - s["f_rc_GHz"]) / a.min_f_rc_ghz
        return {"fom": s["FOM_VdB"], "constraints_ok": not viol, "constraint_violations": viol, "violation_score": score}

    def summary_text(self, s: dict) -> str:
        return (
            f"FOM {s['FOM_VdB']:.2f} V dB = VpiL {s['VpiL_Vcm']:.2f} V cm x alpha {s['alpha_dB_cm']:.1f} dB/cm @ {s['v_ref']:.1f} V; "
            f"f_RC {s['f_rc_GHz']:.0f} GHz (C {s['C_fF_um']:.3f} fF/um, RxL {s['R_ohm_um']:.0f} ohm um); "
            f"dneff core {s['dneff_core_frac']:.0%} / slab {s['dneff_slab_frac']:.0%}; alpha: p {s['alpha_fca_p_dB_cm']:.1f}, n {s['alpha_fca_n_dB_cm']:.1f}, "
            f"contacts {s['alpha_contacts_dB_cm']:.1f}, bg {s['alpha_background_dB_cm']:.1f}; closure err {s['closure_err']:.3f}"
        )

    def digest_terms(self, s: dict) -> str:
        return f"VpiL {s['VpiL_Vcm']:.2f}, alpha {s['alpha_dB_cm']:.1f} (p {s['alpha_fca_p_dB_cm']:.1f}/n {s['alpha_fca_n_dB_cm']:.1f}/ct {s['alpha_contacts_dB_cm']:.1f}), f_RC {s['f_rc_GHz']:.0f} GHz, W {s['W_dep_um']:.3f}"

    def improve_threshold(self, cfg, best_fom: Optional[float]) -> float:
        return cfg.active.improve_threshold_rel * (best_fom if best_fom else 1.0)

    def table_from_dict(self, bd: dict) -> str:
        return _table_rows(PNResult.from_dict(bd["result"]), bd["closure_err"])


def _table_rows(r: PNResult, closure_err: float) -> str:
    rows = ["V(V)   dneff    core     slab   | alpha  fca_p  fca_n  cont   bg   | C(fF/um) W(um)"]
    for i, V in enumerate(r.voltages):
        rows.append(
            f"{V:5.2f}  {r.dneff_total[i]:.2e} {r.dneff_core[i]:.2e} {r.dneff_slab[i]:.2e} | "
            f"{r.alpha_total[i]:5.1f}  {r.alpha_fca_p[i]:5.1f}  {r.alpha_fca_n[i]:5.1f}  {r.alpha_contacts[i]:4.1f}  {r.alpha_background[i]:3.1f} | "
            f"{r.C_fF_um[i]:6.3f}   {r.W_dep_um[i]:.3f}"
        )
    rows.append(f"R x L = {r.R_ohm_m*1e6:.1f} ohm um; closure err {closure_err:.3f}")
    return "\n".join(rows)
