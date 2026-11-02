"""Active family: PN microring modulator = pn_phase_shifter cross-section + ring parameters.

The agent edits the same x-z cross-section as in `pn_phase_shifter` and, in `Geometry.meta`,
the ring: {"radius_um": R, "gap_um": g, "doped_fraction": f}. The judge adds the ring level:

    round-trip loss     alpha_rt = alpha_bend(R) + alpha_bg + alpha_contacts + f * alpha_FCA(V)
    coupling            kappa^2 (gap, R)  (toy CMT model until an FDTD coupler run exists)
    Q budget            1/Q_L = 1/Q_c + 1/Q_bend + 1/Q_FCA + 1/Q_bg + 1/Q_contacts
    resonance shift     dlambda(V) = lambda * f * dneff(V) / n_g           (pm/V reported)
    spectra             all-pass T(dlambda; V) for the 'one' (v_high) and 'zero' (v_low) states
    swing               dT_max = max over wavelength of T_one - T_zero; ER, IL at that wavelength
    bandwidth           1/f_3dB^2 = 1/f_RC^2 + 1/f_photon^2,  f_photon = 1/(2 pi tau),
                        tau = Q_L lambda / (2 pi c)
    energy per bit      C_total V^2 / 4 with C_total = C' * 2 pi R * f

FOM = 1 - dT_max (lower is better). Constraints: f_3dB >= min, IL(one) <= max, ER >= min.
Closure: Q_L from the FWHM of the computed spectrum must agree with 1/Q_L = 1/Q_c + 1/Q_i
(a self-consistency check of the lumped model; with a Lumerical ring spectrum it becomes a
real check), and the phase-shifter closure (perturbation terms vs full solve) still applies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..checks import Violation
from ..geometry import Geometry
from . import Family, register
from .pn_phase_shifter import PNBreakdown, PNPhaseShifter, PNResult, decompose_pn

C_LIGHT = 299792458.0


# ----------------------------------------------------------------------------
# Result: phase-shifter result + ring quantities from the backend
# ----------------------------------------------------------------------------


@dataclass
class RingResult(PNResult):
    alpha_bend_db_cm: float = 0.0  # undoped bent minus undoped straight loss
    kappa2: float = 0.0  # power coupling of the bus coupler
    radius_um: float = 10.0
    gap_um: float = 0.2
    doped_fraction: float = 1.0
    coupling_model: str = "toy"

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(
            alpha_bend_db_cm=float(self.alpha_bend_db_cm), kappa2=float(self.kappa2), radius_um=float(self.radius_um),
            gap_um=float(self.gap_um), doped_fraction=float(self.doped_fraction), coupling_model=self.coupling_model,
        )
        return d

    @staticmethod
    def from_dict(d: dict) -> "RingResult":
        base = PNResult.from_dict(d)
        return RingResult(**{k: getattr(base, k) for k in base.__dataclass_fields__}, alpha_bend_db_cm=d.get("alpha_bend_db_cm", 0.0),
                          kappa2=d.get("kappa2", 0.0), radius_um=d.get("radius_um", 10.0), gap_um=d.get("gap_um", 0.2),
                          doped_fraction=d.get("doped_fraction", 1.0), coupling_model=d.get("coupling_model", "toy"))


def ring_params(geom: Geometry, cfg) -> Dict[str, float]:
    m = geom.meta or {}
    return {
        "radius_um": float(m.get("radius_um", 10.0)),
        "gap_um": float(m.get("gap_um", 0.2)),
        "doped_fraction": float(m.get("doped_fraction", 1.0)),
    }


def toy_bend_loss_db_cm(R: float, cfg) -> float:
    r = cfg.ring
    return float(r.bend_loss_A_db_cm * np.exp(-R / r.bend_loss_R0_um))


def toy_coupling(gap: float, R: float, cfg) -> float:
    r = cfg.ring
    return float(min(0.95, r.coupling_k0 * np.sqrt(max(R, 1.0) / 10.0) * np.exp(-(gap - 0.1) / r.coupling_decay_um)))


# ----------------------------------------------------------------------------
# Ring physics
# ----------------------------------------------------------------------------


def _db_cm_to_per_um(a_db_cm: float) -> float:
    return a_db_cm * np.log(10) / 10.0 * 1e-4


def ring_spectra(res: RingResult, cfg) -> dict:
    """All-pass transmission spectra and the derived ring quantities at the two states."""
    a, r = cfg.active, cfg.ring
    lam = a.wavelength  # um
    R, f, k2 = res.radius_um, res.doped_fraction, res.kappa2
    L = 2 * np.pi * R
    v_low = a.v_ref if r.v_low is None else r.v_low
    iH = int(np.argmin(np.abs(res.voltages - r.v_high)))
    iL = int(np.argmin(np.abs(res.voltages - v_low)))

    def alpha_terms(i):
        fca = f * (res.alpha_fca_p[i] + res.alpha_fca_n[i])
        return {
            "bend": res.alpha_bend_db_cm,
            "background": float(res.alpha_background[i]),
            "contacts": float(res.alpha_contacts[i]),
            "fca": float(fca),
        }

    def Q_of(alpha_db_cm):
        al = _db_cm_to_per_um(alpha_db_cm)  # 1/um
        return (2 * np.pi * r.n_g / (lam * al)) if al > 0 else np.inf

    t = np.sqrt(max(1 - k2, 0.0))
    Qc = 2 * np.pi * r.n_g * L / (lam * max(k2, 1e-9))
    dl = np.linspace(-0.6, 0.6, 4001) * lam**2 / (r.n_g * L)  # +-0.6 FSR in um
    out = {"FSR_nm": 1e3 * lam**2 / (r.n_g * L), "Q_c": Qc, "states": {}}
    spectra = {}
    for name, i in (("one", iH), ("zero", iL)):
        terms = alpha_terms(i)
        a_tot = sum(terms.values())
        a_rt = np.exp(-_db_cm_to_per_um(a_tot) * L / 2)  # round-trip amplitude
        shift = lam * f * float(res.dneff_total[i]) / r.n_g  # um, resonance shift vs 0 V
        phi = 2 * np.pi * r.n_g * L * (dl - shift) / lam**2  # detuning phase (resonance of 0 V at dl=0)
        T = (a_rt**2 - 2 * a_rt * t * np.cos(phi) + t**2) / (1 - 2 * a_rt * t * np.cos(phi) + (a_rt * t) ** 2)
        Qi_terms = {k: Q_of(v) for k, v in terms.items()}
        Qi = 1.0 / sum(1.0 / q for q in Qi_terms.values() if np.isfinite(q))
        QL_model = 1.0 / (1.0 / Qc + 1.0 / Qi)
        # FWHM of the computed dip
        Tmin = T.min()
        half = 0.5 * (1 + Tmin) if Tmin < 1 else 1
        below = dl[T <= half]
        fwhm = float(below.max() - below.min()) if len(below) > 1 else np.nan
        QL_fit = lam / fwhm if (fwhm and fwhm > 0) else np.nan
        spectra[name] = T
        out["states"][name] = {
            "V": float(res.voltages[i]),
            "alpha_terms_db_cm": terms,
            "alpha_rt_db_cm": float(a_tot),
            "Q_i_terms": {k: float(v) for k, v in Qi_terms.items()},
            "Q_i": float(Qi),
            "Q_L_model": float(QL_model),
            "Q_L_fit": float(QL_fit),
            "shift_pm": float(shift * 1e6),
            "T_min": float(Tmin),
            "coupling": "under" if a_rt > t else "over",
        }
    dT = spectra["one"] - spectra["zero"]
    j = int(np.argmax(dT))
    T1, T0 = float(spectra["one"][j]), float(spectra["zero"][j])
    out.update(
        dT_max=float(dT[j]),
        op_detuning_pm=float(dl[j] * 1e6),
        ER_dB=float(10 * np.log10(max(T1, 1e-12) / max(T0, 1e-12))),
        IL_dB=float(-10 * np.log10(max(T1, 1e-12))),
        shift_pm_per_V=float(abs(out["states"]["zero"]["shift_pm"] - out["states"]["one"]["shift_pm"]) / max(abs(v_low - r.v_high), 1e-9)),
    )
    QL = out["states"]["one"]["Q_L_model"]
    tau = QL * lam * 1e-6 / (2 * np.pi * C_LIGHT)
    f_ph = 1.0 / (2 * np.pi * tau) / 1e9
    C_total_fF = float(res.C_fF_um[iL]) * L * f
    f_rc = 1.0 / (2 * np.pi * res.R_ohm_m * float(res.C_fF_um[iL]) * 1e-15 / 1e-6) / 1e9 if res.R_ohm_m > 0 else np.inf
    out.update(
        f_photon_GHz=float(f_ph),
        f_rc_GHz=float(f_rc),
        f_3dB_GHz=float(1.0 / np.sqrt(1.0 / f_ph**2 + 1.0 / f_rc**2)),
        C_total_fF=C_total_fF,
        energy_fJ_per_bit=float(C_total_fF * (v_low - r.v_high) ** 2 / 4),
        Q_L=float(QL),
        dl_um=dl,
        T_one=spectra["one"],
        T_zero=spectra["zero"],
    )
    return out


# ----------------------------------------------------------------------------
# Breakdown
# ----------------------------------------------------------------------------


@dataclass
class RingBreakdown:
    res: RingResult
    cfg: object
    pn: PNBreakdown
    ring: dict
    closure_err: float
    closure_ok: bool

    def summary(self) -> dict:
        s = self.pn.summary()
        g = self.ring
        one, zero = g["states"]["one"], g["states"]["zero"]
        s.update(
            {
                "radius_um": self.res.radius_um,
                "gap_um": self.res.gap_um,
                "doped_fraction": self.res.doped_fraction,
                "kappa2": self.res.kappa2,
                "alpha_bend_dB_cm": self.res.alpha_bend_db_cm,
                "FSR_nm": g["FSR_nm"],
                "Q_c": g["Q_c"],
                "Q_i_one": one["Q_i"],
                "Q_L": g["Q_L"],
                "Q_L_fit": one["Q_L_fit"],
                "Q_bend": one["Q_i_terms"]["bend"],
                "Q_fca_one": one["Q_i_terms"]["fca"],
                "Q_bg": one["Q_i_terms"]["background"],
                "Q_contacts": one["Q_i_terms"]["contacts"],
                "coupling_regime": one["coupling"],
                "shift_pm_per_V": g["shift_pm_per_V"],
                "dT_max": g["dT_max"],
                "ER_dB": g["ER_dB"],
                "IL_dB": g["IL_dB"],
                "op_detuning_pm": g["op_detuning_pm"],
                "f_photon_GHz": g["f_photon_GHz"],
                "f_3dB_GHz": g["f_3dB_GHz"],
                "C_total_fF": g["C_total_fF"],
                "energy_fJ_per_bit": g["energy_fJ_per_bit"],
                "FOM_ring": 1.0 - g["dT_max"],
                "closure_err": float(self.closure_err),
                "closure_ok": bool(self.closure_ok),
            }
        )
        return s

    def table(self) -> str:
        return _table_from_stored(self.to_dict())

    def to_dict(self) -> dict:
        g = {k: v for k, v in self.ring.items() if k not in ("dl_um", "T_one", "T_zero")}
        return {"family": "pn_ring_modulator", "result": self.res.to_dict(), "ring": g, "closure_err": float(self.closure_err), "closure_ok": bool(self.closure_ok), "summary": self.summary()}


def decompose_ring(res: RingResult, cfg) -> RingBreakdown:
    pn = decompose_pn(res, cfg)
    g = ring_spectra(res, cfg)
    one = g["states"]["one"]
    q_err = abs(one["Q_L_fit"] - one["Q_L_model"]) / one["Q_L_model"] if np.isfinite(one["Q_L_fit"]) else 1.0
    err = float(max(pn.closure_err, min(q_err, 1.0)))
    ok = bool(pn.closure_ok and q_err <= 0.15)
    return RingBreakdown(res, cfg, pn, g, err, ok)


# ----------------------------------------------------------------------------
# Family
# ----------------------------------------------------------------------------


@register
class PNRingModulator(PNPhaseShifter):
    name = "pn_ring_modulator"
    default_backends = ("lumerical_ring", "mock_ring")
    fom_label = "1 - max transmission swing dT (lower is better)"
    fom_unit = ""
    terms_help = (
        PNPhaseShifter.terms_help
        + " Ring level: alpha_rt = bend + background + contacts + f x FCA (f = doped fraction) sets Q_i; kappa^2 (gap, R) "
        "sets Q_c; 1/Q_L = 1/Q_c + 1/Q_i. Resonance shift = lambda f dneff / n_g. The 'one' state (v_high) and 'zero' state "
        "(v_low) spectra give the swing dT_max at the best wavelength, with ER and IL there. f_3dB combines f_RC with the "
        "photon-lifetime limit 1/(2 pi tau), tau = Q_L lambda / (2 pi c): high Q helps the swing but caps the bandwidth. "
        "Critical coupling (a_rt = t) maximises ER; the swing is largest slightly under-coupled with a shift of about one linewidth."
    )

    def features(self, geom: Geometry, cfg) -> Dict[str, float]:
        f = super().features(geom, cfg)
        f.update(ring_params(geom, cfg))
        return f

    def check(self, geom: Geometry, cfg) -> List[Violation]:
        v = super().check(geom, cfg)
        r = cfg.ring
        p = ring_params(geom, cfg)
        if not (r.radius_min_um <= p["radius_um"] <= r.radius_max_um):
            v.append(Violation("ring_radius", f"radius_um {p['radius_um']} outside [{r.radius_min_um}, {r.radius_max_um}]"))
        if not (r.gap_min_um <= p["gap_um"] <= r.gap_max_um):
            v.append(Violation("ring_gap", f"gap_um {p['gap_um']} outside [{r.gap_min_um}, {r.gap_max_um}]"))
        if not (r.doped_fraction_min <= p["doped_fraction"] <= 1.0):
            v.append(Violation("doped_fraction", f"doped_fraction {p['doped_fraction']} outside [{r.doped_fraction_min}, 1]"))
        return v

    def decompose(self, result: RingResult, cfg) -> RingBreakdown:
        return decompose_ring(result, cfg)

    def fom_and_constraints(self, bd: RingBreakdown, cfg) -> dict:
        r = cfg.ring
        s = bd.summary()
        viol, score = [], 0.0
        if s["f_3dB_GHz"] < r.min_f3db_ghz:
            viol.append(f"f_3dB {s['f_3dB_GHz']:.1f} GHz < {r.min_f3db_ghz}")
            score += (r.min_f3db_ghz - s["f_3dB_GHz"]) / r.min_f3db_ghz
        if s["IL_dB"] > r.max_il_db:
            viol.append(f"IL(one) {s['IL_dB']:.2f} dB > {r.max_il_db}")
            score += (s["IL_dB"] - r.max_il_db) / r.max_il_db
        if s["ER_dB"] < r.min_er_db:
            viol.append(f"ER {s['ER_dB']:.2f} dB < {r.min_er_db}")
            score += (r.min_er_db - s["ER_dB"]) / r.min_er_db
        return {"fom": s["FOM_ring"], "constraints_ok": not viol, "constraint_violations": viol, "violation_score": score}

    def summary_text(self, s: dict) -> str:
        return (
            f"dT {s['dT_max']:.3f} (ER {s['ER_dB']:.1f} dB, IL {s['IL_dB']:.2f} dB) @ R {s['radius_um']:.1f} um gap {s['gap_um']:.2f} f {s['doped_fraction']:.2f}; "
            f"Q_L {s['Q_L']:.0f} (Q_c {s['Q_c']:.0f}, Q_i {s['Q_i_one']:.0f}: bend {s['Q_bend']:.0f}, FCA {s['Q_fca_one']:.0f}), {s['coupling_regime']}-coupled; "
            f"{s['shift_pm_per_V']:.1f} pm/V; f_3dB {s['f_3dB_GHz']:.0f} GHz (photon {s['f_photon_GHz']:.0f}, RC {s['f_rc_GHz']:.0f}); {s['energy_fJ_per_bit']:.0f} fJ/bit; "
            f"shifter VpiL {s['VpiL_Vcm']:.2f} V cm, alpha {s['alpha_dB_cm']:.1f} dB/cm; closure err {s['closure_err']:.3f}"
        )

    def digest_terms(self, s: dict) -> str:
        return f"dT {s['dT_max']:.3f}, ER {s['ER_dB']:.1f}, IL {s['IL_dB']:.2f}, Q_L {s['Q_L']:.0f} ({s['coupling_regime']}), f_3dB {s['f_3dB_GHz']:.0f}, VpiL {s['VpiL_Vcm']:.2f}, R {s['radius_um']:.1f}, gap {s['gap_um']:.2f}"

    def improve_threshold(self, cfg, best_fom: Optional[float]) -> float:
        return cfg.ring.improve_threshold_abs

    def table_from_dict(self, bd: dict) -> str:
        return _table_from_stored(bd)


def _table_from_stored(bd: dict) -> str:
    """Rebuild the table from the stored ring dict (no cfg needed)."""
    from .pn_phase_shifter import _table_rows

    res = RingResult.from_dict(bd["result"])
    g = bd["ring"]
    rows = [
        f"ring R {res.radius_um:.2f} um, gap {res.gap_um:.3f} um, doped fraction {res.doped_fraction:.2f}, kappa^2 {res.kappa2:.4f} ({res.coupling_model}), FSR {g['FSR_nm']:.2f} nm",
        "state   V     alpha_rt(dB/cm) = bend + bg + contacts + f*FCA   |  Q_i      Q_c      Q_L(model) Q_L(fit)  shift(pm) T_min  coupling",
    ]
    for name in ("one", "zero"):
        st = g["states"][name]
        a = st["alpha_terms_db_cm"]
        rows.append(
            f"{name:5s} {st['V']:5.2f}  {st['alpha_rt_db_cm']:7.2f} = {a['bend']:5.2f} + {a['background']:4.2f} + {a['contacts']:5.2f} + {a['fca']:5.2f}   | "
            f"{st['Q_i']:8.0f} {g['Q_c']:8.0f} {st['Q_L_model']:9.0f} {st['Q_L_fit']:9.0f} {st['shift_pm']:9.1f} {st['T_min']:6.3f}  {st['coupling']}"
        )
    rows.append(
        f"swing dT_max {g['dT_max']:.3f} at {g['op_detuning_pm']:+.0f} pm: ER {g['ER_dB']:.2f} dB, IL {g['IL_dB']:.2f} dB; {g['shift_pm_per_V']:.1f} pm/V; "
        f"f_photon {g['f_photon_GHz']:.0f} GHz, f_RC {g['f_rc_GHz']:.0f} GHz -> f_3dB {g['f_3dB_GHz']:.0f} GHz; C_total {g['C_total_fF']:.1f} fF, {g['energy_fJ_per_bit']:.1f} fJ/bit"
    )
    rows.append("--- phase shifter cross-section ---")
    rows.append(_table_rows(res, bd["closure_err"]))
    return "\n".join(rows)
