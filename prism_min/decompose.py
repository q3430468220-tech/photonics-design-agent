"""Loss decomposition + closure for a passive multi-port device.

From one simulation we get, per wavelength and normalised to injected power:

    through_TE0     power in the fundamental mode at the through port (the useful part)
    through_higher  power at the through port NOT in the fundamental mode
                    (higher-order / unguided near-field: mode mismatch at the output)
    crosstalk       sum of power leaving through the side ports
    reflection      power returning to the input port
    radiation       everything else that left the enclosing flux box
    ----------------------------------------------------------------
    box_total       measured sum of all outward flux (closure: must be ~1)

`radiation` is defined as `box_total - (guided terms)`, so the breakdown always
sums to `box_total` by construction; the *closure check* is whether box_total
itself equals 1. A violated closure means the measurement is untrustworthy
(unconverged run, PML too close, absorption, mesh error) and the run is voided.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .backends.base import SimResult
from .config import SimConfig

TERMS = ["through_TE0", "through_higher", "crosstalk", "reflection", "radiation"]


def db(x) -> np.ndarray:
    x = np.asarray(x, float)
    return 10.0 * np.log10(np.clip(x, 1e-12, None))


@dataclass
class Breakdown:
    wavelengths: np.ndarray
    terms: Dict[str, np.ndarray]  # fractions of injected power
    box_total: np.ndarray
    closure_err: float  # max |box_total - 1| over band
    closure_ok: bool

    # ---- derived scalars (worst case over band unless stated) -------------
    def il_db(self) -> np.ndarray:
        return -db(self.terms["through_TE0"])

    def summary(self) -> dict:
        il = self.il_db()
        i_worst = int(np.argmax(il))
        out = {
            "IL_worst_dB": float(il.max()),
            "IL_mean_dB": float(il.mean()),
            "IL_worst_wl_um": float(self.wavelengths[i_worst]),
            "crosstalk_worst_dB": float(db(self.terms["crosstalk"]).max()),
            "reflection_worst_dB": float(db(self.terms["reflection"]).max()),
            "closure_err": float(self.closure_err),
            "closure_ok": bool(self.closure_ok),
        }
        # loss budget at the worst wavelength, in dB-equivalent and in % of input
        for k in TERMS:
            if k == "through_TE0":
                continue
            frac = float(self.terms[k][i_worst])
            out[f"{k}_frac"] = frac
            out[f"{k}_dB_equiv"] = float(-10 * np.log10(max(1 - frac, 1e-12)))
        return out

    def table(self) -> str:
        """Compact per-wavelength table for the journal / agent."""
        hdr = "wl(um)  IL(dB)  TE0     higher  xtalk   refl    rad     box"
        rows = [hdr]
        il = self.il_db()
        for i, wl in enumerate(self.wavelengths):
            t = self.terms
            rows.append(
                f"{wl:6.4f}  {il[i]:6.3f}  {t['through_TE0'][i]:.4f}  {t['through_higher'][i]:.4f}  "
                f"{t['crosstalk'][i]:.4f}  {t['reflection'][i]:.4f}  {t['radiation'][i]:.4f}  {self.box_total[i]:.4f}"
            )
        return "\n".join(rows)

    def to_dict(self) -> dict:
        return {
            "wavelengths": self.wavelengths.tolist(),
            "terms": {k: v.tolist() for k, v in self.terms.items()},
            "box_total": self.box_total.tolist(),
            "closure_err": float(self.closure_err),
            "closure_ok": bool(self.closure_ok),
            "summary": self.summary(),
        }


def decompose(res: SimResult, cfg: SimConfig) -> Breakdown:
    th_fund = np.clip(res.through_fund, 0.0, None)
    th_total = np.clip(res.through_total, 0.0, None)
    higher = np.clip(th_total - th_fund, 0.0, None)
    xt = np.zeros_like(th_fund)
    for name in cfg.side_ports:
        xt = xt + np.clip(res.side[name], 0.0, None)
    refl = np.clip(res.reflection, 0.0, None)
    box = np.asarray(res.box_total, float)
    rad = box - (th_total + xt + refl)
    closure_err = float(np.max(np.abs(box - 1.0)))
    ok = bool(closure_err <= cfg.closure_tol and res.converged)
    terms = {
        "through_TE0": th_fund,
        "through_higher": higher,
        "crosstalk": xt,
        "reflection": refl,
        "radiation": rad,
    }
    return Breakdown(np.asarray(res.wavelengths, float), terms, box, closure_err, ok)


def fom_and_constraints(bd: Breakdown, cfg: SimConfig) -> dict:
    """Scalar FOM (minimise) + constraint status, both from the breakdown."""
    s = bd.summary()
    violations: List[str] = []
    if s["crosstalk_worst_dB"] > cfg.max_crosstalk_db:
        violations.append(f"crosstalk {s['crosstalk_worst_dB']:.1f} dB > {cfg.max_crosstalk_db} dB")
    if s["reflection_worst_dB"] > cfg.max_reflection_db:
        violations.append(f"reflection {s['reflection_worst_dB']:.1f} dB > {cfg.max_reflection_db} dB")
    return {"fom": s["IL_worst_dB"], "constraints_ok": not violations, "constraint_violations": violations}
