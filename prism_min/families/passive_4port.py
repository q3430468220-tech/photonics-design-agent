"""Passive 4-port family (waveguide crossing and similar): wraps the original passive code."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .. import decompose as _dec
from ..checks import sim_sanity
from ..features import crossing_features
from . import Family, register


@register
class Passive4Port(Family):
    name = "passive_4port"
    default_backends = ("lumerical", "mock")
    fom_label = "worst-case insertion loss (dB, lower is better)"
    fom_unit = "dB"
    terms_help = (
        "Breakdown terms (fractions of injected power, per wavelength): through_TE0 = useful transmission "
        "(FOM = its worst-case loss in dB over the band); through_higher = power at the through port not in "
        "the fundamental mode (output mode mismatch); crosstalk = power into the side ports; reflection = "
        "backward power at the input; radiation = everything else leaving the flux box. box = measured total "
        "outward flux, must be 1 +- closure_tol or the run is voided."
    )

    def features(self, geom, cfg) -> Dict[str, float]:
        return crossing_features(geom, cfg)

    def check(self, geom, cfg):
        return sim_sanity(geom, cfg)

    def decompose(self, result, cfg):
        return _dec.decompose(result, cfg)

    def fom_and_constraints(self, bd, cfg) -> dict:
        out = _dec.fom_and_constraints(bd, cfg)
        s = bd.summary()
        # how far outside the constraints, in dB, for the 'both violate' comparison
        out["violation_score"] = max(0.0, s["crosstalk_worst_dB"] - cfg.max_crosstalk_db) + max(0.0, s["reflection_worst_dB"] - cfg.max_reflection_db)
        return out

    def summary_text(self, s: dict) -> str:
        return (
            f"IL {s['IL_worst_dB']:.4f} dB @ {s['IL_worst_wl_um']:.3f} um; xtalk {s['crosstalk_worst_dB']:.1f} dB; "
            f"refl {s['reflection_worst_dB']:.1f} dB; rad {s['radiation_frac']:.4f}, higher {s['through_higher_frac']:.4f}; closure err {s['closure_err']:.4f}"
        )

    def digest_terms(self, s: dict) -> str:
        return f"rad {s['radiation_frac']:.4f}, xt {s['crosstalk_frac']:.4f}, higher {s['through_higher_frac']:.4f}, refl {s['reflection_frac']:.4f}"

    def improve_threshold(self, cfg, best_fom: Optional[float]) -> float:
        return cfg.improve_threshold_db

    def table_from_dict(self, bd: dict) -> str:
        b = _dec.Breakdown(np.asarray(bd["wavelengths"]), {k: np.asarray(v) for k, v in bd["terms"].items()}, np.asarray(bd["box_total"]), bd["closure_err"], bd["closure_ok"])
        return b.table()
