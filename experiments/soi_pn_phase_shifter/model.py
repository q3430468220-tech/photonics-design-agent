"""model.py - the agent's physical understanding of the PN phase shifter, as code.

Contract:  predict(geom) -> dict(fom=<V dB>, sigma=<V dB>, breakdown=<optional dict>, notes=<str>)

* fom is the predicted VpiL x alpha at v_ref; sigma its 1-sigma uncertainty (V dB).
* Measure the cross-section yourself, e.g.
      from prism_min.config import SimConfig
      from prism_min.families import family_for
      f = family_for(CFG).features(geom, CFG)   # w_rib, x_j_offset, Na_core, Nd_core, d_pp_p/n, d_ct_p/n, N_slab_p/n, W_dep_vref
  or parse_xsection() in prism_min.families.pn_phase_shifter for the full doping profile.

Initial model: a literature prior only. Textbook lateral PN shifters at ~5e17 cm^-3 give
VpiL ~ 2 V cm and ~10-15 dB/cm, i.e. a product of order 25 V dB. No geometry dependence yet.
"""
from prism_min.geometry import Geometry


def predict(geom: Geometry) -> dict:
    return {
        "fom": 25.0,
        "sigma": 8.0,
        "breakdown": {"VpiL_Vcm": 2.0, "alpha_dB_cm": 12.0},
        "notes": "literature prior for a 5e17 lateral PN shifter; no geometry dependence yet",
    }
