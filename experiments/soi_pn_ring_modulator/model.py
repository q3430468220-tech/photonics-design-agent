"""model.py - the agent's understanding of the ring modulator, as code.

Contract:  predict(geom) -> dict(fom=<1 - dT>, sigma=<...>, breakdown=<optional>, notes=<str>)

The features (family_for(CFG).features(geom, CFG)) include the cross-section features and
radius_um, gap_um, doped_fraction. A useful model predicts Q_L from bend + FCA + coupling,
the shift from VpiL and the doped fraction, and dT from shift / linewidth near critical coupling.

Initial model: a prior only. A 10 um lateral-PN ring at 2 V swing typically reaches a
transmission swing of 0.3-0.5, i.e. FOM ~ 0.6.
"""
from prism_min.geometry import Geometry


def predict(geom: Geometry) -> dict:
    return {"fom": 0.6, "sigma": 0.15, "notes": "prior: dT ~ 0.4 for a 10 um lateral-PN ring at 2 V; no geometry dependence yet"}
