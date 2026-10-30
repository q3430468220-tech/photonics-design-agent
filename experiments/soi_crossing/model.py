"""model.py - the agent's current physical understanding, as code.

Contract:  predict(geom) -> dict(fom=<dB>, sigma=<dB>, breakdown=<optional dict>, notes=<str>)

* `fom` is the predicted worst-case insertion loss in dB; `sigma` its 1-sigma uncertainty.
* `breakdown` (optional) may predict fractions for: through_higher, crosstalk,
  reflection, radiation - the journal will compare them with the measurement.
* The loop registers this prediction BEFORE every simulation. If the measured FOM
  misses by > 2 sigma, the next turn must revise this file instead of the design.
* Measure the geometry yourself (prism_min.geometry.rasterize / run_profile_x etc.);
  do not rely on constants hidden in design.py.

Initial model: a literature prior only. A plain 500 nm crossing on 220 nm SOI
typically loses about 1 dB; we do not yet know how this changes with geometry,
so the uncertainty is deliberately wide.
"""
from prism_min.geometry import Geometry


def predict(geom: Geometry) -> dict:
    return {
        "fom": 1.2,
        "sigma": 0.6,
        "breakdown": {"crosstalk": 0.08, "reflection": 0.003},
        "notes": "literature prior for a plain crossing; no geometry dependence yet",
    }
