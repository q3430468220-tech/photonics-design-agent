"""design.py - the ONLY file the agent edits in a design turn (pn_ring_modulator family).

Contract:  build() -> prism_min.geometry.Geometry(plane="xz", meta={ring parameters})
  * the cross-section follows the phase-shifter rules (see experiments/soi_pn_phase_shifter)
  * meta = {"radius_um": 3-50, "gap_um": 0.1-1.0, "doped_fraction": 0.1-1}

Starting point: textbook lateral 5e17/5e17 junction, R = 10 um, gap 0.2 um, 85 % doped.
"""
from prism_min.geometry import Geometry, xz_rect

H_CORE, H_SLAB, X_MAX = 0.22, 0.09, 3.0
W_RIB, X_J = 0.50, 0.00
NA, ND = 5e17, 5e17
D_PP, N_PP = 0.80, 1e20
D_CT, W_CT = 1.20, 0.50

RADIUS_UM = 10.0
GAP_UM = 0.20
DOPED_FRACTION = 0.85


def build() -> Geometry:
    xr = W_RIB / 2
    polys = [
        xz_rect(-X_MAX, X_MAX, 0.0, H_SLAB, "SI"),
        xz_rect(-xr, xr, 0.0, H_CORE, "SI"),
        xz_rect(-X_MAX, X_J, 0.0, H_CORE, "P", doping=NA, dopant="p"),
        xz_rect(X_J, X_MAX, 0.0, H_CORE, "N", doping=ND, dopant="n"),
        xz_rect(-X_MAX, -xr - D_PP, 0.0, H_CORE, "P++", doping=N_PP, dopant="p"),
        xz_rect(xr + D_PP, X_MAX, 0.0, H_CORE, "N++", doping=N_PP, dopant="n"),
        xz_rect(-xr - D_CT - W_CT, -xr - D_CT, H_SLAB, H_SLAB + 0.3, "METAL", terminal="anode"),
        xz_rect(xr + D_CT, xr + D_CT + W_CT, H_SLAB, H_SLAB + 0.3, "METAL", terminal="cathode"),
    ]
    return Geometry(polys, [], thickness=H_CORE, plane="xz", meta={"radius_um": RADIUS_UM, "gap_um": GAP_UM, "doped_fraction": DOPED_FRACTION})
