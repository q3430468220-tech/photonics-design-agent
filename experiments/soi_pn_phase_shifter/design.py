"""design.py - the ONLY file the agent edits in a design turn (pn_phase_shifter family).

Contract:  build() -> prism_min.geometry.Geometry(plane="xz")
  * x lateral, z height above the BOX, micrometres; all polygons axis-aligned rectangles
  * layer SI: rib (full 0.22 um) and slab (0.09 um) - both heights are locked
  * implant windows on P / N / P+ / N+ / P++ / N++ with props doping (cm^-3) and dopant;
    windows add up (net = donors - acceptors) exactly as CHARGE doping objects do
  * two METAL contacts with props terminal="anode" (p side) / "cathode" (n side), each on
    >= 1e19 silicon of its type, >= 0.6 um from the rib sidewall
  * heavy (>= 1e19) doping >= 0.4 um from the rib sidewall; half width <= 6 um

Helper: prism_min.geometry.xz_rect(x0, x1, z0, z1, layer, **props)

Junction shapes other than a straight lateral junction are drawn with several windows,
e.g. an L-shaped junction = p on the left AND under the centre, n on the right AND above
the centre (the judge's toy model resolves lateral and vertical junction segments row by
row; CHARGE resolves anything). Interleaved junctions vary along the propagation
direction and cannot be represented in a single cross-section.

Starting point: symmetric 5e17 / 5e17 junction at the rib centre, P++/N++ 0.8 um from the
rib, contacts 1.2 um from the rib - a textbook lateral PN phase shifter.
"""
from prism_min.geometry import Geometry, xz_rect

H_CORE = 0.22  # locked
H_SLAB = 0.09  # locked
X_MAX = 3.0  # half width of the modelled cross-section

W_RIB = 0.50  # rib width
X_J = 0.00  # metallurgical junction position (x of the p/n boundary)
NA, ND = 5e17, 5e17  # core doping, cm^-3
D_PP = 0.80  # rib sidewall -> start of P++ / N++
N_PP = 1e20
D_CT = 1.20  # rib sidewall -> contact inner edge
W_CT = 0.50  # contact width


def build() -> Geometry:
    xr = W_RIB / 2
    polys = [
        xz_rect(-X_MAX, X_MAX, 0.0, H_SLAB, "SI"),  # slab
        xz_rect(-xr, xr, 0.0, H_CORE, "SI"),  # rib
        xz_rect(-X_MAX, X_J, 0.0, H_CORE, "P", doping=NA, dopant="p"),
        xz_rect(X_J, X_MAX, 0.0, H_CORE, "N", doping=ND, dopant="n"),
        xz_rect(-X_MAX, -xr - D_PP, 0.0, H_CORE, "P++", doping=N_PP, dopant="p"),
        xz_rect(xr + D_PP, X_MAX, 0.0, H_CORE, "N++", doping=N_PP, dopant="n"),
        xz_rect(-xr - D_CT - W_CT, -xr - D_CT, H_SLAB, H_SLAB + 0.3, "METAL", terminal="anode"),
        xz_rect(xr + D_CT, xr + D_CT + W_CT, H_SLAB, H_SLAB + 0.3, "METAL", terminal="cathode"),
    ]
    return Geometry(polys, [], thickness=H_CORE, plane="xz")
