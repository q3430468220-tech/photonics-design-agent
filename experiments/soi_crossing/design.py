"""design.py - the ONLY file the agent edits in a design turn.

Contract:  build() -> prism_min.geometry.Geometry
  * units are micrometres; single 220 nm silicon layer
  * ports W, E, N, S must sit on the bbox edges, width 0.5, orientations 180/0/90/270
  * keep >= 1 um of uniform 0.5 um waveguide behind every port
  * the device must be 4-fold symmetric about the origin
  * footprint <= 40 x 40 um; min feature / gap 150 nm

Helpers available from prism_min.geometry:
  rect(x0, y0, x1, y1)                 axis-aligned rectangle
  straight(length, width, center, angle)
  taper(x0, w0, x1, w1)                linear taper along +x
  profile_taper(xs, ws)                taper from a sampled width profile w(x)
  rotated(poly, angle_deg)             rotate about the origin
  four_fold(polys)                     replicate under 0/90/180/270 deg rotations
  Polygon(points), Port(name, center, width, orientation), Geometry(polys, ports)

Starting point: a plain crossing of two 0.5 um waveguides, arms 5 um long.
"""
import numpy as np

from prism_min.geometry import Geometry, Port, four_fold, rect

W_PORT = 0.5  # um, fixed by the interface
ARM = 5.0  # um, half-length of each arm (port at x = -ARM)


def build() -> Geometry:
    # one arm from the west port to the centre; four_fold() makes the other three
    arm = [rect(-ARM, -W_PORT / 2, 0.0, W_PORT / 2)]
    polys = four_fold(arm)
    ports = [
        Port("W", (-ARM, 0.0), W_PORT, 180),
        Port("E", (ARM, 0.0), W_PORT, 0),
        Port("N", (0.0, ARM), W_PORT, 90),
        Port("S", (0.0, -ARM), W_PORT, 270),
    ]
    return Geometry(polys, ports, thickness=0.22)
