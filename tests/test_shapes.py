import math
import unittest

import numpy as np

from prism_min.config import SimConfig
from prism_min.checks import errors_only, run_drc
from prism_min.geometry import Geometry, Port, rasterize, rect
from prism_min.shapes import annulus, arc, bezier_path, euler_bend, euler_centerline, mirrored, path_to_polygon, periodic_array, sbend


class TestShapes(unittest.TestCase):
    def test_arc_geometry(self):
        a = arc(5, 90, 0.5)
        self.assertAlmostEqual(a.area(), math.pi / 2 * 5 * 0.5, delta=0.02)
        pts = a.points
        n = len(pts) // 2
        end = (pts[n - 1] + pts[n]) / 2
        np.testing.assert_allclose(end, [5, 5], atol=0.02)
        cw = arc(5, -90, 0.5)
        np.testing.assert_allclose((cw.points[n - 1] + cw.points[n]) / 2, [5, -5], atol=0.02)

    def test_euler(self):
        cl = euler_centerline(5, 90)
        heading = math.degrees(math.atan2(*(cl[-1] - cl[-2])[::-1]))
        self.assertAlmostEqual(heading, 90, delta=0.5)
        self.assertGreater(cl[-1][0], 5)  # effective radius larger than the minimum radius
        eb = euler_bend(5, 90, 0.5)
        self.assertGreater(eb.area(), 0.5 * 5 * math.pi / 2)
        arc_like = euler_centerline(5, 90, p=0.0)
        np.testing.assert_allclose(arc_like[-1], [5, 5], atol=0.05)

    def test_sbend_bezier_path(self):
        s = sbend(10, 3, 0.5)
        self.assertAlmostEqual(s.points[len(s.points) // 2 - 1][1], 3.25, delta=0.01)
        b = bezier_path([(0, 0), (5, 0), (5, 5), (10, 5)], [0.5, 2.0, 0.5])
        self.assertGreater(b.area(), 10 * 0.5)
        with self.assertRaises(ValueError):
            path_to_polygon(np.array([[0, 0]]), 0.5)

    def test_annulus_even_odd(self):
        r = rasterize(Geometry([annulus(5, 0.5)], []), 0.02)
        self.assertAlmostEqual(r.mask.sum() * 0.02**2, 2 * math.pi * 5 * 0.5, delta=0.1)
        self.assertFalse(r.filled_at(0.0, 0.0), "the hole must stay empty")

    def test_array_mirror(self):
        arr = periodic_array(rect(0, 0, 0.2, 0.5, "WG", {"k": 1}), 0.4, 5)
        self.assertEqual(len(arr), 5)
        self.assertAlmostEqual(arr[-1].bbox()[0], 1.6)
        self.assertEqual(arr[-1].props, {"k": 1})
        m = mirrored(arc(5, 90, 0.5), "x")
        self.assertLess(m.points[:, 1].min(), -4)

    def test_curved_crossing_passes_drc(self):
        # a crossing with a ring around the junction (curved polygons) must survive the raster DRC
        cfg = SimConfig()
        L, w = 5.0, 0.5
        polys = [rect(-L, -w / 2, L, w / 2), rect(-w / 2, -L, w / 2, L), annulus(1.5, 0.5)]
        g = Geometry(polys, [Port("W", (-L, 0), w, 180), Port("E", (L, 0), w, 0), Port("N", (0, L), w, 90), Port("S", (0, -L), w, 270)])
        self.assertEqual([v.rule for v in errors_only(run_drc(g, cfg))], [])


if __name__ == "__main__":
    unittest.main()
