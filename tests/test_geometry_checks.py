import unittest

import numpy as np

from prism_min.checks import errors_only, run_drc, sim_sanity
from prism_min.config import SimConfig
from prism_min.geometry import Geometry, Polygon, four_fold, rasterize, rect, straight
from tests.helpers import plain, ports, tapered


class TestGeometry(unittest.TestCase):
    def test_hash_stable_and_sensitive(self):
        g1, g2 = plain(), plain()
        self.assertEqual(g1.hash(), g2.hash())
        g3 = plain(L=5.001)
        self.assertNotEqual(g1.hash(), g3.hash())

    def test_roundtrip(self):
        g = tapered()
        g2 = Geometry.from_dict(g.to_dict())
        self.assertEqual(g.hash(), g2.hash())

    def test_polygon_validation(self):
        with self.assertRaises(ValueError):
            Polygon(np.array([[0, 0], [1, 1]]))
        with self.assertRaises(ValueError):
            straight(1, 0.5, center=0)

    def test_run_profile_tracks_one_arm(self):
        g = tapered(L=6, wc=1.5, Lt=3)
        r = rasterize(g, 0.01)
        x, w = r.run_profile_x(0.0)
        i = int(round((1.0 - x[0]) / r.dx))  # x = 1.0: inside the 1.5 um wide centre block, outside the junction
        self.assertAlmostEqual(w[i], 1.5, delta=0.03)
        i = int(round((5.5 - x[0]) / r.dx))  # port straight
        self.assertAlmostEqual(w[i], 0.5, delta=0.03)

    def test_four_fold_symmetric(self):
        g = tapered()
        xmin, ymin, xmax, ymax = g.bbox()
        self.assertAlmostEqual(xmin, -xmax)
        self.assertAlmostEqual(ymin, -ymax)


class TestChecks(unittest.TestCase):
    cfg = SimConfig()

    def test_plain_passes(self):
        g = plain()
        self.assertEqual(errors_only(run_drc(g, self.cfg)), [])
        self.assertEqual(errors_only(sim_sanity(g, self.cfg)), [])

    def test_sliver_fails_min_width(self):
        g = plain()
        g = Geometry(g.polygons + [rect(0.9, 1.0, 1.0, 2.0)], g.ports)
        rules = [v.rule for v in errors_only(run_drc(g, self.cfg))]
        self.assertIn("min_width", rules)

    def test_narrow_gap_fails(self):
        g = plain()
        g = Geometry(g.polygons + [rect(0.35, 1.0, 1.35, 2.0)], g.ports)  # 0.1 um from the vertical arm
        rules = [v.rule for v in errors_only(run_drc(g, self.cfg))]
        self.assertIn("min_gap", rules)

    def test_taper_at_port_fails_sanity(self):
        g = tapered(straight=0.1)
        rules = [v.rule for v in errors_only(sim_sanity(g, self.cfg))]
        self.assertIn("port_straight", rules)

    def test_asymmetric_fails_sanity(self):
        L = 5.0
        polys = [rect(-L, -0.25, L, 0.25), rect(-0.5, -L, 0.5, L)]  # vertical arm wider than horizontal
        g = Geometry(polys, [p for p in ports(L)])
        g.ports[2].width = 1.0
        g.ports[3].width = 1.0
        rules = [v.rule for v in errors_only(sim_sanity(g, self.cfg))]
        self.assertIn("symmetry", rules)

    def test_missing_port_fails(self):
        g = plain()
        g = Geometry(g.polygons, g.ports[:3])
        rules = [v.rule for v in errors_only(sim_sanity(g, self.cfg))]
        self.assertIn("ports", rules)

    def test_extent_fails(self):
        g = plain(L=25)
        rules = [v.rule for v in errors_only(run_drc(g, self.cfg))]
        self.assertIn("extent", rules)


if __name__ == "__main__":
    unittest.main()
