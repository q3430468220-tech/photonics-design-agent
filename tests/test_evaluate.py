import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prism_min.agent import ParseError, parse_turn
from prism_min.backends import get_backend
from prism_min.config import SimConfig
from prism_min.decompose import decompose, fom_and_constraints
from prism_min.evaluate import evaluate
from prism_min.experiment import Experiment
from prism_min.guard import GuardError, PredictionRegistry
from tests.helpers import plain, tapered, turn

ROOT = Path(__file__).resolve().parent.parent


class TestDecompose(unittest.TestCase):
    cfg = SimConfig()

    def test_mock_closes_and_sums(self):
        res = get_backend("mock").run(tapered(L=12, wc=3, Lt=10), self.cfg)
        bd = decompose(res, self.cfg)
        self.assertTrue(bd.closure_ok)
        total = sum(bd.terms[k] for k in bd.terms)
        np.testing.assert_allclose(total, bd.box_total, atol=1e-9)
        s = bd.summary()
        self.assertLess(s["IL_worst_dB"], 0.25)
        self.assertGreater(s["IL_worst_dB"], 0.05)

    def test_plain_is_lossy_and_violates_crosstalk(self):
        res = get_backend("mock").run(plain(), self.cfg)
        bd = decompose(res, self.cfg)
        fc = fom_and_constraints(bd, self.cfg)
        self.assertGreater(fc["fom"], 0.8)
        self.assertFalse(fc["constraints_ok"])

    def test_unconverged_breaks_closure(self):
        res = get_backend("mock").run(tapered(straight=0.2), self.cfg)
        bd = decompose(res, self.cfg)
        self.assertFalse(bd.closure_ok)


class TestGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_refuse_consume_mismatch(self):
        reg = PredictionRegistry(self.tmp)
        with self.assertRaises(GuardError):
            reg.consume("abc")
        reg.register({"fom": 1.0, "sigma": 0.1}, "abc", 1)
        with self.assertRaises(GuardError):
            reg.consume("xyz")  # design changed after registering
        rec = reg.consume("abc")
        self.assertEqual(rec["fom"], 1.0)
        with self.assertRaises(GuardError):
            reg.consume("abc")  # already consumed
        with self.assertRaises(GuardError):
            reg.register({"fom": 1.0}, "abc", 2)  # missing sigma


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        for n in ("program.md", "sim_config.py", "design.py", "model.py", "laws.md"):
            shutil.copy(ROOT / "experiments" / "soi_crossing" / n, self.tmp / n)
        self.exp = Experiment(self.tmp)
        self.backend = get_backend("mock")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_guard_then_score(self):
        r = evaluate(self.exp, self.backend, 1)
        self.assertFalse(r.valid)
        self.assertEqual(r.stage, "guard")
        self.assertFalse(r.sim_spent)
        g = self.exp.build_design()
        PredictionRegistry(self.exp.dir).register(self.exp.predict(g), g.hash(), 1)
        r = evaluate(self.exp, self.backend, 1)
        self.assertTrue(r.valid)
        self.assertTrue(r.sim_spent)
        self.assertIsNotNone(r.miss_sigma)
        self.assertTrue((self.tmp / "runs" / "iter_001" / "eval.json").exists())

    def test_checks_block_before_sim(self):
        g = tapered(straight=0.1)
        r = evaluate(self.exp, self.backend, 2, geom=g, manual=True)
        self.assertFalse(r.valid)
        self.assertEqual(r.stage, "checks")
        self.assertFalse(r.sim_spent)

    def test_closure_voids_run(self):
        # straight long enough for the sanity check (0.5) but too short for the mock to "converge" (<0.4)
        g = tapered(straight=0.55)
        r = evaluate(self.exp, self.backend, 3, geom=g, manual=True)
        self.assertTrue(r.sim_spent)
        self.assertEqual(r.stage, "scored" if r.valid else "closure")

    def test_manual_bypass(self):
        r = evaluate(self.exp, self.backend, 4, manual=True)
        self.assertTrue(r.valid)
        self.assertIn("manual", r.prediction["source"])


class TestParse(unittest.TestCase):
    def test_parse_ok(self):
        t = parse_turn(turn("h", "c", "design.py", "x = 1", laws="- [tentative] law A\n- none"), "design.py")
        self.assertEqual(t.hypothesis, "h")
        self.assertEqual(t.code, "x = 1\n")
        self.assertEqual(t.laws, ["[tentative] law A"])

    def test_parse_wrong_file(self):
        with self.assertRaises(ParseError):
            parse_turn(turn("h", "c", "model.py", "x = 1"), "design.py")

    def test_parse_change_type_and_label(self):
        t = parse_turn(turn("h", "c", "design.py", "x = 1", change_type="topology", label="add MMI section (self-imaging)"), "design.py")
        self.assertEqual(t.change_type, "geometry:topology")  # bare kind defaults to geometry
        self.assertEqual(t.label, "add MMI section (self-imaging)")
        t = parse_turn(turn("h", "c", "design.py", "x = 1", change_type="doping:parameter"), "design.py")
        self.assertEqual(t.change_type, "doping:parameter")
        t = parse_turn(turn("h", "c", "design.py", "x = 1", change_type="Electrode / topology"), "design.py")
        self.assertEqual(t.change_type, "electrode:topology")
        t = parse_turn(turn("h", "widen the junction to 2 um", "design.py", "x = 1", change_type="nonsense", label=""), "design.py")
        self.assertEqual(t.change_type, "unspecified")
        self.assertTrue(t.label.startswith("widen the junction"))
        text = "### HYPOTHESIS\nh\n### CHANGE\nc\n### CHANGE TYPE: shape\n### FILE design.py\n```python\nx=1\n```\n"
        self.assertEqual(parse_turn(text, "design.py").change_type, "geometry:shape")

    def test_classify_change_geometry(self):
        from prism_min.agent import classify_change

        self.assertEqual(classify_change(None, plain()), "baseline")
        self.assertEqual(classify_change(plain(), plain()), "none")
        self.assertEqual(classify_change(plain(), tapered()), "geometry:topology")  # more polygons
        self.assertEqual(classify_change(tapered(L=9, wc=2.5, Lt=6), tapered(L=13, wc=2.5, Lt=10)), "geometry:parameter")
        self.assertEqual(classify_change(tapered(L=13, wc=2.5, Lt=10), tapered(L=13, wc=2.5, Lt=10, shape="par")), "geometry:shape")

    def test_classify_change_doping_and_electrode(self):
        from prism_min.agent import classify_change
        from prism_min.geometry import Geometry, rect

        g0 = plain()
        pn = [rect(-2, -1.5, 0, 1.5, "P", {"doping": 1e18, "dopant": "p"}), rect(0, -1.5, 2, 1.5, "N", {"doping": 1e18, "dopant": "n"})]
        g1 = Geometry(g0.polygons + pn, g0.ports)
        g_conc = Geometry(g0.polygons + [rect(-2, -1.5, 0, 1.5, "P", {"doping": 5e17, "dopant": "p"}), pn[1]], g0.ports)
        g_off = Geometry(g0.polygons + [rect(-2, -1.5, -0.1, 1.5, "P", {"doping": 1e18, "dopant": "p"}), pn[1]], g0.ports)
        g_pp = Geometry(g1.polygons + [rect(-4, -1.5, -2, 1.5, "P+", {"doping": 1e20, "dopant": "p"})], g0.ports)
        g_m = Geometry(g1.polygons + [rect(-5, 2, -3, 3, "M1", {"material": "Al"})], g0.ports)
        self.assertEqual(classify_change(g0, g1), "doping:topology")
        self.assertEqual(classify_change(g1, g_conc), "doping:parameter")
        self.assertEqual(classify_change(g1, g_off), "doping:parameter")
        self.assertEqual(classify_change(g1, g_pp), "doping:topology")
        self.assertEqual(classify_change(g1, g_m), "electrode:topology")
        self.assertNotEqual(g1.hash(), g_conc.hash(), "props must be part of the geometry hash")

    def test_parse_tolerant_headings(self):
        text = "Some preamble\n## HYPOTHESIS\nh\n### CHANGE: c\n### FILE: `design.py`\n```py\nx=2\n```\n"
        t = parse_turn(text, "design.py")
        self.assertEqual(t.code, "x=2\n")


if __name__ == "__main__":
    unittest.main()
