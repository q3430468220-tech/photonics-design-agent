"""pn_ring_modulator family: ring physics trends, Q budget closure, checks, loop."""
import shutil
import tempfile
import unittest
from pathlib import Path

from prism_min.agent import Agent
from prism_min.backends import get_backend
from prism_min.config import SimConfig
from prism_min.experiment import Experiment
from prism_min.families import get_family
from prism_min.geometry import Geometry
from prism_min.journal import Journal
from prism_min.llm import ScriptedLLM
from tests.helpers import turn
from tests.test_pn_family import DESIGN_SRC, l_junction, xsec

ROOT = Path(__file__).resolve().parent.parent
CFG = SimConfig(device_family="pn_ring_modulator", closure_tol=0.03)
FAM = get_family("pn_ring_modulator")


def ring(g, R=10.0, gap=0.2, f=0.85):
    return Geometry(g.polygons, [], plane="xz", meta={"radius_um": R, "gap_um": gap, "doped_fraction": f})


def ring_design(R=10.0, gap=0.2, f=0.85, **kw):
    src = DESIGN_SRC.format(w=kw.get("w", 0.5), Na=kw.get("Na", 5e17), Nd=kw.get("Nd", 5e17), xj=kw.get("xj", 0.0), d_pp=kw.get("d_pp", 0.8), d_ct=kw.get("d_ct", 1.2))
    return src.replace('return Geometry(polys, [], thickness=h, plane="xz")', f'return Geometry(polys, [], thickness=h, plane="xz", meta={{"radius_um": {R}, "gap_um": {gap}, "doped_fraction": {f}}})')


class TestRingFamily(unittest.TestCase):
    def summ(self, g):
        return FAM.decompose(get_backend("mock_ring").run(g, CFG), CFG).summary()

    def test_trends(self):
        base = self.summ(ring(xsec()))
        wide_gap = self.summ(ring(xsec(), gap=0.3))
        self.assertGreater(wide_gap["Q_c"], base["Q_c"], "wider gap -> less coupling -> higher Q_c")
        self.assertGreater(wide_gap["Q_L"], base["Q_L"])
        self.assertLess(wide_gap["f_photon_GHz"], base["f_photon_GHz"], "higher Q -> longer photon lifetime")
        small = self.summ(ring(xsec(), R=6))
        self.assertLess(small["Q_bend"], base["Q_bend"], "smaller radius -> more bend loss")
        self.assertGreater(small["FSR_nm"], base["FSR_nm"])
        lj = self.summ(ring(l_junction(Na=1e18, Nd=1e18), f=1.0))
        self.assertGreater(lj["shift_pm_per_V"], base["shift_pm_per_V"], "L-junction + full doping -> more shift per volt")
        self.assertTrue(0 < base["dT_max"] < 1 and base["FOM_ring"] == 1 - base["dT_max"])

    def test_q_budget_closure(self):
        bd = FAM.decompose(get_backend("mock_ring").run(ring(xsec()), CFG), CFG)
        self.assertTrue(bd.closure_ok)
        one = bd.ring["states"]["one"]
        inv = sum(1 / q for q in one["Q_i_terms"].values())
        self.assertAlmostEqual(1 / one["Q_i"], inv, delta=1e-9)
        self.assertAlmostEqual(1 / one["Q_L_model"], 1 / one["Q_i"] + 1 / bd.ring["Q_c"], delta=1e-9)
        self.assertLess(abs(one["Q_L_fit"] - one["Q_L_model"]) / one["Q_L_model"], 0.15)
        self.assertIn("Q_L(model)", FAM.table_from_dict(bd.to_dict()))

    def test_checks(self):
        self.assertEqual([v.rule for v in FAM.check(ring(xsec()), CFG) if v.severity == "error"], [])
        self.assertIn("ring_radius", [v.rule for v in FAM.check(ring(xsec(), R=1.0), CFG)])
        self.assertIn("ring_gap", [v.rule for v in FAM.check(ring(xsec(), gap=0.05), CFG)])
        self.assertIn("doped_fraction", [v.rule for v in FAM.check(ring(xsec(), f=1.5), CFG)])
        self.assertIn("contact_distance", [v.rule for v in FAM.check(ring(xsec(d_ct=0.3)), CFG)])


class TestRingLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        for n in ("program.md", "sim_config.py", "design.py", "model.py", "laws.md"):
            shutil.copy(ROOT / "experiments" / "soi_pn_ring_modulator" / n, self.tmp / n)
        self.exp = Experiment(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_loop(self):
        responses = [
            turn("Q_L 12k caps f_photon at 16 GHz; coupling is the loss-free way to lower Q.", "Gap 0.20 -> 0.15 um.",
                 "design.py", ring_design(gap=0.15), change_type="geometry:parameter", label="gap 0.20->0.15 (lower Q_L)"),
            turn("Shift per volt is the limit now.", "Core doping 1e18, doped fraction 1.0.",
                 "design.py", ring_design(gap=0.15, f=1.0, Na=1e18, Nd=1e18), change_type="doping:parameter", label="1e18, f=1.0 (shift)"),
        ]
        Agent(self.exp, get_backend("mock_ring"), ScriptedLLM(responses), log=lambda s: None).run(max_iters=3)
        es = Journal(self.tmp).entries()
        self.assertEqual(es[0]["mode"], "baseline")
        self.assertFalse(es[0]["constraints_ok"])
        self.assertTrue(es[1]["valid"])
        self.assertIn("dT", es[1]["summary_text"])
        self.assertIn(es[1]["decision"], ("keep", "revert"))
        self.assertTrue((self.tmp / "evolution.png").exists())


if __name__ == "__main__":
    unittest.main()
