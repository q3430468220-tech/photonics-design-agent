"""pn_phase_shifter family: parsing, checks, toy physics trends, decomposition/closure, loop."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prism_min.agent import Agent, classify_change
from prism_min.backends import get_backend
from prism_min.config import SimConfig
from prism_min.experiment import Experiment
from prism_min.families import get_family
from prism_min.geometry import Geometry, xz_rect
from prism_min.journal import Journal
from prism_min.llm import ScriptedLLM
from tests.helpers import turn

ROOT = Path(__file__).resolve().parent.parent
CFG = SimConfig(device_family="pn_phase_shifter", closure_tol=0.03)
FAM = get_family("pn_phase_shifter")


def xsec(w=0.5, Na=5e17, Nd=5e17, xj=0.0, d_pp=0.8, N_pp=1e20, d_ct=1.2, N_mid=None, d_mid=0.4, X=3.0, slab=0.09):
    h = 0.22
    polys = [
        xz_rect(-X, X, 0, slab, "SI"), xz_rect(-w / 2, w / 2, 0, h, "SI"),
        xz_rect(-X, xj, 0, h, "P", doping=Na, dopant="p"), xz_rect(xj, X, 0, h, "N", doping=Nd, dopant="n"),
        xz_rect(-X, -w / 2 - d_pp, 0, h, "P++", doping=N_pp, dopant="p"), xz_rect(w / 2 + d_pp, X, 0, h, "N++", doping=N_pp, dopant="n"),
        xz_rect(-w / 2 - d_ct - 0.5, -w / 2 - d_ct, slab, slab + 0.3, "METAL", terminal="anode"),
        xz_rect(w / 2 + d_ct, w / 2 + d_ct + 0.5, slab, slab + 0.3, "METAL", terminal="cathode"),
    ]
    if N_mid:
        polys += [xz_rect(-X, -w / 2 - d_mid, 0, h, "P+", doping=N_mid, dopant="p"), xz_rect(w / 2 + d_mid, X, 0, h, "N+", doping=N_mid, dopant="n")]
    return Geometry(polys, [], plane="xz")


DESIGN_SRC = '''
from prism_min.geometry import Geometry, xz_rect
def build():
    w, Na, Nd, xj, d_pp, d_ct, X, h, slab = {w}, {Na}, {Nd}, {xj}, {d_pp}, {d_ct}, 3.0, 0.22, 0.09
    polys = [xz_rect(-X, X, 0, slab, "SI"), xz_rect(-w/2, w/2, 0, h, "SI"),
             xz_rect(-X, xj, 0, h, "P", doping=Na, dopant="p"), xz_rect(xj, X, 0, h, "N", doping=Nd, dopant="n"),
             xz_rect(-X, -w/2-d_pp, 0, h, "P++", doping=1e20, dopant="p"), xz_rect(w/2+d_pp, X, 0, h, "N++", doping=1e20, dopant="n"),
             xz_rect(-w/2-d_ct-0.5, -w/2-d_ct, slab, slab+0.3, "METAL", terminal="anode"),
             xz_rect(w/2+d_ct, w/2+d_ct+0.5, slab, slab+0.3, "METAL", terminal="cathode")]
    return Geometry(polys, [], thickness=h, plane="xz")
'''


def design(w=0.5, Na=5e17, Nd=5e17, xj=0.0, d_pp=0.8, d_ct=1.2):
    return DESIGN_SRC.format(w=w, Na=Na, Nd=Nd, xj=xj, d_pp=d_pp, d_ct=d_ct)


class TestPNFamily(unittest.TestCase):
    def test_features_and_checks(self):
        f = FAM.features(xsec(), CFG)
        self.assertAlmostEqual(f["w_rib"], 0.5, delta=0.01)
        self.assertAlmostEqual(f["Na_core"], 5e17)
        self.assertAlmostEqual(f["d_ct_p"], 1.2, delta=0.01)
        self.assertEqual([v.rule for v in FAM.check(xsec(), CFG) if v.severity == "error"], [])
        rules = [v.rule for v in FAM.check(xsec(d_ct=0.3), CFG)]
        self.assertIn("contact_distance", rules)
        rules = [v.rule for v in FAM.check(xsec(d_pp=0.2), CFG)]
        self.assertIn("pp_distance", rules)
        g = xsec()
        g.polygons = [p for p in g.polygons if p.props.get("terminal") != "cathode"]
        self.assertIn("contacts", [v.rule for v in FAM.check(g, CFG)])
        g = Geometry(xsec().polygons, [], plane="xy")
        self.assertIn("plane", [v.rule for v in FAM.check(g, CFG)])

    def test_toy_trends(self):
        b = get_backend("mock_pn")

        def summ(g):
            return FAM.decompose(b.run(g, CFG), CFG).summary()

        s5, s1 = summ(xsec()), summ(xsec(Na=1e18, Nd=1e18))
        self.assertLess(s1["VpiL_Vcm"], s5["VpiL_Vcm"], "more doping -> lower VpiL")
        self.assertGreater(s1["alpha_dB_cm"], s5["alpha_dB_cm"], "more doping -> more loss")
        self.assertGreater(s1["C_fF_um"], s5["C_fF_um"])
        s_close = summ(xsec(d_pp=0.5))
        self.assertGreater(s_close["alpha_dB_cm"], s5["alpha_dB_cm"], "P++ closer -> more loss")
        self.assertGreater(s_close["f_rc_GHz"], s5["f_rc_GHz"], "P++ closer -> lower R -> higher f_RC")
        self.assertTrue(1.0 < s5["VpiL_Vcm"] < 3.0 and 5 < s5["alpha_dB_cm"] < 25 and 30 < s5["f_rc_GHz"] < 200)

    def test_decompose_closure(self):
        res = get_backend("mock_pn").run(xsec(), CFG)
        bd = FAM.decompose(res, CFG)
        self.assertTrue(bd.closure_ok)
        i = int(np.argmin(np.abs(res.voltages - CFG.active.v_ref)))
        self.assertAlmostEqual(res.dneff_core[i] + res.dneff_slab[i], res.dneff_total[i], delta=0.05 * res.dneff_total[i])
        fc = FAM.fom_and_constraints(bd, CFG)
        self.assertAlmostEqual(fc["fom"], bd.summary()["FOM_VdB"])
        # table and dict round trip
        d = bd.to_dict()
        self.assertIn("V(V)", FAM.table_from_dict(d))

    def test_classify_change_doping_domain(self):
        self.assertEqual(classify_change(xsec(), xsec(Na=1e18)), "doping:parameter")
        self.assertEqual(classify_change(xsec(), xsec(N_mid=2e18)), "doping:topology")
        self.assertEqual(classify_change(xsec(), xsec(d_ct=1.0)), "electrode:parameter")
        self.assertEqual(classify_change(xsec(), xsec(w=0.45)), "geometry:parameter")


def l_junction(Na=5e17, Nd=5e17, X=3.0, w=0.5, h=0.22, slab=0.09, d_pp=0.8, d_ct=1.2, zL=0.11, xL=0.1):
    """L-shaped junction: p on the left and under the rib centre, n on the right and above it."""
    polys = [
        xz_rect(-X, X, 0, slab, "SI"), xz_rect(-w / 2, w / 2, 0, h, "SI"),
        xz_rect(-X, -xL, 0, h, "P", doping=Na, dopant="p"), xz_rect(-xL, xL, 0, zL, "P", doping=Na, dopant="p"),
        xz_rect(xL, X, 0, h, "N", doping=Nd, dopant="n"), xz_rect(-xL, xL, zL, h, "N", doping=Nd, dopant="n"),
        xz_rect(-X, -w / 2 - d_pp, 0, h, "P++", doping=1e20, dopant="p"), xz_rect(w / 2 + d_pp, X, 0, h, "N++", doping=1e20, dopant="n"),
        xz_rect(-w / 2 - d_ct - 0.5, -w / 2 - d_ct, slab, slab + 0.3, "METAL", terminal="anode"),
        xz_rect(w / 2 + d_ct, w / 2 + d_ct + 0.5, slab, slab + 0.3, "METAL", terminal="cathode"),
    ]
    return Geometry(polys, [], plane="xz")


class TestJunctionShapes(unittest.TestCase):
    def test_l_junction_trades_vpil_for_capacitance(self):
        b = get_backend("mock_pn")
        lat = FAM.decompose(b.run(xsec(), CFG), CFG).summary()
        lj = FAM.decompose(b.run(l_junction(), CFG), CFG).summary()
        self.assertLess(lj["VpiL_Vcm"], lat["VpiL_Vcm"], "L-junction: more depletion-mode overlap")
        self.assertGreater(lj["C_fF_um"], lat["C_fF_um"], "L-junction: more junction area -> more C")
        f = FAM.features(l_junction(), CFG)
        self.assertGreater(f["vertical_junction_frac"], 0.1)
        self.assertEqual([v.rule for v in FAM.check(l_junction(), CFG) if v.severity == "error"], [])
        self.assertEqual(classify_change(xsec(), l_junction()), "doping:topology")


class TestPNLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        for n in ("program.md", "sim_config.py", "design.py", "model.py", "laws.md"):
            shutil.copy(ROOT / "experiments" / "soi_pn_phase_shifter" / n, self.tmp / n)
        self.exp = Experiment(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_loop(self):
        responses = [
            turn("Loss is dominated by electrons on the n side; lower Nd and raise Na.", "Na 8e17 / Nd 3e17, junction shifted 50 nm to the n side.",
                 "design.py", design(Na=8e17, Nd=3e17, xj=0.05), change_type="doping:parameter", label="asymmetric doping p-heavy (electron FCA)"),
            turn("Contacts too close: metal loss.", "Contacts at 0.3 um.", "design.py", design(d_ct=0.3), change_type="electrode:parameter", label="contacts closer (R)"),
            turn("Trade some efficiency for loss.", "Na 4e17 / Nd 2e17.", "design.py", design(Na=4e17, Nd=2e17, xj=0.05), change_type="doping:parameter", label="lower doping"),
        ]
        llm = ScriptedLLM(responses)
        Agent(self.exp, get_backend("mock_pn"), llm, log=lambda s: None).run(max_iters=4)
        es = Journal(self.tmp).entries()
        self.assertEqual(es[0]["mode"], "baseline")
        self.assertEqual(es[0]["decision"], "keep")
        self.assertEqual(es[1]["change_type"], "doping:parameter")
        self.assertEqual(es[1]["change_type_auto"], "doping:parameter")
        self.assertTrue(es[1]["valid"])
        self.assertEqual(es[2]["stage"], "checks", "contact too close must be stopped before simulating")
        self.assertFalse(es[2]["sim_spent"])
        self.assertTrue(es[3]["valid"])
        self.assertIn("VpiL", es[3]["summary_text"])
        self.assertTrue((self.tmp / "evolution.png").exists())
        self.assertIn("pn_phase_shifter", llm.calls[0]["user"])
        self.assertIn("dneff_core", llm.calls[0]["user"])


if __name__ == "__main__":
    unittest.main()
