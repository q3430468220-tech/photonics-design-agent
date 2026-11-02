"""End-to-end: scripted LLM + mock backend through the full PRISM-Min loop.

Exercises: baseline -> design keep -> design that fails sanity (no sim spent, reverted)
-> design keep -> prediction miss > 2 sigma -> forced MODEL turn -> design turn with the
new model -> journal/laws/best bookkeeping.
"""
import shutil
import tempfile
import unittest
from pathlib import Path

from prism_min.agent import Agent
from prism_min.backends import get_backend
from prism_min.experiment import Experiment
from prism_min.journal import Journal
from prism_min.llm import ScriptedLLM
from tests.helpers import design_source, turn

ROOT = Path(__file__).resolve().parent.parent

TIGHT_MODEL = '''
from prism_min.geometry import Geometry
def predict(geom: Geometry) -> dict:
    # deliberately over-confident constant model: right for the plain crossing, wrong for everything else
    return {"fom": 1.1, "sigma": 0.05, "notes": "constant 1.1 dB"}
'''

FEATURE_MODEL = '''
import numpy as np
from prism_min.config import SimConfig
from prism_min.features import crossing_features
from prism_min.geometry import Geometry

CFG = SimConfig()

def predict(geom: Geometry) -> dict:
    f = crossing_features(geom, CFG)
    wc = max(f["w_c"], 0.3)
    F = f["w_j"] * (1.55 / 2.8) / wc**2
    rad = 0.2 * F**2 / (0.5 + F**2)
    xt = 0.05 * F**3 / (0.1 + F**3)
    a = f["a_max"]
    mis = 0.02 * a**2 / (2 + a**2)
    T = 1 - rad - 2 * xt - mis - 0.002
    fom = -10 * np.log10(max(T, 1e-6))
    return {"fom": float(fom), "sigma": float(max(0.02, 0.25 * fom)),
            "breakdown": {"radiation": float(rad + mis), "crosstalk": float(2 * xt)},
            "notes": "diffraction-across-gap model F = w_j*lambda_eff/w_c^2 plus taper adiabaticity"}
'''


class TestLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        for n in ("program.md", "sim_config.py", "design.py", "laws.md"):
            shutil.copy(ROOT / "experiments" / "soi_crossing" / n, self.tmp / n)
        (self.tmp / "model.py").write_text(TIGHT_MODEL)
        self.exp = Experiment(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_loop(self):
        responses = [
            # iter 2: tapered crossing -> measured ~0.36 dB vs predicted 1.1 +- 0.05 -> miss >> 2 sigma
            turn("Radiation dominates: the mode diffracts across the 0.5 um junction.",
                 "Add 3 um linear tapers to 1.5 um and a 1.5 um square junction.",
                 "design.py", design_source(L=6, wc=1.5, Lt=3), laws="- [tentative] widening the junction cuts radiation"),
            # iter 3: forced MODEL turn
            turn("The constant model ignores geometry; loss should scale with diffraction across the gap.",
                 "Replace with a feature-based diffraction model.", "model.py", FEATURE_MODEL),
            # iter 4: design with taper starting at the port -> sanity failure, no simulation
            turn("Longer taper should be more adiabatic.", "Taper from the port plane (no straight).",
                 "design.py", design_source(L=6, wc=1.5, Lt=5.9, straight=0.1)),
            # iter 5: proper wide parabolic crossing -> big improvement, kept
            turn("Both radiation and crosstalk fall with junction width; parabolic taper keeps a_max constant.",
                 "4 um parabolic tapers over 15 um.", "design.py", design_source(L=17, wc=4.0, Lt=15, shape="par"),
                 laws="- [tentative] parabolic taper keeps the local adiabaticity parameter constant"),
            # iter 6: slightly worse variant -> reverted
            turn("Try a shorter taper to save footprint.", "Same width, 8 um linear taper.",
                 "design.py", design_source(L=10, wc=4.0, Lt=8)),
        ]
        llm = ScriptedLLM(responses)
        agent = Agent(self.exp, get_backend("mock"), llm, log=lambda s: None)
        agent.run(max_iters=6)

        j = Journal(self.tmp)
        es = j.entries()
        modes = [e["mode"] for e in es]
        decisions = [e.get("decision") for e in es]
        self.assertEqual(modes[0], "baseline")
        self.assertEqual(decisions[0], "keep")
        self.assertEqual(modes[1], "design")
        self.assertEqual(decisions[1], "keep")
        self.assertTrue(es[1]["revise_model_next"], "miss > 2 sigma must trigger a model turn")
        self.assertEqual(modes[2], "model")
        self.assertEqual(decisions[2], "model updated")
        self.assertEqual(modes[3], "design")
        self.assertEqual(es[3]["stage"], "checks")
        self.assertFalse(es[3]["sim_spent"], "a sanity failure must not spend a simulation")
        self.assertEqual(decisions[3], "revert")
        self.assertEqual(decisions[4], "keep")
        self.assertLess(es[4]["fom"], 0.1)
        self.assertEqual(decisions[5], "revert")
        self.assertEqual(j.sims_used(), 4)  # baseline, iter2, iter5, iter6

        # bookkeeping
        best = j.best()
        self.assertEqual(best["iteration"], 5)
        self.assertIn("wc, Lt, straight = 17, 4.0, 15", (self.tmp / "best" / "design.py").read_text())
        self.assertIn("wc, Lt, straight = 17, 4.0, 15", (self.tmp / "design.py").read_text(), "design.py must be restored to the best after a revert")
        self.assertIn("diffraction-across-gap", (self.tmp / "model.py").read_text())
        laws = (self.tmp / "laws.md").read_text()
        self.assertIn("widening the junction cuts radiation (iter 2)", laws)
        self.assertIn("parabolic taper", laws)
        self.assertTrue((self.tmp / "journal.md").exists())
        self.assertTrue((self.tmp / "runs" / "iter_005" / "design.py").exists())
        # change-type bookkeeping and the evolution chart
        self.assertEqual(es[0]["change_type"], "baseline")
        self.assertEqual(es[1]["change_type_auto"], "topology")  # plain -> tapered adds polygons
        self.assertEqual(es[2]["change_type"], "model")
        self.assertIn(es[4]["change_type_auto"], ("shape", "parameter", "topology"))
        self.assertTrue((self.tmp / "evolution.png").exists())
        # the model turn must have seen the calibration table and the history
        model_prompt = llm.calls[1]["user"]
        self.assertIn("MODE: MODEL TURN", model_prompt)
        self.assertIn("Calibration", model_prompt)
        # the design turn after the model turn must include the latest breakdown table
        self.assertIn("wl(um)  IL(dB)", llm.calls[2]["user"])


if __name__ == "__main__":
    unittest.main()
