"""Offline demo for the RING level: PN microring modulator on the mock_ring backend.

No API key, no Lumerical. A scripted "LLM" works through the ring design tension:
Q_L vs bandwidth (coupling), shift per volt (doping, doped fraction, L-junction), bend loss.

    python examples/demo_ring_scripted.py     # writes examples/demo_ring_evolution.png
"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prism_min.agent import Agent  # noqa: E402
from prism_min.backends import get_backend  # noqa: E402
from prism_min.experiment import Experiment  # noqa: E402
from prism_min.journal import Journal  # noqa: E402
from prism_min.llm import LLM  # noqa: E402
from tests.helpers import turn  # noqa: E402


class ModeAwareScripted(LLM):
    """Replays design responses in order; answers any MODEL turn with the lumped model
    (first time) or a re-fit with a wider sigma (later), so the demo cannot stall."""

    name = "scripted"

    def __init__(self, designs, model_src):
        self.designs = list(designs)
        self.model_src = model_src
        self.model_calls = 0

    def complete(self, system, user, max_tokens=4000, temperature=0.3):
        if "MODE: MODEL TURN" in user:
            self.model_calls += 1
            src = self.model_src if self.model_calls == 1 else self.model_src.replace("0.2 * fom", f"{0.2 + 0.1 * self.model_calls:.1f} * fom")
            return turn("The prior / previous model cannot follow the last change.", "Lumped model: Q budget + shift/linewidth swing (re-fit).",
                        "model.py", src, change_type="model", label="lumped ring model")
        if not self.designs:
            from prism_min.llm import LLMError

            raise LLMError("demo script exhausted")  # the loop records a skip and the run() cap ends it
        return self.designs.pop(0)

DESIGN = '''
from prism_min.geometry import Geometry, xz_rect
H, SLAB, X = 0.22, 0.09, 3.0
def build():
    w, Na, Nd, R, gap, f = {w}, {Na}, {Nd}, {R}, {gap}, {f}
    xr = w / 2
    polys = [xz_rect(-X, X, 0, SLAB, "SI"), xz_rect(-xr, xr, 0, H, "SI")]
    if {L}:   # L-shaped junction: p left + under the centre, n right + above it
        polys += [xz_rect(-X, -0.1, 0, H, "P", doping=Na, dopant="p"), xz_rect(-0.1, 0.1, 0, 0.11, "P", doping=Na, dopant="p"),
                  xz_rect(0.1, X, 0, H, "N", doping=Nd, dopant="n"), xz_rect(-0.1, 0.1, 0.11, H, "N", doping=Nd, dopant="n")]
    else:
        polys += [xz_rect(-X, 0.0, 0, H, "P", doping=Na, dopant="p"), xz_rect(0.0, X, 0, H, "N", doping=Nd, dopant="n")]
    polys += [xz_rect(-X, -xr - 0.8, 0, H, "P++", doping=1e20, dopant="p"), xz_rect(xr + 0.8, X, 0, H, "N++", doping=1e20, dopant="n"),
              xz_rect(-xr - 1.7, -xr - 1.2, SLAB, SLAB + 0.3, "METAL", terminal="anode"),
              xz_rect(xr + 1.2, xr + 1.7, SLAB, SLAB + 0.3, "METAL", terminal="cathode")]
    return Geometry(polys, [], thickness=H, plane="xz", meta={{"radius_um": R, "gap_um": gap, "doped_fraction": f}})
'''

MODEL = '''
import numpy as np
from prism_min.config import SimConfig
from prism_min.families import family_for
CFG = SimConfig(device_family="pn_ring_modulator")
def predict(geom):
    f = family_for(CFG).features(geom, CFG)
    R, gap, frac = f["radius_um"], f["gap_um"], f["doped_fraction"]
    Nm = np.sqrt(max(f["Na_core"], 1e16) * max(f["Nd_core"], 1e16))
    k2 = 0.25 * np.sqrt(R / 10) * np.exp(-(gap - 0.1) / 0.07)
    Qc = 2 * np.pi * 4.0 * 2 * np.pi * R / (1.55 * k2)
    Qi = 2 * np.pi * 4.0 / (1.55 * (3000 * np.exp(-R / 1.6) + 1.5 + frac * 12 * Nm / 5e17) * np.log(10) / 10 * 1e-4)
    QL = 1 / (1 / Qc + 1 / Qi)
    shift_pm = 2.0 * 13 * (Nm / 5e17) ** 0.5 * frac * (1.6 if f["vertical_junction_frac"] > 0.1 else 1.0)
    linewidth_pm = 1.55e6 / QL
    x = shift_pm / linewidth_pm
    dT = 0.9 * x / (1 + x) * (1 - 0.5 * abs(np.log(Qc / Qi)))
    fom = float(np.clip(1 - dT, 0.05, 0.99))
    return {"fom": fom, "sigma": float(max(0.02, 0.2 * fom)), "notes": "lumped Q budget + shift/linewidth swing model"}
'''


def D(w=0.5, Na=5e17, Nd=5e17, R=10.0, gap=0.2, f=0.85, L=False):
    return DESIGN.format(w=w, Na=Na, Nd=Nd, R=R, gap=gap, f=f, L=int(L))


TIGHT_PRIOR = '''
from prism_min.geometry import Geometry
def predict(geom):
    return {"fom": 0.82, "sigma": 0.05, "notes": "over-confident prior tuned to the textbook ring (demo)"}
'''

RESPONSES = [
    turn("Q_L = 12k caps f_photon at 16 GHz; the loss-free way to lower Q is coupling.", "Gap 0.20 -> 0.15 um.",
         "design.py", D(gap=0.15), change_type="geometry:parameter", label="gap 0.20->0.15 (Q_L for bandwidth)"),
    turn("With Q_L ~ 7k the swing is shift-limited (13 pm/V).", "Core 1e18 and dope the whole ring (f = 1.0).",
         "design.py", D(gap=0.15, Na=1e18, Nd=1e18, f=1.0), change_type="doping:parameter", label="1e18, doped fraction 1.0 (shift)"),
    turn("An L-shaped junction puts more depletion where the mode is.", "L-junction, same doping.",
         "design.py", D(gap=0.15, Na=1e18, Nd=1e18, f=1.0, L=True), change_type="doping:topology", label="L-shaped junction (overlap)"),
    turn("f_3dB has margin now; trade some back for swing with a little more Q.", "Gap 0.15 -> 0.17 um.",
         "design.py", D(gap=0.17, Na=1e18, Nd=1e18, f=1.0, L=True), change_type="geometry:parameter", label="gap 0.15->0.17 (more Q_L)"),
    turn("Shift per volt still limits; FCA loss has headroom in Q_i.", "Core 1.5e18.",
         "design.py", D(gap=0.17, Na=1.5e18, Nd=1.5e18, f=1.0, L=True), change_type="doping:parameter", label="core 1e18->1.5e18 (shift)"),
    turn("Try to buy ER with a narrower coupler gap... no, with less coupling.", "Gap 0.20.",
         "design.py", D(gap=0.20, Na=1.5e18, Nd=1.5e18, f=1.0, L=True), change_type="geometry:parameter", label="gap 0.17->0.20"),
    turn("Smaller ring: larger FSR, same Q_c per unit gap.", "R 10 -> 8 um.",
         "design.py", D(R=8, gap=0.17, Na=1.5e18, Nd=1.5e18, f=1.0, L=True), change_type="geometry:parameter", label="R 10->8 um"),
]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="prism_ring_demo_"))
    for n in ("program.md", "sim_config.py", "design.py", "laws.md"):
        shutil.copy(ROOT / "experiments" / "soi_pn_ring_modulator" / n, tmp / n)
    (tmp / "model.py").write_text(TIGHT_PRIOR)
    exp = Experiment(tmp)
    llm = ModeAwareScripted(RESPONSES, MODEL)
    agent = Agent(exp, get_backend("mock_ring"), llm)
    while llm.designs:
        agent.step()
    print("\n" + "=" * 78)
    print(Journal(tmp).digest(last_n=12))
    print("=" * 78)
    out = Path(__file__).resolve().parent / "demo_ring_evolution.png"
    if (tmp / "evolution.png").exists():
        shutil.copy(tmp / "evolution.png", out)
        print(f"evolution chart: {out}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
