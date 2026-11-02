"""Offline demo for the ACTIVE family: lateral PN phase shifter on the mock_pn backend.

No API key, no Lumerical. A scripted "LLM" walks through typical modulator design moves:
asymmetric doping, intermediate implants, junction offset, contact placement, lower doping.

    python examples/demo_pn_scripted.py          # writes examples/demo_pn_evolution.png
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
from prism_min.llm import ScriptedLLM  # noqa: E402
from tests.helpers import turn  # noqa: E402

DESIGN = '''
from prism_min.geometry import Geometry, xz_rect
H, SLAB, X = 0.22, 0.09, 3.0
def build():
    w, xj = {w}, {xj}
    xr = w / 2
    polys = [xz_rect(-X, X, 0, SLAB, "SI"), xz_rect(-xr, xr, 0, H, "SI"),
             xz_rect(-X, xj, 0, H, "P", doping={Na}, dopant="p"), xz_rect(xj, X, 0, H, "N", doping={Nd}, dopant="n"),
             xz_rect(-X, -xr - {d_pp}, 0, H, "P++", doping=1e20, dopant="p"), xz_rect(xr + {d_pp}, X, 0, H, "N++", doping=1e20, dopant="n"),
             xz_rect(-xr - {d_ct} - 0.5, -xr - {d_ct}, SLAB, SLAB + 0.3, "METAL", terminal="anode"),
             xz_rect(xr + {d_ct}, xr + {d_ct} + 0.5, SLAB, SLAB + 0.3, "METAL", terminal="cathode")]
    if {N_mid}:
        polys += [xz_rect(-X, -xr - {d_mid}, 0, H, "P+", doping={N_mid}, dopant="p"), xz_rect(xr + {d_mid}, X, 0, H, "N+", doping={N_mid}, dopant="n")]
    return Geometry(polys, [], thickness=H, plane="xz")
'''

MODEL = '''
import numpy as np
from prism_min.config import SimConfig
from prism_min.families import family_for
CFG = SimConfig(device_family="pn_phase_shifter")
def predict(geom):
    f = family_for(CFG).features(geom, CFG)
    Na, Nd = max(f["Na_core"], 1e16), max(f["Nd_core"], 1e16)
    Nm = np.sqrt(Na * Nd)
    vpil = 2.0 * (5e17 / Nm) ** 0.5                       # depletion width ~ N^-1/2
    alpha = 1.5 + 4.4 * Na / 5e17 + 6.3 * Nd / 5e17        # Soref: electrons absorb more
    for side in ("p", "n"):
        d = f.get(f"d_pp_{side}", 1.0)
        if d == d: alpha += 25 * np.exp(-d / 0.15)         # heavy doping tail
    fom = vpil * alpha
    return {"fom": float(fom), "sigma": float(max(0.5, 0.15 * fom)),
            "notes": "VpiL ~ N^-1/2 scaling + linear Soref loss + exponential P++ tail"}
'''


def D(w=0.5, Na=5e17, Nd=5e17, xj=0.0, d_pp=0.8, d_ct=1.2, N_mid=0, d_mid=0.4):
    return DESIGN.format(w=w, Na=Na, Nd=Nd, xj=xj, d_pp=d_pp, d_ct=d_ct, N_mid=N_mid, d_mid=d_mid)


TIGHT_PRIOR = '''
from prism_min.geometry import Geometry
def predict(geom: Geometry) -> dict:
    return {"fom": 25.0, "sigma": 2.0, "notes": "over-confident literature prior (demo)"}
'''

RESPONSES = [
    turn("Loss (12 dB/cm) is free-carrier absorption in the undepleted core on both sides; VpiL has margin to the 3 V cm cap.",
         "Lower both core dopings 5e17 -> 3e17.", "design.py", D(Na=3e17, Nd=3e17),
         change_type="doping:parameter", label="core doping 5e17 -> 3e17 (FCA)"),
    turn("The constant prior cannot follow doping changes.", "Scaling model: VpiL ~ N^-1/2, loss linear in N per Soref, exponential P++ tail.",
         "model.py", MODEL, change_type="model", label="scaling model"),
    turn("With lighter core doping the P++/N++ tails are now a visible share of the loss.", "Move P++/N++ out to 1.2 um and the contacts to 1.6 um.",
         "design.py", D(Na=3e17, Nd=3e17, d_pp=1.2, d_ct=1.6), change_type="doping:parameter", label="P++/N++ out to 1.2 um (tail loss)"),
    turn("A narrower rib concentrates the mode on the junction, raising dneff per volt.", "Rib 0.50 -> 0.45 um.",
         "design.py", D(w=0.45, Na=3e17, Nd=3e17, d_pp=1.2, d_ct=1.6), change_type="geometry:parameter", label="rib 0.50 -> 0.45 um (overlap)"),
    turn("Holes give more index change per carrier than electrons; put more of the depletion on the p side.", "Junction +80 nm towards n.",
         "design.py", D(w=0.45, Na=3e17, Nd=3e17, xj=0.08, d_pp=1.2, d_ct=1.6), change_type="doping:parameter", label="junction +80 nm to n side"),
    turn("f_RC dropped to ~50 GHz with the far contacts; bring them in.", "Contacts at 0.5 um.",
         "design.py", D(w=0.45, Na=3e17, Nd=3e17, xj=0.08, d_pp=1.2, d_ct=0.5), change_type="electrode:parameter", label="contacts 1.6 -> 0.5 um"),
    turn("Restore f_RC with intermediate implants instead of moving metal.", "Add P+/N+ 1.5e18 at 0.5 um and lower the core to 2.5e17.",
         "design.py", D(w=0.45, Na=2.5e17, Nd=2.5e17, xj=0.08, d_pp=1.2, d_ct=1.6, N_mid=1.5e18, d_mid=0.5),
         change_type="doping:topology", label="add P+/N+ 1.5e18 implants (R), core 2.5e17"),
]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="prism_pn_demo_"))
    for n in ("program.md", "sim_config.py", "design.py", "laws.md"):
        shutil.copy(ROOT / "experiments" / "soi_pn_phase_shifter" / n, tmp / n)
    (tmp / "model.py").write_text(TIGHT_PRIOR)
    exp = Experiment(tmp)
    Agent(exp, get_backend("mock_pn"), ScriptedLLM(list(RESPONSES))).run(max_iters=len(RESPONSES) + 1)
    print("\n" + "=" * 78)
    print(Journal(tmp).digest(last_n=12))
    print("=" * 78)
    out = Path(__file__).resolve().parent / "demo_pn_evolution.png"
    if (tmp / "evolution.png").exists():
        shutil.copy(tmp / "evolution.png", out)
        print(f"evolution chart: {out}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
