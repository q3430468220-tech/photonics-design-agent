"""Offline demo: run the PRISM-Min loop with a scripted "LLM" and the mock backend.

No API key, no Lumerical. Shows what the loop does and produces the annotated
evolution chart (evolution.png): baseline -> design keep -> prediction miss ->
forced model turn -> sanity failure (no sim spent) -> keep -> revert -> ...

    python examples/demo_scripted.py            # prints the journal digest, writes demo_evolution.png next to this file
    python examples/demo_scripted.py --keep     # keep the demo experiment folder
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
from tests.helpers import design_source, turn  # noqa: E402
from tests.test_loop_mock import FEATURE_MODEL, TIGHT_MODEL  # noqa: E402

D = design_source

RESPONSES = [
    turn("Radiation dominates: the mode diffracts across the 0.5 um junction.",
         "Add 3 um linear tapers to 1.5 um and a 1.5 um square junction.",
         "design.py", D(L=6, wc=1.5, Lt=3), change_type="topology", label="add linear tapers to 1.5 um (less diffraction)",
         laws="- [tentative] widening the junction cuts radiation"),
    turn("The constant model ignores geometry; loss should scale with diffraction across the gap.",
         "Replace with a feature-based diffraction model.", "model.py", FEATURE_MODEL, change_type="model", label="diffraction model"),
    turn("Longer taper should be more adiabatic.", "Taper from the port plane (no straight).",
         "design.py", D(L=6, wc=1.5, Lt=5.9, straight=0.1), change_type="parameter", label="taper from port plane"),
    turn("Radiation still dominates; widen further.", "Junction 2.5 um, 6 um linear taper.",
         "design.py", D(L=9, wc=2.5, Lt=6), change_type="parameter", label="widen junction 1.5->2.5 um"),
    turn("Taper mismatch now comparable to radiation.", "Same width, 10 um linear taper.",
         "design.py", D(L=13, wc=2.5, Lt=10), change_type="parameter", label="lengthen taper 6->10 um (adiabatic)"),
    turn("Linear taper is steepest at the wide end; a parabolic profile keeps the adiabaticity parameter constant.",
         "Switch to parabolic taper, same width and length.",
         "design.py", D(L=13, wc=2.5, Lt=10, shape="par"), change_type="shape", label="linear->parabolic taper",
         laws="- [tentative] parabolic taper keeps the local adiabaticity parameter constant"),
    turn("Crosstalk is the binding constraint; it falls steeply with junction width.", "Junction 4 um, 15 um parabolic taper.",
         "design.py", D(L=17, wc=4.0, Lt=15, shape="par"), change_type="parameter", label="widen junction 2.5->4 um (crosstalk)"),
    turn("Try to save footprint.", "Same width, 8 um linear taper.",
         "design.py", D(L=10, wc=4.0, Lt=8), change_type="shape", label="shorter linear taper (footprint)"),
    turn("Push the junction wider still.", "4.4 um parabolic tapers over 16.8 um.",
         "design.py", D(L=20, wc=4.4, Lt=16.8, shape="par"), change_type="parameter", label="widen junction 4->4.4 um"),
    turn("Check sensitivity to taper length at the bbox limit.", "4.4 um, 15 um parabolic taper.",
         "design.py", D(L=18, wc=4.4, Lt=15, shape="par"), change_type="parameter", label="shorten taper 16.8->15 um"),
]


def main(keep: bool = False):
    tmp = Path(tempfile.mkdtemp(prefix="prism_demo_"))
    for n in ("program.md", "sim_config.py", "design.py", "laws.md"):
        shutil.copy(ROOT / "experiments" / "soi_crossing" / n, tmp / n)
    (tmp / "model.py").write_text(TIGHT_MODEL)

    exp = Experiment(tmp)
    agent = Agent(exp, get_backend("mock"), ScriptedLLM(list(RESPONSES)))
    agent.run(max_iters=len(RESPONSES) + 1)

    print("\n" + "=" * 78)
    print(Journal(tmp).digest(last_n=12))
    print("=" * 78)
    print(f"\nlaws.md:\n{(tmp / 'laws.md').read_text()}")
    out = Path(__file__).resolve().parent / "demo_evolution.png"
    if (tmp / "evolution.png").exists():
        shutil.copy(tmp / "evolution.png", out)
        print(f"evolution chart: {out}")
    if keep:
        print(f"demo experiment kept at {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main(keep="--keep" in sys.argv)
