"""Offline demo: run the PRISM-Min loop with a scripted "LLM" and the mock backend.

No API key, no Lumerical. Shows what the loop does: baseline -> design keep ->
prediction miss -> forced model turn -> sanity failure (no sim spent) -> keep -> revert.

    python examples/demo_scripted.py            # prints the journal digest
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


def main(keep: bool = False):
    tmp = Path(tempfile.mkdtemp(prefix="prism_demo_"))
    for n in ("program.md", "sim_config.py", "design.py", "laws.md"):
        shutil.copy(ROOT / "experiments" / "soi_crossing" / n, tmp / n)
    (tmp / "model.py").write_text(TIGHT_MODEL)

    responses = [
        turn("Radiation dominates: the mode diffracts across the 0.5 um junction.",
             "Add 3 um linear tapers to 1.5 um and a 1.5 um square junction.",
             "design.py", design_source(L=6, wc=1.5, Lt=3), laws="- [tentative] widening the junction cuts radiation"),
        turn("The constant model ignores geometry; loss should scale with diffraction across the gap.",
             "Replace with a feature-based diffraction model.", "model.py", FEATURE_MODEL),
        turn("Longer taper should be more adiabatic.", "Taper from the port plane (no straight).",
             "design.py", design_source(L=6, wc=1.5, Lt=5.9, straight=0.1)),
        turn("Both radiation and crosstalk fall with junction width; parabolic taper keeps a_max constant.",
             "4 um parabolic tapers over 15 um.", "design.py", design_source(L=17, wc=4.0, Lt=15, shape="par"),
             laws="- [tentative] parabolic taper keeps the local adiabaticity parameter constant"),
        turn("Try a shorter taper to save footprint.", "Same width, 8 um linear taper.",
             "design.py", design_source(L=10, wc=4.0, Lt=8)),
        turn("Push the junction wider; crosstalk is the binding constraint.", "4.4 um parabolic tapers over 16.8 um.",
             "design.py", design_source(L=20, wc=4.4, Lt=16.8, shape="par")),
    ]
    exp = Experiment(tmp)
    agent = Agent(exp, get_backend("mock"), ScriptedLLM(responses))
    agent.run(max_iters=7)

    print("\n" + "=" * 78)
    print(Journal(tmp).digest(last_n=10))
    print("=" * 78)
    print(f"\nlaws.md:\n{(tmp / 'laws.md').read_text()}")
    if keep:
        print(f"demo experiment kept at {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main(keep="--keep" in sys.argv)
