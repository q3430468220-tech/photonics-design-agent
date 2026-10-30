from pathlib import Path
import sys

# make `tests.helpers` importable when running `python -m unittest discover` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
