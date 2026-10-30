"""Prediction guard: no simulation without a pre-registered prediction.

The agent (via model.py) must register {fom, sigma, breakdown?} for the exact
geometry hash it wants simulated. `evaluate()` consumes the registration; if it is
missing, or the design changed after registering, the run is refused. Human
validation runs bypass the guard explicitly with `--manual`, and those are
labelled as such in the journal.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


class GuardError(RuntimeError):
    pass


class PredictionRegistry:
    def __init__(self, exp_dir: Path):
        self.path = Path(exp_dir) / "runs" / "pending_prediction.json"

    def register(self, prediction: dict, design_hash: str, iteration: int, source: str = "model.py") -> dict:
        for k in ("fom", "sigma"):
            if k not in prediction:
                raise GuardError(f"prediction missing '{k}'")
        fom, sigma = float(prediction["fom"]), float(prediction["sigma"])
        if not (sigma > 0):
            raise GuardError("prediction sigma must be > 0")
        rec = {
            "fom": fom,
            "sigma": sigma,
            "breakdown": prediction.get("breakdown"),
            "notes": prediction.get("notes", ""),
            "design_hash": design_hash,
            "iteration": int(iteration),
            "source": source,
            "registered_at": time.time(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(rec, indent=2))
        return rec

    def peek(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        rec = json.loads(self.path.read_text())
        return None if rec.get("consumed") else rec

    def consume(self, design_hash: str) -> dict:
        rec = self.peek()
        if rec is None:
            raise GuardError("no registered prediction - refusing to simulate (rule: predict before you measure)")
        if rec["design_hash"] != design_hash:
            raise GuardError(f"registered prediction is for design {rec['design_hash']}, but current design is {design_hash}; re-register")
        # mark consumed in place (keeps a trace, needs no delete permission)
        self.path.write_text(json.dumps({**rec, "consumed": True, "consumed_at": time.time()}, indent=2))
        return rec

    def clear(self) -> None:
        if self.path.exists():
            self.path.write_text(json.dumps({"consumed": True, "cleared_at": time.time()}))
