"""Journal: one JSON record per iteration (journal.jsonl) + a Markdown rendering.

The journal is the agent's long-term memory. It is structured so the agent can
compare *predicted* vs *measured* for every run, not just read a narrative.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional


class Journal:
    def __init__(self, exp_dir: Path):
        self.jsonl = Path(exp_dir) / "journal.jsonl"
        self.md = Path(exp_dir) / "journal.md"

    # -- storage -----------------------------------------------------------------
    def entries(self) -> List[dict]:
        if not self.jsonl.exists():
            return []
        out = []
        for line in self.jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def append(self, entry: dict) -> None:
        entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        self.render()

    # -- queries -------------------------------------------------------------------
    def scored(self) -> List[dict]:
        return [e for e in self.entries() if e.get("valid")]

    def sims_used(self) -> int:
        return sum(1 for e in self.entries() if e.get("sim_spent"))

    def best(self) -> Optional[dict]:
        kept = [e for e in self.entries() if e.get("decision") == "keep"]
        return kept[-1] if kept else None

    def last(self) -> Optional[dict]:
        es = self.entries()
        return es[-1] if es else None

    def next_iteration(self) -> int:
        es = self.entries()
        return (es[-1]["iteration"] + 1) if es else 1

    # -- rendering -------------------------------------------------------------------
    def render(self) -> None:
        es = self.entries()
        lines = ["# Experiment journal", "", f"_{len(es)} iterations, {sum(1 for e in es if e.get('sim_spent'))} simulations spent_", ""]
        lines += ["| iter | mode | type | predicted | measured | miss (sigma) | decision | note |", "|---|---|---|---|---|---|---|---|"]
        for e in es:
            p = e.get("prediction") or {}
            pred = f"{p['fom']:.4g} +- {p['sigma']:.2g}" if p.get("fom") is not None else "-"
            meas = f"{e['fom']:.4g}" if e.get("fom") is not None else "-"
            miss = f"{e['miss_sigma']:+.2f}" if e.get("miss_sigma") is not None else "-"
            note = (e.get("label") or e.get("reason") or e.get("change") or "").replace("\n", " ")[:80]
            lines.append(f"| {e['iteration']} | {e.get('mode','')} | {e.get('change_type','')} | {pred} | {meas} | {miss} | {e.get('decision','')} | {note} |")
        lines.append("")
        for e in es:
            lines += [f"## Iteration {e['iteration']} - {e.get('mode','')} - {e.get('decision','')}", ""]
            if e.get("hypothesis"):
                lines += [f"**Hypothesis.** {e['hypothesis']}", ""]
            if e.get("change"):
                lines += [f"**Change.** {e['change']}", ""]
            p = e.get("prediction") or {}
            if p.get("fom") is not None:
                lines += [f"**Predicted.** FOM {p['fom']:.3f} +- {p['sigma']:.3f} dB. {p.get('notes','')}", ""]
            if e.get("summary"):
                lines += [f"**Measured.** {e.get('summary_text') or e['summary']}", ""]
            if e.get("breakdown_table"):
                lines += ["```", e["breakdown_table"], "```", ""]
            if e.get("miss_sigma") is not None:
                lines += [f"**Prediction miss.** {e['miss_sigma']:+.2f} sigma" + (" -> model revision required" if e.get("revise_model_next") else ""), ""]
            if e.get("reason"):
                lines += [f"**Outcome.** {e['reason']}", ""]
            if e.get("checks"):
                lines += ["Checks: " + "; ".join(e["checks"]), ""]
            if e.get("laws_added"):
                lines += ["Laws added: " + " | ".join(e["laws_added"]), ""]
        self.md.write_text("\n".join(lines), encoding="utf-8")

    # -- agent-facing digest -------------------------------------------------------------
    def digest(self, last_n: int = 8) -> str:
        """Compact text the agent reads each turn: best so far + recent iterations."""
        es = self.entries()
        if not es:
            return "(empty journal - this is the first iteration)"
        out = []
        b = self.best()
        if b:
            out.append(f"BEST so far: iter {b['iteration']} FOM {b['fom']:.4g} ({b.get('summary_text','')}) - design in best/design.py")
        else:
            out.append("BEST so far: none kept yet")
        out.append(f"simulations spent: {self.sims_used()}")
        out.append("")
        for e in es[-last_n:]:
            p = e.get("prediction") or {}
            pred = f"pred {p['fom']:.4g}+-{p['sigma']:.2g}" if p.get("fom") is not None else "pred -"
            if e.get("mode") == "model":
                meas = f"MODEL UPDATED: {(e.get('reason') or '')[:160]}" if e.get("decision") == "model updated" else f"model turn failed ({e.get('stage','')}): {(e.get('reason') or '')[:160]}"
            elif e.get("valid"):
                meas = f"meas FOM {e['fom']:.4g} ({e.get('summary_text','')})" + (f" miss {e['miss_sigma']:+.2f}s" if e.get("miss_sigma") is not None else "")
            else:
                meas = f"INVALID ({e.get('stage','')}): {(e.get('reason') or '')[:160]}"
            out.append(f"[{e['iteration']}] {e.get('mode','')}: {e.get('change','')[:140]}")
            out.append(f"     {pred}; {meas}; decision={e.get('decision','')}")
        return "\n".join(out)
