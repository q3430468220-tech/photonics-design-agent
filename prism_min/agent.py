"""The PRISM-Min agent loop.

    while budget:
        mode = MODEL if (last miss > 2 sigma) or (understanding gate failed) else DESIGN
        DESIGN: LLM -> new design.py -> build -> model.py prediction registered -> evaluate
                -> keep (save best) / revert (restore best) -> journal (+ laws)
        MODEL : LLM -> new model.py -> must import and predict every past scored geometry
                -> calibration recorded -> journal; no simulation spent

Everything the LLM says is recorded; everything the LLM is *allowed to do* is
enforced here, not in the prompt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import prompts
from .backends.base import Backend
from .decompose import Breakdown
from .evaluate import EvalResult, evaluate
from .experiment import DesignError, Experiment
from .features import crossing_features, format_features
from .guard import PredictionRegistry
from .journal import Journal
from .llm import LLM, LLMError

# ----------------------------------------------------------------------------
# Response parsing
# ----------------------------------------------------------------------------


@dataclass
class Turn:
    hypothesis: str
    change: str
    expected: str
    filename: str
    code: str
    laws: List[str]
    raw: str


class ParseError(ValueError):
    pass


_SECTION = re.compile(r"^#{2,4}[ \t]+([A-Z]+(?: [A-Z]+)*)(?:[ \t]*:?[ \t]*`?([\w.\-]*[a-z.][\w.\-]*)`?)?[ \t]*$", re.M)


def parse_turn(text: str, expected_file: str) -> Turn:
    """Parse the fixed heading format. Tolerant of extra prose outside the sections."""
    matches = list(_SECTION.finditer(text))
    if not matches:
        raise ParseError("no '### ' sections found")
    sections = {}
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        arg = m.group(2)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections[name] = (arg, body)
    for req in ("HYPOTHESIS", "CHANGE", "FILE"):
        if req not in sections:
            raise ParseError(f"missing section ### {req}")
    fname, fbody = sections["FILE"]
    fname = (fname or "").strip()
    if fname != expected_file:
        raise ParseError(f"expected '### FILE {expected_file}', got '### FILE {fname}'")
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", fbody, re.S)
    if not m:
        raise ParseError("no ```python code block under ### FILE")
    code = m.group(1).rstrip() + "\n"
    laws_body = sections.get("LAWS", (None, ""))[1]
    laws = []
    for line in laws_body.splitlines():
        s = line.strip()
        if s.startswith(("-", "*")) and s.lower() not in ("- none", "* none"):
            laws.append(s.lstrip("-* ").strip())
    return Turn(
        hypothesis=sections["HYPOTHESIS"][1],
        change=sections["CHANGE"][1],
        expected=sections.get("EXPECTED", (None, ""))[1],
        filename=fname,
        code=code,
        laws=laws,
        raw=text,
    )


# ----------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------


class Agent:
    def __init__(self, exp: Experiment, backend: Backend, llm: LLM, log: Callable[[str], None] = print):
        self.exp = exp
        self.backend = backend
        self.llm = llm
        self.log = log
        self.cfg = exp.config
        self.journal = Journal(exp.dir)
        self.registry = PredictionRegistry(exp.dir)

    # -- public ---------------------------------------------------------------
    def run(self, max_iters: Optional[int] = None, max_sims: Optional[int] = None) -> None:
        max_iters = max_iters or self.cfg.max_iters
        max_sims = max_sims or self.cfg.max_sims
        done = 0
        while True:
            if self.journal.sims_used() >= max_sims:
                self.log(f"budget exhausted: {self.journal.sims_used()} simulations")
                break
            if done >= max_iters:
                self.log(f"iteration cap reached: {done}")
                break
            self.step()
            done += 1

    def step(self) -> dict:
        iteration = self.journal.next_iteration()
        if not self.journal.entries():
            self.log("\n=== iteration 1 | BASELINE | measuring the initial design.py")
            return self._baseline_turn(iteration)
        mode, reason = self._decide_mode()
        self.log(f"\n=== iteration {iteration} | {mode.upper()} | {reason}")
        if mode == "model":
            return self._model_turn(iteration, reason)
        return self._design_turn(iteration, reason)

    # -- baseline ---------------------------------------------------------------
    def _baseline_turn(self, iteration: int) -> dict:
        """Measure the human-provided starting design so the agent reasons from data."""
        exp, cfg = self.exp, self.cfg
        entry = {"iteration": iteration, "mode": "baseline", "mode_reason": "initial design.py", "valid": False, "sim_spent": False,
                 "hypothesis": "baseline measurement of the starting design", "change": "none (initial design.py)"}
        try:
            geom = exp.build_design()
            exp.snapshot(iteration)
            entry["design_hash"] = geom.hash()
            entry["features"] = _safe_features(geom, cfg)
            pred = exp.predict(geom)
        except DesignError as e:
            entry.update(stage="build", reason=str(e), decision="skip")
            self.journal.append(entry)
            self.log(f"  baseline failed: {e}")
            return entry
        self.registry.register(pred, geom.hash(), iteration, source="model.py (baseline)")
        res = evaluate(exp, self.backend, iteration, geom=geom)
        entry.update(stage=res.stage, reason=res.reason, valid=res.valid, sim_spent=res.sim_spent, prediction=res.prediction,
                     checks=res.checks, fom=res.fom, constraints_ok=res.constraints_ok, constraint_violations=res.constraint_violations,
                     miss_sigma=res.miss_sigma, backend=res.backend)
        if res.breakdown:
            entry["summary"] = res.breakdown["summary"]
            entry["breakdown_table"] = _table_from_dict(res.breakdown)
        if res.valid:
            exp.save_best()
            entry["decision"] = "keep"
            if res.miss_sigma is not None and abs(res.miss_sigma) > cfg.prediction_miss_sigma:
                entry["revise_model_next"] = True
        else:
            entry["decision"] = "skip"
        self.log(f"  {res.short()}")
        self.journal.append(entry)
        return entry

    # -- mode decision --------------------------------------------------------------
    def _decide_mode(self) -> Tuple[str, str]:
        es = self.journal.entries()
        last = es[-1] if es else None
        consecutive_model = 0
        for e in reversed(es):
            if e.get("mode") == "model":
                consecutive_model += 1
            else:
                break
        if consecutive_model >= self.cfg.max_consecutive_model_turns:
            return "design", f"{consecutive_model} model turns in a row - forcing a design turn so the loop cannot stall"
        if last and last.get("revise_model_next"):
            return "model", f"iteration {last['iteration']} missed its prediction by {last['miss_sigma']:+.2f} sigma (> {self.cfg.prediction_miss_sigma})"
        if last and last.get("model_broken"):
            return "model", f"model.py failed: {last.get('reason','')[:200]}"
        cal = self.calibration()
        n = len(cal)
        if n >= self.cfg.gate_min_runs:
            frac = sum(1 for c in cal if c["within"]) / n
            if frac < self.cfg.gate_min_fraction:
                return "model", f"understanding gate: current model.py predicts only {frac:.0%} of the last {n} scored runs within 2 sigma (need {self.cfg.gate_min_fraction:.0%})"
        return "design", "proceed with design"

    # -- calibration ------------------------------------------------------------------
    def calibration(self, model_source: Optional[str] = None, window: Optional[int] = None) -> List[dict]:
        """Run the (current or candidate) model.py on the geometries of past scored runs."""
        window = window or self.cfg.gate_window
        out = []
        for e in self.journal.scored()[-window:]:
            snap = self.exp.runs / f"iter_{e['iteration']:03d}" / "design.py"
            if not snap.exists():
                continue
            try:
                geom = self.exp.build_design(snap.read_text(encoding="utf-8"))
                pred = self.exp.predict(geom, model_source)
            except DesignError as err:
                out.append({"iteration": e["iteration"], "error": str(err), "within": False})
                continue
            sigma = max(pred["sigma"], 1e-9)
            z = (e["fom"] - pred["fom"]) / sigma
            cap = max(self.cfg.sigma_floor_db, self.cfg.sigma_cap_rel * max(pred["fom"], 1e-9))
            out.append(
                {
                    "iteration": e["iteration"],
                    "measured": e["fom"],
                    "predicted": pred["fom"],
                    "sigma": pred["sigma"],
                    "z": z,
                    "sigma_ok": pred["sigma"] <= cap,
                    "within": abs(z) <= self.cfg.prediction_miss_sigma and pred["sigma"] <= cap,
                }
            )
        return out

    @staticmethod
    def _format_calibration(cal: List[dict]) -> str:
        if not cal:
            return "(no scored runs yet)"
        rows = ["iter  measured  predicted  sigma   z      ok"]
        for c in cal:
            if "error" in c:
                rows.append(f"{c['iteration']:>4}  model error: {c['error'][:80]}")
            else:
                rows.append(f"{c['iteration']:>4}  {c['measured']:8.4f}  {c['predicted']:9.4f}  {c['sigma']:6.4f}  {c['z']:+5.2f}  {'yes' if c['within'] else ('sigma too large' if not c['sigma_ok'] else 'no')}")
        return "\n".join(rows)

    # -- design turn ------------------------------------------------------------------
    def _design_turn(self, iteration: int, mode_reason: str) -> dict:
        exp, cfg = self.exp, self.cfg
        prev_design = exp.read("design.py")
        entry = {"iteration": iteration, "mode": "design", "mode_reason": mode_reason, "valid": False, "sim_spent": False}

        # 1. ask the LLM
        try:
            user = self._design_prompt(iteration, mode_reason)
            text = self.llm.complete(prompts.SYSTEM, user)
            turn = parse_turn(text, "design.py")
        except (LLMError, ParseError) as e:
            entry.update(stage="llm", reason=f"{type(e).__name__}: {e}", decision="skip")
            self.journal.append(entry)
            self.log(f"  LLM/parse failure: {e}")
            return entry
        entry.update(hypothesis=turn.hypothesis, change=turn.change, expected=turn.expected)
        self._save_raw(iteration, text)

        # 2. build candidate
        try:
            geom = exp.build_design(turn.code)
        except DesignError as e:
            entry.update(stage="build", reason=str(e), decision="revert")
            self.journal.append(entry)
            self.log(f"  build failed: {e}")
            return entry
        exp.write("design.py", turn.code)
        exp.snapshot(iteration)
        entry["design_hash"] = geom.hash()
        entry["features"] = _safe_features(geom, cfg)

        # 3. prediction from model.py
        try:
            pred = exp.predict(geom)
        except DesignError as e:
            exp.write("design.py", prev_design)
            entry.update(stage="predict", reason=f"model.py cannot predict this geometry: {e}", decision="revert", model_broken=True)
            self.journal.append(entry)
            self.log(f"  model.py failed: {e}")
            return entry
        cap = max(cfg.sigma_floor_db, cfg.sigma_cap_rel * max(pred["fom"], 1e-9))
        if pred["sigma"] > cap:
            exp.write("design.py", prev_design)
            entry.update(stage="predict", reason=f"model.py sigma {pred['sigma']:.3f} dB exceeds the cap {cap:.3f} dB (predicted FOM {pred['fom']:.3f}); tighten the model", decision="revert", model_broken=True)
            self.journal.append(entry)
            self.log("  prediction too vague")
            return entry
        self.registry.register(pred, geom.hash(), iteration)

        # 4. evaluate
        res: EvalResult = evaluate(exp, self.backend, iteration, geom=geom)
        entry.update(
            stage=res.stage,
            reason=res.reason,
            valid=res.valid,
            sim_spent=res.sim_spent,
            prediction=res.prediction,
            checks=res.checks,
            fom=res.fom,
            constraints_ok=res.constraints_ok,
            constraint_violations=res.constraint_violations,
            miss_sigma=res.miss_sigma,
            backend=res.backend,
        )
        if res.breakdown:
            entry["summary"] = res.breakdown["summary"]
            entry["breakdown_table"] = _table_from_dict(res.breakdown)
        self.log(f"  {res.short()}")

        # 5. decide
        if not res.valid:
            exp.write("design.py", prev_design)
            entry["decision"] = "revert"
        else:
            best = self.journal.best()
            if self._better(entry, best):
                exp.save_best()
                entry["decision"] = "keep"
            else:
                exp.write("design.py", prev_design)
                entry["decision"] = "revert"
            if res.miss_sigma is not None and abs(res.miss_sigma) > cfg.prediction_miss_sigma:
                entry["revise_model_next"] = True
        self.log(f"  decision: {entry['decision']}" + (" | model revision next" if entry.get("revise_model_next") else ""))

        # 6. laws
        if turn.laws:
            entry["laws_added"] = self._append_laws(turn.laws, iteration)
        self.journal.append(entry)
        return entry

    def _better(self, new: dict, best: Optional[dict]) -> bool:
        thr = self.cfg.improve_threshold_db
        if best is None:
            return True
        if new["constraints_ok"] and not best.get("constraints_ok"):
            return True
        if not new["constraints_ok"] and best.get("constraints_ok"):
            return False
        gain = new["fom"] < best["fom"] - thr
        if not new["constraints_ok"]:
            # both violate the constraints: progress on the binding constraint also counts,
            # as long as the FOM does not get worse than noise
            xt_new = new["summary"]["crosstalk_worst_dB"]
            xt_best = best["summary"]["crosstalk_worst_dB"]
            not_worse = new["fom"] <= best["fom"] + thr
            return (gain and xt_new <= xt_best + 0.5) or (xt_new < xt_best - 1.0 and not_worse)
        return gain

    # -- model turn --------------------------------------------------------------------
    def _model_turn(self, iteration: int, mode_reason: str) -> dict:
        exp = self.exp
        entry = {"iteration": iteration, "mode": "model", "mode_reason": mode_reason, "valid": False, "sim_spent": False}
        try:
            user = self._model_prompt(iteration, mode_reason)
            text = self.llm.complete(prompts.SYSTEM, user)
            turn = parse_turn(text, "model.py")
        except (LLMError, ParseError) as e:
            entry.update(stage="llm", reason=f"{type(e).__name__}: {e}", decision="skip")
            self.journal.append(entry)
            self.log(f"  LLM/parse failure: {e}")
            return entry
        entry.update(hypothesis=turn.hypothesis, change=turn.change, expected=turn.expected)
        self._save_raw(iteration, text)

        # the candidate model must import and predict the current geometry and all past scored ones
        try:
            geom = exp.build_design()
            pred_now = exp.predict(geom, turn.code)
        except DesignError as e:
            entry.update(stage="model_build", reason=str(e), decision="revert")
            self.journal.append(entry)
            self.log(f"  candidate model.py failed: {e}")
            return entry
        cal = self.calibration(model_source=turn.code)
        if any("error" in c for c in cal):
            errs = "; ".join(c["error"][:100] for c in cal if "error" in c)
            entry.update(stage="model_build", reason=f"candidate model.py fails on past geometries: {errs}", decision="revert")
            self.journal.append(entry)
            self.log("  candidate model.py fails on past geometries")
            return entry

        exp.write("model.py", turn.code)
        exp.snapshot(iteration)
        n_ok = sum(1 for c in cal if c["within"])
        entry.update(
            stage="model_updated",
            decision="model updated",
            calibration=cal,
            calibration_text=self._format_calibration(cal),
            reason=f"model.py replaced; calibration {n_ok}/{len(cal)} within 2 sigma; predicts current design {pred_now['fom']:.3f} +- {pred_now['sigma']:.3f} dB",
        )
        if turn.laws:
            entry["laws_added"] = self._append_laws(turn.laws, iteration)
        self.journal.append(entry)
        self.log(f"  {entry['reason']}")
        return entry

    # -- prompts -------------------------------------------------------------------------
    def _design_prompt(self, iteration: int, mode_reason: str) -> str:
        exp, cfg = self.exp, self.cfg
        try:
            feats = format_features(_safe_features(exp.build_design(), cfg))
        except Exception as e:  # noqa: BLE001
            feats = f"(current design.py is broken: {e})"
        return prompts.DESIGN_TURN.format(
            mode_reason=mode_reason,
            program=exp.read("program.md"),
            config_summary=cfg.summary_for_agent(),
            sims_used=self.journal.sims_used(),
            max_sims=cfg.max_sims,
            iteration=iteration,
            features=feats,
            design=exp.read("design.py"),
            model=exp.read("model.py"),
            laws=exp.read("laws.md") or "(empty)",
            journal=self.journal.digest(),
            last_detail=self._last_detail(),
        )

    def _model_prompt(self, iteration: int, mode_reason: str) -> str:
        exp, cfg = self.exp, self.cfg
        hist = []
        for e in self.journal.scored():
            f = e.get("features") or {}
            s = e["summary"]
            hist.append(
                f"[{e['iteration']}] {format_features({k: v for k, v in f.items() if isinstance(v, (int, float))})} -> IL {e['fom']:.4f} dB; rad {s['radiation_frac']:.4f}, xt {s['crosstalk_frac']:.4f}, "
                f"higher {s['through_higher_frac']:.4f}, refl {s['reflection_frac']:.4f}"
            )
        return prompts.MODEL_TURN.format(
            mode_reason=mode_reason,
            sigma_cap=f"max({cfg.sigma_floor_db}, {cfg.sigma_cap_rel} x predicted FOM)",
            program=exp.read("program.md"),
            config_summary=cfg.summary_for_agent(),
            model=exp.read("model.py"),
            calibration=self._format_calibration(self.calibration()),
            history="\n".join(hist) or "(none)",
            laws=exp.read("laws.md") or "(empty)",
            journal=self.journal.digest(),
        )

    def _last_detail(self) -> str:
        es = self.journal.entries()
        if not es:
            return "(no runs yet)"
        # the latest entry that carried a measurement or a failure worth reading
        for e in reversed(es):
            if e.get("mode") == "model":
                continue
            parts = [f"iteration {e['iteration']}: decision = {e.get('decision')}"]
            if e.get("reason"):
                parts.append(e["reason"])
            if e.get("checks"):
                parts.append("checks: " + "; ".join(e["checks"]))
            if e.get("breakdown_table"):
                parts.append(e["breakdown_table"])
            if e.get("prediction") and e["prediction"].get("fom") is not None:
                p = e["prediction"]
                parts.append(f"model.py predicted {p['fom']:.3f} +- {p['sigma']:.3f} dB; miss {e.get('miss_sigma', 0):+.2f} sigma")
            if e.get("decision") == "revert" and e.get("valid"):
                parts.append("(design.py was restored to the best kept design)")
            if e.get("decision") == "revert" and not e.get("valid"):
                parts.append("(design.py was restored to the previous version; no score)")
            return "\n".join(parts)
        return "(no design turns yet)"

    # -- misc ---------------------------------------------------------------------------------
    def _append_laws(self, laws: List[str], iteration: int) -> List[str]:
        text = self.exp.read("laws.md")
        if text and not text.endswith("\n"):
            text += "\n"
        added = []
        for law in laws:
            line = f"- {law} (iter {iteration})"
            text += line + "\n"
            added.append(law)
        self.exp.write("laws.md", text)
        return added

    def _save_raw(self, iteration: int, text: str) -> None:
        d = self.exp.run_dir(iteration)
        (d / "llm_response.md").write_text(text, encoding="utf-8")


def _safe_features(geom, cfg) -> dict:
    """Features are descriptive only; never let them crash a turn (e.g. renamed ports)."""
    try:
        return crossing_features(geom, cfg)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:120]}


def _table_from_dict(bd: dict) -> str:
    import numpy as np

    b = Breakdown(np.asarray(bd["wavelengths"]), {k: np.asarray(v) for k, v in bd["terms"].items()}, np.asarray(bd["box_total"]), bd["closure_err"], bd["closure_ok"])
    return b.table()
