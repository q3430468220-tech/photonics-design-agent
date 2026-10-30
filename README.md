# PRISM-Min

**A minimal, attribution-capable LLM agent loop for photonic device design.**
Predict → simulate → decompose the loss → keep or revert → journal.
One rule: if the measurement misses the prediction by more than 2σ, the agent must fix its
*model*, not the design.

[中文说明 (README_zh.md)](README_zh.md)

---

## What it is

PRISM-Min is the "autoresearch"-style design loop (human writes `program.md`, agent edits a
Python file, a simulator judges, keep/discard) with three additions and two rules that turn a
black-box search into something whose every step can be attributed:

| | Figure-style agent loop | PRISM-Min |
|---|---|---|
| Feedback per simulation | one scalar FOM | FOM **plus a loss breakdown**: through-TE0, higher-order, crosstalk, reflection, radiation, and a measured **closure** (flux box ≈ 1) |
| Before simulating | nothing | `model.py` must **register a prediction** (FOM ± σ); the judge refuses to run without one |
| Memory | narrative journal | structured journal (predicted vs measured, breakdown, features) + `model.py` (understanding as code) + `laws.md` (lessons with evidence) |
| Rule 1 | — | prediction miss > 2σ ⇒ next turn is a **model turn** (edit `model.py` only) |
| Rule 2 | — | **understanding gate**: `model.py` must predict most recent runs within 2σ (and σ may not be inflated) before more simulations are spent |

Everything the agent is *allowed* to do is enforced by the runner, not by the prompt: it edits
exactly one file per turn, cannot touch simulation settings or the FOM, and a failed closure
voids the run instead of scoring it.

```
                 human: program.md  (goal, constraints, FOM, budget)
                        sim_config.py (LOCKED: mesh, PML, band, FOM thresholds)
                                    │
   ┌────────────────────────────────▼─────────────────────────────────┐
   │  DESIGN turn: LLM reads program + design.py + model.py + laws +   │
   │  journal digest + last breakdown  →  hypothesis, change, new      │
   │  design.py                                                        │
   │         │  build()  →  DRC + sanity  (fail ⇒ revert, no sim)      │
   │         │  model.py.predict(geom) → registered  (guard)           │
   │         ▼                                                         │
   │  SIMULATE (mock | Lumerical FDTD)  →  flux at 4 ports + box       │
   │         ▼                                                         │
   │  DECOMPOSE: TE0 / higher / crosstalk / reflection / radiation     │
   │  CLOSURE:  |box_total − 1| ≤ tol  (fail ⇒ run voided)             │
   │         ▼                                                         │
   │  Better than best by > noise threshold & constraints?  keep/revert│
   │  |miss| > 2σ ?  ⇒  next turn = MODEL turn (rewrite model.py)       │
   │  journal.jsonl / journal.md / laws.md updated                     │
   └───────────────────────────────────────────────────────────────────┘
```

## Install

```bash
git clone <this repo> && cd prism-min
pip install -e .            # numpy, matplotlib, requests only
cp .env.example .env        # fill in your LLM key (and LUMERICAL_ROOT)
```

Python ≥ 3.9. No vendor SDKs: the Anthropic and OpenAI-compatible APIs are called with
`requests`, so DeepSeek / Qwen / relay endpoints work by setting `OPENAI_BASE_URL`.

## Quick start

**1. See the loop run offline (no key, no simulator):**

```bash
python examples/demo_scripted.py
```

A scripted "LLM" drives seven iterations against the analytic mock backend: baseline →
tapered crossing (kept, but the over-confident model misses by 15σ) → forced model turn →
a design that fails the port-straight sanity check (reverted, **no simulation spent**) →
parabolic crossing (kept) → worse variant (reverted) → wider parabolic (kept).

**2. Create and validate your own experiment:**

```bash
prism init experiments/my_crossing            # copies the soi_crossing template
# edit experiments/my_crossing/program.md
prism check experiments/my_crossing --plot    # build, DRC, sanity, features, model prediction
prism eval  experiments/my_crossing --manual  # ONE human-triggered simulation; inspect the breakdown
```

Do not skip `prism eval --manual`. The whole loop is only as good as the judge: look at the
breakdown table and the closure error of your starting design before letting the agent spend
budget. With `--backend lumerical` this is also where you validate the FDTD setup.

**3. Run the agent:**

```bash
prism run experiments/my_crossing --backend mock --iters 30          # toy physics, fast
prism run experiments/my_crossing --backend lumerical --sims 50      # real FDTD (see below)
prism status experiments/my_crossing
prism review experiments/my_crossing     # fresh-session audit of laws.md vs the journal
```

## The experiment folder

| file | written by | purpose |
|---|---|---|
| `program.md` | human | goal, platform, interface, FOM, budget, hints. Read-only for the agent |
| `sim_config.py` | human | `CONFIG = SimConfig(...)`: band, mesh, PML, DRC rules, FOM thresholds, budget, gate. **Locked**; its hash is recorded with every run |
| `design.py` | agent (design turns) | `build() -> Geometry` using `prism_min.geometry` helpers |
| `model.py` | agent (model turns) | `predict(geom) -> {fom, sigma, breakdown?, notes}` — the agent's understanding, as code |
| `laws.md` | agent (append-only) | lessons tagged `[prior]` / `[tentative]` / `[confirmed]`, with iteration numbers |
| `journal.jsonl`, `journal.md` | runner | one record per iteration: hypothesis, change, prediction, measurement, breakdown, decision |
| `runs/iter_NNN/` | runner | snapshots of `design.py`/`model.py`, raw LLM response, `sim_result.json`, `eval.json`, plots |
| `best/design.py` | runner | the best kept design (also what `design.py` is restored to after a revert) |

## What one simulation returns

All quantities are fractions of the injected power, per wavelength:

```
wl(um)  IL(dB)  TE0     higher  xtalk   refl    rad     box
1.5300   1.069  0.7818  0.0001  0.0925  0.0002  0.1239  0.9986
```

`TE0` is the useful transmission (FOM = worst-case −10·log10 over the band). `higher` is power at
the through port not in the fundamental mode (output mode mismatch), `xtalk` the sum over side
ports, `refl` the backward power at the input, `rad` whatever else left the enclosing flux box.
`box` is the *measured* total outward flux; it must be 1 ± `closure_tol`, otherwise the run is
voided — a bad measurement is not a bad design.

The same decomposition idea applies to active devices with different algebra
(1/Q = 1/Q_c + 1/Q_int; VπL ∝ g/(n³ r₃₃ Γ); 1/f² ≈ 1/f_RC² + 1/f_photon²). Those backends are
not written yet.

## Decision rules (all in `SimConfig`)

- **keep** iff constraints are satisfied (or improve, when none are yet) **and** FOM improves by
  more than `improve_threshold_db` (the noise threshold); otherwise `design.py` is restored.
- **prediction miss** `|measured − predicted| / σ > prediction_miss_sigma` ⇒ next turn is a model
  turn. σ is capped at `max(sigma_floor_db, sigma_cap_rel × predicted FOM)` so the model cannot
  hide behind huge uncertainty.
- **understanding gate**: after `gate_min_runs` scored runs, the current `model.py` must predict
  ≥ `gate_min_fraction` of the last `gate_window` runs within 2σ, else a model turn is forced.
  At most `max_consecutive_model_turns` in a row, so the loop cannot stall.
- **budget**: `max_sims` simulations; checks and model turns are free.

## Backends

### `mock` — analytic toy for the crossing
Measures the geometry from a raster (junction width, through-arm width, taper adiabaticity
parameter, port straight length) and returns diffraction-, crosstalk-, mismatch- and
reflection-like terms with 2 % deterministic "mesh noise". It closes (box ≈ 1) unless the port
straights are too short, in which case it reports an unconverged run. **It is not physics** —
it exists so the loop, guard, closure and bookkeeping can be tested anywhere. Its optimum is a
~4.4 µm parabolic-taper crossing at ≈0.06 dB.

### `lumerical` — FDTD via lumapi
`prism_min/backends/lumerical.py` builds the full simulation from `SimConfig` and the
`Geometry`: extended port waveguides through the PML, mode source + reflection monitor at the
input, power + mode-expansion monitors at the other ports, a 6-face flux box for closure, a
z-normal field monitor for the plot. Set `LUMERICAL_ROOT` (folder containing `api/python/lumapi.py`,
ASCII path only).

> **Status: written against lumapi conventions but not executed where this package was
> authored.** Validate with `prism eval <dir> --backend lumerical --manual` first. The lines most
> likely to need a fix are the material names and the mode-expansion result keys.

## Repository layout

```
prism_min/
  geometry.py     Geometry IR (polygons + ports), builders, raster + morphology (numpy only)
  checks.py       DRC (min width/gap/area, ports) and sanity (mesh resolution, port straights, 4-fold symmetry)
  config.py       SimConfig (locked settings) + hash
  features.py     geometry features shared by mock, prompts and model.py
  backends/       base (SimResult), mock, lumerical
  decompose.py    loss breakdown, closure, FOM + constraints
  guard.py        PredictionRegistry: no sim without a registered prediction for this exact geometry
  evaluate.py     the judge: build → checks → guard → simulate → closure → score
  experiment.py   experiment folder, safe loading of design.py / model.py, snapshots, best/
  journal.py      JSONL + Markdown journal, agent-facing digest
  llm.py          Anthropic / OpenAI-compatible / scripted providers (requests only)
  prompts.py      system prompt, design turn, model turn, review
  agent.py        the loop: mode decision, calibration, design/model turns, keep/revert, laws
  cli.py          prism init | check | eval | run | status | review
experiments/soi_crossing/   template: 220 nm SOI crossing, IL < 0.06 dB target
examples/demo_scripted.py   offline demonstration
tests/                      24 unit + integration tests (python -m unittest discover -s tests -t .)
```

## How to evaluate whether PRISM-Min is actually better

The claim behind this repo is a hypothesis, not a result. The honest test:

1. Reproduce the plain loop (disable the prediction guard and the gate, hide the breakdown) on
   the same task and backend; this is the baseline.
2. Add one module at a time: breakdown → prediction + 2σ rule → `model.py`/gate. Run each at the
   same simulation budget with ≥ 3 seeds; plot best FOM vs simulations spent.
3. Compare against random search and a standard optimizer (CMA-ES / Bayesian) on the same
   parametrisation. Report the prediction-miss distribution as the "understanding" metric.

Everything needed for (1)–(3) is in the journal (`journal.jsonl` is machine-readable).

## Roadmap

- Validate and harden the Lumerical backend on a real crossing (field plots, convergence check
  of the kept design with a finer mesh).
- Inner adjoint / gradient polish of continuous parameters inside a structure the LLM chose.
- Multi-fidelity screening (coarse mesh / 2.5D first).
- Small archive of structurally different candidates instead of one best.
- Active-device decomposition backends (MRM: Δn_eff/V × overlap, 1/Q budget; TFLN MZM: VπL factors, bandwidth limits).

## License

MIT
