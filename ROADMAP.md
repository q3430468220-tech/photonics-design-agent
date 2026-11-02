# Roadmap

The project grows in the order in which evidence can be collected: first a loop that cannot cheat, then real simulations, then the measurements that show whether the additions help, and only then more devices and smarter search. A stage is done when the journal of a real run demonstrates it, not when the code exists.

## Stage 0 — the loop (v0.1, done)

| item | content | status |
|---|---|---|
| judge | DRC + sanity checks, prediction guard, loss breakdown, measured closure, FOM + constraints, locked and hashed `sim_config.py` | v0.1 |
| agent loop | design / model turns, 2σ rule, understanding gate with σ cap, keep / revert with noise threshold, baseline measurement of the starting design | v0.1 |
| record | `journal.jsonl` / `.md`, append-only `laws.md` with iteration tags, `prism review` | v0.1 |
| change taxonomy | declared `domain:kind` (geometry / doping / stack / electrode × parameter / shape / topology) cross-checked by `classify_change()`; evolution chart | v0.1 |
| backends | `mock` (analytic crossing), `lumerical` (FDTD via lumapi, unvalidated) | v0.1 |
| tests | 27 unit and integration tests, scripted end-to-end demo | v0.1 |

## Stage 1 — real simulations

| item | content | status |
|---|---|---|
| Lumerical validation | `prism eval --manual --backend lumerical` on the plain crossing: IL ≈ 1 dB, crosstalk ≈ −10 dB, `box` ≈ 1; fix material names / result keys | next |
| convergence re-check | kept designs re-simulated with a finer mesh and a larger box every N keeps; disagreement above the noise threshold is journaled | planned |
| robustness | simulation timeout and retry, session recovery when lumapi dies, resume from an existing journal | planned |
| first real run | 50 simulations on the crossing with a real LLM; journal and evolution chart published in `docs/` | planned |

## Stage 2 — evidence

| item | content | status |
|---|---|---|
| baseline mode | `--plain` flag: guard off, gate off, breakdown hidden from the agent — the figure-style loop on the same task | planned |
| optimiser baselines | random search and CMA-ES on a fixed parametrisation (junction width, taper length, taper exponent) with the same judge and budget | planned |
| ablations | breakdown only; breakdown + prediction; full PRISM-Min; ≥ 3 seeds each; best FOM vs simulations spent | planned |
| understanding metric | distribution of prediction misses in σ over the run; fraction of laws that survive `prism review` | planned |

## Stage 3 — passive device library

| item | content | status |
|---|---|---|
| geometry helpers | arcs, Euler bends, Bézier edges, periodic arrays, boolean union, `from_gdsfactory` tested | planned |
| templates | 90° bend (vs. the Flexcompute SiN benchmark), edge-coupler taper, 1×2 MMI, directional coupler | planned |
| 2.5D screening | varFDTD backend as a cheap first pass; promote to 3D only when the coarse result improves | planned |

## Stage 4 — active devices

| item | content | status |
|---|---|---|
| CHARGE + FDE backend | doped regions from `Polygon.props`; carrier import into FDE; Δn_eff(V), Δα(V), C(V), R | planned |
| active decomposition | Δn_eff/V = ΔN(V) × mode overlap with closure; 1/Q = 1/Q_c + 1/Q_bend + 1/Q_scatter + 1/Q_FCA; ER and IL from Q_c/Q_int; 1/f² ≈ 1/f_RC² + 1/f_photon² | planned |
| templates | lateral-PN microring modulator (from the layout cookbook), TFLN MZM (VπL ∝ g / n³ r₃₃ Γ; RF loss, velocity and impedance mismatch limits) | planned |

## Stage 5 — search

| item | content | status |
|---|---|---|
| adjoint polish | gradient optimisation of continuous parameters inside a structure the agent chose; `Better?` compares polished optima | planned |
| archive | keep 3–5 structurally different candidates; branch when the best plateaus | planned |
| journal compression | every 10 iterations the agent distils the journal into lessons; raw records stay on disk | planned |
