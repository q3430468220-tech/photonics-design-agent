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
| Lumerical validation | `prism eval --manual --real` on the plain crossing (IL ≈ 1 dB, crosstalk ≈ −10 dB, `box` ≈ 1) and on the 5e17 PN shifter (VπL ≈ 2 V cm, α ≈ 10 dB/cm, perturbation terms close within 3 %); fix material names / result keys | next |
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
| geometry helpers | `shapes.py`: arc, Euler bend (clothoid), S-bend, Bézier path, path_to_polygon, annulus, periodic_array, mirrored; acute-notch DRC warning | v0.3 |
| more helpers | boolean union, `from_gdsfactory` tested | planned |
| templates | 90° bend (vs. the Flexcompute SiN benchmark), edge-coupler taper, 1×2 MMI, directional coupler | planned |
| 2.5D screening | varFDTD backend as a cheap first pass; promote to 3D only when the coarse result improves | planned |

## Stage 4 — active devices

| item | content | status |
|---|---|---|
| device-family layer | `families/`: features, rules, decomposition, FOM, summary per device kind; passive code wrapped as `passive_4port` | v0.2 |
| PN phase-shifter family | x-z cross-section IR (`xz_rect`, implant windows with concentrations, contacts with terminals); rules (contact / heavy-doping distance, ohmic contact, junction in rib); 1-D depletion + Soref toy (`mock_pn`); breakdown dneff core/slab, α p/n/contacts/background, C, R, f_RC; FOM VπL × α; template `soi_pn_phase_shifter`; demo + 6 tests | v0.2 |
| CHARGE + FDE backend | `lumerical_charge_fde.py`: CHARGE sweep with Masetti/high-field/Slotboom, charge export, FDE with np-density, undoped reference solve, python-side perturbation integrals and R × L | written, **not validated** |
| junction shapes | row-wise 2-D depletion in the toy: lateral, offset, L-shaped, vertical junctions; sub-cell edge resolution; junction-shape features | v0.3 |
| ring level | `pn_ring_modulator`: Q budget (Q_c, Q_bend, Q_fca, Q_bg, Q_contacts), shift/V, two-state all-pass spectra, swing / ER / IL, f_3dB from photon lifetime + RC, energy/bit; FOM = 1 − dT; `mock_ring`; `LumericalRing` (bent FDE for α_bend, analytic coupler); template, demo, 4 tests | v0.3 |
| coupler | FDTD / varFDTD run of the bus coupler for κ²(gap, R) replacing the analytic model | planned |
| more families | TFLN MZM (VπL ∝ g / n³ r₃₃ Γ; RF loss, velocity and impedance mismatch limits), thermo-optic shifter (HEAT + FDE), Ge photodetector (FDTD generation → CHARGE) | planned |

## Stage 5 — search

| item | content | status |
|---|---|---|
| adjoint polish | gradient optimisation of continuous parameters inside a structure the agent chose; `Better?` compares polished optima | planned |
| archive | keep 3–5 structurally different candidates; branch when the best plateaus | planned |
| journal compression | every 10 iterations the agent distils the journal into lessons; raw records stay on disk | planned |
