# Task: low-loss waveguide crossing on 220 nm SOI

## Goal
Design a single-layer, 4-port waveguide crossing with **worst-case insertion loss
< 0.06 dB** over 1530-1570 nm for the fundamental TE mode, while keeping
crosstalk below -30 dB and reflection below -25 dB.

## Platform (fixed)
- 220 nm silicon on buried oxide, SiO2 top cladding (fully clad).
- Single etch layer: every polygon is full-thickness silicon. No partial etch,
  no second layer, no metals.
- Foundry rules: minimum feature width 150 nm, minimum gap 150 nm.

## Interface (fixed)
- Four ports named W, E, N, S at the bbox edges, 500 nm wide, single mode.
- Light enters at W; E is the through port; N and S are the crosstalk ports.
- Each port must be preceded by at least 1 um of uniform 500 nm waveguide.
- The device must be 4-fold symmetric about the origin (it is used in all
  directions; only the W->E path is simulated).
- Total footprint must stay inside a 40 x 40 um box.

## Figure of merit
FOM = worst-case insertion loss (dB) over the band, through port, fundamental
TE mode. Lower is better. A design is kept only if FOM improves by more than the
noise threshold AND the crosstalk / reflection constraints are satisfied.

## Budget
50 simulations. Checks (DRC, sanity) and model.py revisions are free.

## What the agent may change
Only `design.py` (geometry) in design turns and only `model.py` in model-revision
turns. Simulation settings, FOM definition and this file are locked.

## What counts as understanding
Before each simulation `model.py` must predict the FOM with an uncertainty. If
the measurement misses the prediction by more than 2 sigma, the next turn must
fix the model, not the design. Record in `laws.md` only regularities that are
supported by at least two measurements; mark anything speculative as such.

## Hints the human is willing to give
- Plain 500 nm crossings lose ~1 dB and leak ~-10 dB into the side arms because
  the mode diffracts across the unguided junction.
- The standard remedies are (a) widening the junction with tapers so the beam
  diverges less, (b) shaping the tapers so the mode stays adiabatic or
  self-images at the junction centre (MMI-type crossings).
- Think about *where* the loss goes (radiation vs. crosstalk vs. reflection vs.
  higher-order modes) before deciding what to change: the breakdown tells you.
