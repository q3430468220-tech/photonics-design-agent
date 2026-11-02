# Task: lateral PN depletion phase shifter on 220 nm SOI (cross-section)

## Goal
Design the cross-section of a reverse-biased lateral PN phase shifter so that the
**efficiency-loss product VpiL x alpha at -2 V is below 20 V dB**, with the RC bandwidth
above 40 GHz. Lower is better; this is the figure of merit of a depletion modulator arm
or of the doped section of a microring modulator.

## Platform (fixed)
- 220 nm silicon rib on 2 um BOX, SiO2 cladding; partial etch leaves a 90 nm slab.
- Implants: any number of windows of type p or n with concentrations between 1e17 and
  1e20 cm^-3; windows add (net doping = donors - acceptors) as in CHARGE.
- Contacts: aluminium on silicide; the contact must sit on >= 1e19 cm^-3 silicon of its
  own type and stay >= 0.6 um from the rib sidewall. Heavy (>= 1e19) doping must stay
  >= 0.4 um from the rib sidewall.
- Cross-section half width <= 6 um. All polygons are axis-aligned rectangles
  (`prism_min.geometry.xz_rect`).

## Interface (fixed)
- `Geometry(plane="xz")`: layer `SI` for rib and slab, `P`/`N`/`P+`/`N+`/`P++`/`N++`
  for implants (props `doping` in cm^-3, `dopant` "p"/"n"), `METAL` for the two contacts
  (props `terminal` = "anode" on the p side, "cathode" on the n side).
- Anode bias list 0 ... -3 V; everything is reported at v_ref = -2 V (secant from 0 V).

## Figure of merit
FOM = VpiL(-2 V) [V cm] x alpha(-2 V) [dB/cm]. Kept only if it improves by more than 2 %
and all constraints hold: alpha <= 25 dB/cm, VpiL <= 3 V cm, f_RC = 1/(2 pi R C) >= 40 GHz.

## Budget
40 simulations (each = one CHARGE bias sweep + FDE at every bias). Checks and model.py
revisions are free.

## What the agent may change
Only `design.py` in design turns (rib width, junction position, implant windows and
concentrations, contact positions) and only `model.py` in model turns. Slab and core
thickness, wavelength, bias list, mesh and FOM definition are locked.

## What counts as understanding
Before every simulation `model.py` must predict the FOM with an uncertainty. A miss by
more than 2 sigma means the next turn fixes the model. `laws.md` only takes regularities
supported by at least two measurements.

## Hints the human is willing to give
- Efficiency comes from the depletion region sweeping through the part of the mode that
  carries most of the power: the rib centre. The junction position relative to the mode
  and the doping on both sides set how far the depletion edges move per volt.
- Loss comes from free carriers wherever the mode still has intensity: the undepleted
  core, the slab near the rib, and heavily doped regions placed too close. Holes absorb
  less per carrier than electrons at a given index change, which is why many designs
  dope the two sides differently.
- Bandwidth is RC: capacitance is set by the depletion width at the junction, resistance
  by the doping and length of the slab path to the contacts. Intermediate (P+/N+)
  windows between the core and the contacts lower R at some cost in loss.
- Read the breakdown before changing anything: it tells you whether the loss is p- or
  n-side, whether the depletion is in the rib or the slab, and whether R or C limits f_RC.
- Junction shape is a design variable: offset lateral, L-shaped (a horizontal segment under
  the rib centre plus a vertical one) and vertical junctions trade more depletion-mode
  overlap (lower VpiL) against more capacitance (lower f_RC). Draw them with several
  implant windows.
