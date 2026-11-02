# Task: PN microring modulator on 220 nm SOI

## Goal
Design a depletion-mode microring modulator (lateral PN cross-section + ring radius,
coupler gap, doped fraction) that gives the **largest transmission swing between the
'one' state (0 V) and the 'zero' state (-2 V)** at the best wavelength, with an
electro-optic bandwidth of at least 20 GHz, insertion loss of the 'one' state below
3 dB and an extinction ratio above 3 dB. FOM = 1 - dT_max, lower is better.

## Platform (fixed)
Same as the phase-shifter task: 220 nm rib, 90 nm slab, SiO2 clad, implants 1e17-1e20,
contacts >= 0.6 um and heavy doping >= 0.4 um from the rib sidewall, half width <= 6 um,
axis-aligned rectangles. Group index 4.0, undoped n_eff 2.5.

## Interface (fixed)
- `Geometry(plane="xz")` as in the phase-shifter task, plus `Geometry.meta`:
  `{"radius_um": 3-50, "gap_um": 0.1-1.0, "doped_fraction": 0.1-1}`.
- The ring is a point-coupled all-pass resonator on a bus waveguide; the doped
  fraction is the part of the circumference that carries the PN junction.
- Bias list 0 ... -3 V; 'one' = 0 V, 'zero' = -2 V.

## Figure of merit and constraints
FOM = 1 - dT_max where dT = T_one - T_zero over wavelength near a resonance. Kept only
if dT improves by more than 0.01 and: f_3dB >= 20 GHz (1/f^2 = 1/f_RC^2 + 1/f_photon^2),
IL(one) <= 3 dB and ER >= 3 dB at the operating wavelength.

## Budget
40 simulations. Checks and model.py revisions are free.

## What the agent may change
`design.py`: everything in the cross-section (as in the phase-shifter task) and the
three ring parameters in `meta`. Locked: slab/core thickness, wavelength, bias list,
n_g, mesh, the coupler model and the FOM.

## Hints the human is willing to give
- The swing is set by the resonance shift (pm/V x drive) compared with the linewidth
  (lambda / Q_L) and by how close the ring sits to critical coupling. A shift of about
  one linewidth with slight under-coupling gives the largest swing.
- Q_L caps the bandwidth through the photon lifetime: f_photon = c / (lambda Q_L)
  roughly; 20 GHz needs Q_L below ~9000 at 1550 nm. Loss (bend, FCA) and coupling both
  lower Q_L - but only coupling does so without wasting light.
- The breakdown gives the Q budget (bend / FCA / background / coupling) and the shift
  per volt. If Q_bend dominates, the radius is too small; if Q_FCA dominates, the doped
  fraction or the doping is the lever; if Q_c dominates, change the gap.
- Everything from the phase-shifter task about doping, junction shape, contacts and
  R x C still applies; R x C and the photon lifetime add in quadrature.
