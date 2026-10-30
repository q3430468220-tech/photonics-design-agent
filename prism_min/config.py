"""Locked simulation / evaluation configuration.

An experiment's `sim_config.py` builds one `SimConfig` and exposes it as `CONFIG`.
The agent is never asked to edit this file and the runner never writes to it.
Every run records `config_hash` so any tampering shows up in the journal.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class SimConfig:
    # ---- physics / band ----------------------------------------------------
    wl_min: float = 1.53  # um
    wl_max: float = 1.57
    n_wl: int = 5
    polarization: str = "TE"
    thickness: float = 0.22  # um device layer
    core_index: float = 3.476  # used by mock / 2D estimates
    clad_index: float = 1.444
    core_material: str = "Si (Silicon) - Palik"  # Lumerical material names
    clad_material: str = "SiO2 (Glass) - Palik"

    # ---- ports --------------------------------------------------------------
    input_port: str = "W"
    through_port: str = "E"
    side_ports: List[str] = field(default_factory=lambda: ["N", "S"])

    # ---- FDTD settings (LOCKED) -------------------------------------------
    mesh_accuracy: int = 3
    mesh_dx: float = 0.02  # um, mesh override in device region
    mesh_dz: float = 0.02
    pml_margin: float = 1.0  # um, from device bbox to simulation boundary (x, y)
    z_span: float = 2.4  # um total simulation height
    box_margin: float = 0.5  # um, enclosing flux box sits this far outside the device bbox
    port_offset: float = 0.3  # um, monitors/source this far inside the port plane
    sim_time_fs: float = 3000.0
    auto_shutoff: float = 1e-5
    mode_expansion_modes: int = 1  # fundamental only

    # ---- DRC ----------------------------------------------------------------
    min_width: float = 0.15  # um
    min_gap: float = 0.15
    min_area: float = 0.02  # um^2 (slivers)
    raster_dx: float = 0.01
    max_extent: float = 40.0  # um, bbox side limit (keeps sims affordable)

    # ---- sanity --------------------------------------------------------------
    min_mesh_cells_per_feature: int = 4
    port_straight_len: float = 0.5  # um, waveguide must be uniform this far inside each port
    require_fourfold_symmetry: bool = True  # crossing must look the same from every port
    symmetry_tol: float = 0.02  # fraction of mismatching pixels allowed

    # ---- FOM / decision -------------------------------------------------------
    fom_description: str = "worst-case insertion loss (dB) over the band, through port, fundamental mode"
    max_crosstalk_db: float = -30.0  # constraint, worst-case over band
    max_reflection_db: float = -25.0
    improve_threshold_db: float = 0.01  # 'Better?' must beat the best by more than this
    closure_tol: float = 0.02  # |box_total - 1| allowed

    # ---- budget / gate --------------------------------------------------------
    max_sims: int = 50
    max_iters: int = 80
    prediction_miss_sigma: float = 2.0  # miss > this => revise model.py next turn
    gate_min_runs: int = 3  # understanding gate starts checking after this many sims
    gate_min_fraction: float = 0.6  # fraction of recent runs model.py must predict within 2 sigma
    gate_window: int = 5
    sigma_floor_db: float = 0.02  # a prediction may never claim to be more certain than this...
    sigma_cap_rel: float = 0.5  # ...and may not hide behind sigma > cap_rel * predicted FOM
    max_consecutive_model_turns: int = 3

    def wavelengths(self):
        import numpy as np

        return np.linspace(self.wl_min, self.wl_max, self.n_wl)

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

    def summary_for_agent(self) -> str:
        """Human-readable summary shown to the agent (read-only)."""
        return (
            f"band {self.wl_min}-{self.wl_max} um ({self.n_wl} pts), {self.polarization}, "
            f"{self.thickness*1000:.0f} nm core (n={self.core_index}) in n={self.clad_index} cladding; "
            f"ports: in={self.input_port}, through={self.through_port}, side={self.side_ports}; "
            f"DRC: min width {self.min_width} um, min gap {self.min_gap} um, bbox <= {self.max_extent} um; "
            f"mesh {self.mesh_dx} um (features need >= {self.min_mesh_cells_per_feature} cells); "
            f"FOM = {self.fom_description}; constraints: crosstalk <= {self.max_crosstalk_db} dB, "
            f"reflection <= {self.max_reflection_db} dB; keep only if better by > {self.improve_threshold_db} dB; "
            f"budget {self.max_sims} sims."
        )
