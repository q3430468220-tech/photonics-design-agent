"""LOCKED simulation / evaluation settings for the SOI crossing experiment.

The agent never edits this file. Change it by hand only, and expect every earlier
journal entry to become non-comparable when you do (the config hash is recorded
with each run).
"""
from prism_min.config import SimConfig

CONFIG = SimConfig(
    # band / platform
    wl_min=1.53,
    wl_max=1.57,
    n_wl=5,
    polarization="TE",
    thickness=0.22,
    core_index=3.476,
    clad_index=1.444,
    core_material="Si (Silicon) - Palik",
    clad_material="SiO2 (Glass) - Palik",
    # ports
    input_port="W",
    through_port="E",
    side_ports=["N", "S"],
    # FDTD (locked)
    mesh_accuracy=3,
    mesh_dx=0.02,
    mesh_dz=0.02,
    pml_margin=1.0,
    z_span=2.4,
    box_margin=0.5,
    port_offset=0.3,
    sim_time_fs=3000.0,
    auto_shutoff=1e-5,
    # DRC
    min_width=0.15,
    min_gap=0.15,
    min_area=0.02,
    raster_dx=0.01,
    max_extent=40.0,
    # sanity
    min_mesh_cells_per_feature=4,
    port_straight_len=0.5,
    require_fourfold_symmetry=True,
    # FOM / decision
    max_crosstalk_db=-30.0,
    max_reflection_db=-25.0,
    improve_threshold_db=0.01,
    closure_tol=0.02,
    # budget / gate
    max_sims=50,
    max_iters=80,
    prediction_miss_sigma=2.0,
    gate_min_runs=3,
    gate_min_fraction=0.6,
    gate_window=5,
)
