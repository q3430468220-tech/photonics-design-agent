"""LOCKED settings for the lateral PN phase-shifter experiment (family pn_phase_shifter).

The agent never edits this file. Its hash is recorded with every run.
"""
from prism_min.config import ActiveConfig, SimConfig

CONFIG = SimConfig(
    device_family="pn_phase_shifter",
    active=ActiveConfig(
        wavelength=1.55,
        core_height=0.22,
        slab_height=0.09,
        v_list=[0.0, -0.5, -1.0, -1.5, -2.0, -3.0],
        v_ref=-2.0,
        norm_length_um=1000.0,
        charge_mesh_edge_um=0.02,
        fde_mesh_dx=0.01,
        fde_mesh_dz=0.005,
        fde_span_x=6.0,
        fde_span_z=3.0,
        min_contact_to_rib=0.6,
        min_pp_to_rib=0.4,
        max_half_width=6.0,
        max_alpha_db_cm=25.0,
        max_vpil_vcm=3.0,
        min_f_rc_ghz=40.0,
        improve_threshold_rel=0.02,
    ),
    # generic fields still used by the judge
    thickness=0.22,
    min_width=0.3,  # minimum rib width
    closure_tol=0.03,  # perturbation terms vs full solve, relative
    max_sims=40,
    max_iters=70,
    prediction_miss_sigma=2.0,
    gate_min_runs=3,
    gate_min_fraction=0.6,
    gate_window=5,
    sigma_floor_db=0.5,  # here the FOM unit is V dB; a prediction may not claim better than +-0.5 V dB ...
    sigma_cap_rel=0.4,  # ... nor hide behind sigma > 40 % of the predicted FOM
)
