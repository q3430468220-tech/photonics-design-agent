"""LOCKED settings for the PN microring modulator experiment (family pn_ring_modulator)."""
from prism_min.config import ActiveConfig, RingConfig, SimConfig

CONFIG = SimConfig(
    device_family="pn_ring_modulator",
    active=ActiveConfig(
        wavelength=1.55,
        core_height=0.22,
        slab_height=0.09,
        v_list=[0.0, -0.5, -1.0, -1.5, -2.0, -3.0],
        v_ref=-2.0,
        min_contact_to_rib=0.6,
        min_pp_to_rib=0.4,
        max_half_width=6.0,
        max_alpha_db_cm=40.0,
        max_vpil_vcm=5.0,
        min_f_rc_ghz=20.0,
    ),
    ring=RingConfig(
        n_eff0=2.50,
        n_g=4.00,
        radius_min_um=3.0,
        radius_max_um=50.0,
        gap_min_um=0.10,
        gap_max_um=1.00,
        doped_fraction_min=0.1,
        v_high=0.0,
        v_low=-2.0,
        min_f3db_ghz=20.0,
        max_il_db=3.0,
        min_er_db=3.0,
        improve_threshold_abs=0.01,
    ),
    thickness=0.22,
    min_width=0.3,
    closure_tol=0.03,
    max_sims=40,
    max_iters=70,
    prediction_miss_sigma=2.0,
    gate_min_runs=3,
    gate_min_fraction=0.6,
    gate_window=5,
    sigma_floor_db=0.02,  # FOM is 1 - dT (dimensionless): never claim better than +-0.02 ...
    sigma_cap_rel=0.4,  # ... nor hide behind sigma > 40 % of the predicted FOM
)
