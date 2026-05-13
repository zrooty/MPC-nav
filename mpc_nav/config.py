"""All tunable simulation parameters in one place.

Grouped by subsystem. Values preserved verbatim from the original MPC_v3.py.
"""

# ----------------------- Simulation ----------------------------------
Ts          = 0.10          # [s] simulation step
T_end       = 150.0         # [s] total sim time
Va_ref      = 21.0          # [m/s] desired airspeed reference
tau_mu      = 0.25          # [s] effective roll time constant of LQR closed-loop
                            #     (LQR dominant pole ≈ -4 rad/s → τ=0.25s, vs LOES 0.6s)

# --------------- Longitudinal (airspeed) + throttle PI ---------------
m_aircraft  = 1.2           # [kg] mass
k_drag      = 0.08          # [N/(m/s)^2] drag coefficient aggregate
k_thrust    = 9.0           # [N] max thrust, so T = k_thrust * thr
tau_thr     = 0.30          # [s] throttle first-order time constant
thr0        = 0.40          # [-] initial throttle
Kp_thr      = 2.9           # [-] PI throttle Kp on (Va_ref - V)
Ki_thr      = 0.1           # [-] PI throttle Ki on (Va_ref - V)

# --------------- Roll dynamics ---------------------------------------
a_p = -4.5
b_p = 18.0
aileron_limit_deg = 25.0

# --------------- Wind model ------------------------------------------
wind_mode      = "constant"   # "constant" | "rotating" | "gust" | "randomwalk" | "custom"
wind_mean      = (6.0, 0.0)
wind_max       = 6.0

wind_rot_deg_s = 2.0          # rotating mode: deg/s, +CCW

gust_amp       = 2.5          # gust: m/s amplitude
gust_T         = 17.0         # gust: period [s]

rw_sigma_deg_s = 15.0         # randomwalk (deterministic for reproducibility)
rw_sigma_mps_s = 0.8

# --------------- Loiter path -----------------------------------------
circle_C = (0.0, 0.0)         # center (North, East)
circle_R = 90.0               # [m] radius
cw       = True               # True: clockwise, False: counterclockwise

# --------------- Initial condition -----------------------------------
start_pos          = (-120.0, 100.0)   # [m] (North, East)
start_heading_type = "to_center"       # "to_center" | "east" | "north" | angle_rad
start_bank         = 0.0               # [rad]

# --------------- Bank limits & slew ----------------------------------
bank_limit_deg   = 35.0
slew_limit_deg_s = 80.0

# --------------- L1 parameters ---------------------------------------
L1_period  = 17.0
L1_damping = 0.75

# --------------- LTV-MPC weights -------------------------------------
w_et     = 3.0
w_echi   = 4.0
w_mu     = 0.25
w_u      = 1.0
w_et_T   = 2.0
w_echi_T = 3.0

# --------------- LTV-MPC horizon & smoothing -------------------------
N_horizon      = 15           # steps (3s lookahead; capture from echi=90° ~5s)
w_du           = 40.0         # weight on Δu
w_mu_Term      = 4.0          # terminal penalty on (mu_N - mu_ss)^2
use_cmd_filter = False        # keep False to assess controller behaviour
alpha_cmd      = 0.30

# --------------- Speed-aware Δu scaling (optional) -------------------
use_w_du_scaling = False
Va_nominal       = Va_ref

# --------------- Hybrid L1 → MPC (orange) ----------------------------
use_hybrid_l1_mpc = False
r_err_enter_frac  = 0.35      # |ρ - R| < 35%R → eligible to enter MPC
head_align_deg    = 45.0      # |χ - χ_tangent| < 45°
hold_enter_steps  = 10        # keep above conditions ~1 s (if Ts=0.1)
r_err_exit_frac   = 0.90
head_exit_deg     = 80.0
blend_seconds     = 1.0

# --------------- Plot / animation settings ---------------------------
make_animation   = True
anim_every_kstep = 2          # initial subsampling factor (may be auto-adjusted)
plane_size_m     = 12.0
anim_duration_s  = 20
save_gif         = True
save_mp4         = False
show_planes      = False
alpha_circle     = 0.50
alpha_trails     = 0.75
alpha_quiver     = 0.80
