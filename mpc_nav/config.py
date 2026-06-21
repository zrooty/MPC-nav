"""All tunable simulation parameters in one place.

Grouped by subsystem. Values preserved verbatim from the original MPC_v3.py.
"""

# ----------------------- Simulation ----------------------------------
Ts          = 0.10          # [s] outer control / sim step
inner_loop_substeps = 5     # plant roll inner loop sub-steps per Ts (dt=Ts/5=0.02s).
                            # The LQR closed-loop roll pole (~-49 rad/s) is unstable
                            # if sampled at Ts=0.1 (|s*dt|~5); sub-stepping makes the
                            # inner loop track the commanded bank without bias/ripple.
T_end       = 150.0         # [s] total sim time
Va_ref      = 21.0          # [m/s] desired airspeed reference
tau_mu      = 0.25          # [s] effective roll time constant of LQR closed-loop
                            #     (LQR dominant pole ≈ -4 rad/s → τ=0.25s, vs LOES 0.6s)

# --------------- Longitudinal (airspeed) + throttle PI ---------------
m_aircraft  = 1.2           # [kg] mass
k_drag      = 0.08          # [N/(m/s)^2] drag coefficient aggregate
k_thrust    = 60.0          # [N] max thrust, so T = k_thrust * thr.
                            #     Steady level flight: k_thrust*thr = k_drag*V^2.
                            #     Cruise at Va_ref=21 -> thr = 0.08*21^2/60 = 0.588;
                            #     full throttle tops out at sqrt(60/0.08) = 27.4 m/s.
                            #     (was 9.0, which capped airspeed at only 10.6 m/s.)
tau_thr     = 0.30          # [s] throttle first-order time constant
thr0        = 0.59          # [-] initial throttle (~cruise for Va_ref=21)
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
# Tuned via scripts/tune_l1.py sweep (T x zeta, 10–20 x 0.65–0.85):
# T=12.0, z=0.80 minimises RMS (6.61 m vs 7.09 m baseline) and steady-state
# error (-20%) at the cost of only +1.1 deg/s bank-rate roughness.
# Shorter lookahead L1_dist=(1/pi)*0.80*12*21=64m (71%R vs 96%R at T=17)
# improves constant-wind rejection without becoming too aggressive.
L1_period  = 12.0
L1_damping = 0.80

# --------------- PI loiter baseline ----------------------------------
# phi_cmd = sign_dir * (phi_ff + Kp*e_r + Ki*∫e_r dt), e_r = rho - R [m]
# NOTE: this pure-PI baseline is intentionally naive and is UNSTABLE for the
# loiter task (no cross-track-rate damping → underdamped double-integrator;
# heading-blind → no capture). See mpc_nav/pi_loiter.py and
# .docs/pi_baseline_notes.md. Kept as a documented negative baseline.
PI_Kp            = 0.020      # [rad/m]   proportional on radial error
PI_Ki            = 0.003      # [rad/(m·s)] integral on radial error
PI_centripetal_ff = True      # add atan(Vg²/(g·R)) steady-state bank feed-forward

# --------------- PID loiter baseline (functional) --------------------
# phi_cmd = sign_dir * (phi_ff + Kp*e_r + Kd*rdot + Ki*∫e_r dt)
# rdot = radial velocity (Ahat·v_g); the Kd term is the cross-track-rate
# damping that the pure PI lacks (analogue of L1's Kv). This baseline holds
# the circle. See .docs/pi_baseline_notes.md.
PID_Kp            = 0.025     # [rad/m]       swept best: Kp=0.025, Kd=0.10, Ki=0.0
PID_Kd            = 0.100     # [rad/(m/s)]  damping on radial velocity
PID_Ki            = 0.000     # [rad/(m·s)]  Ki=0 wins: centripetal FF handles SS offset
PID_centripetal_ff = True

# --------------- LTV-MPC weights -------------------------------------
w_et     = 3.0
w_echi   = 4.0
w_mu     = 0.25
w_u      = 1.0
w_et_T   = 2.0
w_echi_T = 3.0

# --------------- LTV-MPC horizon & smoothing -------------------------
N_horizon      = 40           # steps (4.0s lookahead). At Va=21 m/s, N=15 (1.5s,
                              # ~31m path) is too short to plan the turn-in to an
                              # R=90m orbit -> capture overshoots to ~96m (RMS 23m).
                              # N=40 (~84m lookahead, ~radius-scale) removes the
                              # overshoot: MAX=63.6m=initial error, RMS 6.7m (beats
                              # L1 7.1 / PID 6.8). Knee of the N-sweep; see
                              # .docs/mpc_retune_va21.md.
w_du           = 300.0        # weight on Δu (control-rate penalty). Was 40, which
                              # under-penalised slew: MPC bank-rate RMS was 28.6 deg/s
                              # (~2x rougher than L1/PID) and chattered in steady state.
                              # w_du=300 cuts bank-rate RMS to 17.9 deg/s (near PID's
                              # 15.6) with NO tracking cost (RMS 6.66->6.60). See
                              # .docs/mpc_retune_va21.md sec.9.
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
