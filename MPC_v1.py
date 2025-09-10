# =====================================================================

from __future__ import annotations
import os, math, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon
import scipy.sparse as sp
import osqp
from dataclasses import dataclass
from typing import Dict, Tuple, Any
from datetime import datetime

# ----------------------- USER PARAMETERS ------------------------------
# Simulation
Ts          = 0.10          # [s] simulation step
T_end       = 150.0          # [s] total sim time
Va_ref      = 21.0          # [m/s] desired airspeed reference (used by controllers)
tau_mu      = 0.6           # [s] roll (bank) time constant (LOES)

# Longitudinal (airspeed) dynamics & throttle PI
m_aircraft  = 1.2           # [kg] mass
k_drag      = 0.08          # [N/(m/s)^2] drag coefficient aggregate
k_thrust    = 9.0           # [N] max thrust, so T = k_thrust * thr
tau_thr     = 0.30          # [s] throttle first-order time constant
thr0        = 0.40          # [-] initial throttle
Kp_thr      = 2.9          # [-] PI throttle Kp on (Va_ref - V)
Ki_thr      = 0.1          # [-] PI throttle Ki on (Va_ref - V)

# --- Wind model ---
wind_mode = "gust"   # "constant" | "rotating" | "gust" | "randomwalk" | "custom"
wind_mean = (0.0, 2.0)   # mean vector for most modes
wind_max  = 12.0

# rotating
wind_rot_deg_s = 2.0     # deg/s, +CCW

# gust
gust_amp = 2.5           # m/s amplitude
gust_T   = 17.0          # s period

# randomwalk (deterministic here for reproducibility)
rw_sigma_deg_s = 15.0
rw_sigma_mps_s = 0.8

# Loiter path
circle_C    = (0.0, 0.0)    # center (North, East)
circle_R    = 90.0          # [m] radius
cw          = True          # True: clockwise, False: counterclockwise

# Initial condition
start_pos           = (-120.0, 100.0)  # [m] (North, East)
start_heading_type  = "to_center"      # "to_center" | "east" | "north" | angle_rad
start_bank          = 0.0              # [rad]

# Bank limits & slew
bank_limit_deg     = 35.0              # [deg]
slew_limit_deg_s   = 120.0             # [deg/s]

# L1 parameters
L1_period   = 17.0                     # [s]
L1_damping  = 0.75                     # [-]

# LTV-MPC weights (state & terminal)
@dataclass
class MPCWeights:
    w_et: float
    w_echi: float
    w_mu: float
    w_u: float
    w_et_T: float
    w_echi_T: float

w_et        = 30.0
w_echi      = 4.0
w_mu        = 0.25
w_u         = 10.0
w_et_T      = 20.0
w_echi_T    = 3.0

# LTV-MPC horizon & smoothing
N_horizon       = 30                   # steps
w_du            = 40.0                # weight on Δu
w_mu_Term       = 4.0                 # terminal penalty on (mu_N - mu_ss)^2
use_cmd_filter  = False               # keep False to assess controller behaviour
alpha_cmd       = 0.30

# Speed-aware Δu scaling (optional)
use_w_du_scaling = False              # if True: scale w_du with Va_model^2 / Va_nom^2
Va_nominal       = Va_ref

# ==== Hybrid L1 → MPC (orange) ====
use_hybrid_l1_mpc = True
r_err_enter_frac  = 0.35   # |ρ - R| < 35%R → eligible to enter MPC
head_align_deg    = 45.0   # |χ - χ_tangent| < 45°
hold_enter_steps  = 10     # keep above conditions for ~1 s (if Ts=0.1)
r_err_exit_frac   = 0.90
head_exit_deg     = 80.0
blend_seconds     = 1.0

# --- Plot/Animation settings ---
make_animation     = True               # enable animation
anim_every_kstep   = 2                 # initial subsampling factor (may be auto-adjusted)
plane_size_m       = 12.0
anim_duration_s    = 20
save_gif           = True
save_mp4           = False
show_planes        = False
alpha_circle       = 0.50
alpha_trails       = 0.75
alpha_quiver       = 0.80

# Output folder (timestamped)
from datetime import datetime
run_tag = datetime.now().strftime("run_%Y%m%d-%H%M%S")
os.makedirs(run_tag, exist_ok=True)

def savefig_here(name: str):
    plt.tight_layout()
    plt.savefig(os.path.join(run_tag, name), dpi=160)

def savepath(name: str) -> str:
    return os.path.join(run_tag, name)

def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))

# ----------------- Geometry / dynamics helpers -----------------------
class CirclePath:
    def __init__(self, C: Tuple[float, float], R: float, cw: bool=True):
        self.C = tuple(C); self.R = float(R); self.cw = bool(cw)
    def closest_angle(self, p: Tuple[float, float]) -> float:
        r = np.array(p, dtype=float) - np.asarray(self.C, dtype=float)
        return math.atan2(r[1], r[0])

def rk4_step_long(x: np.ndarray, mu_cmd: float, thr_cmd: float, Ts: float,
                  wind: Tuple[float,float], tau_mu: float,
                  m: float, k_d: float, k_t: float, tau_thr: float) -> np.ndarray:
    """RK4 integrator for extended state x=[n,e,chi,mu,V,thr]."""
    wn, we = wind
    g = 9.81
    def f(z):
        n,e,chi,mu,V,thr = z
        V_eff = max(1.0, float(V))  # prevent division by zero in chi_dot
        n_dot   = V*math.cos(chi) + wn
        e_dot   = V*math.sin(chi) + we
        chi_dot = (g/V_eff) * math.tan(mu)
        mu_dot  = -(mu - mu_cmd)/tau_mu
        V_dot   = (k_t*thr - k_d*V*V)/m
        thr_dot = (thr_cmd - thr)/tau_thr
        return np.array([n_dot, e_dot, chi_dot, mu_dot, V_dot, thr_dot], dtype=float)
    k1 = f(x)
    k2 = f(x + 0.5*Ts*k1)
    k3 = f(x + 0.5*Ts*k2)
    k4 = f(x + Ts*k3)
    return x + (Ts/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ---------------- L1 loiter --------------------------
class L1Loiter:
    """dir = -1 → CW (right / negative bank), dir = +1 → CCW (left / positive bank)."""
    def __init__(self, period: float, damping: float, bank_lim_deg: float, loiter_direction: int):
        self.period=float(period); self.damping=float(damping)
        self.bank_lim=float(np.radians(bank_lim_deg))
        self.dir = -1 if (loiter_direction < 0) else +1

    def command(self, x: np.ndarray, path: CirclePath, Va_for_ctrl: float, wind: Tuple[float,float]) -> float:
        # Va_for_ctrl is not used directly (we use groundspeed for geometry), but kept for API symmetry.
        n,e,chi,mu,V,thr = x; wn,we = wind
        vg = np.array([V*math.cos(chi)+wn, V*math.sin(chi)+we])
        Vg = max(1.0, np.linalg.norm(vg))

        omega = 2.0*math.pi / self.period
        Kx = omega**2
        Kv = 2.0*self.damping*omega
        K_L1 = 4.0*(self.damping**2)
        L1_dist = (1.0/math.pi) * self.damping * self.period * Vg  # use groundspeed for geometric lookahead

        A = np.array([n,e]) - np.asarray(path.C)
        r = np.linalg.norm(A) + 1e-9
        Ahat = A / r

        # capture geometry
        xtrackVelCap = Ahat[0]*vg[1] - Ahat[1]*vg[0]
        ltrackVelCap = - (Ahat @ vg)
        Nu_cap = math.atan2(xtrackVelCap, ltrackVelCap)
        Nu_cap = np.clip(Nu_cap, -math.pi/2, math.pi/2)
        latAccCap = K_L1 * Vg*Vg / max(L1_dist,1e-6) * math.sin(Nu_cap)

        # circle PD
        xtrackVelCirc = -ltrackVelCap
        xtrackErrCirc = r - path.R
        latAccCircPD = (xtrackErrCirc*Kx + xtrackVelCirc*Kv)

        # tangential velocity sign per direction
        velTangent = xtrackVelCap * (+1.0 if self.dir>0 else -1.0)

        # prevent wrong-way
        if ltrackVelCap < 0.0 and velTangent < 0.0:
            latAccCircPD = max(latAccCircPD, 0.0)

        # centripetal
        latAccCircCtr = (velTangent**2) / max(0.5*path.R, (path.R + xtrackErrCirc))
        latAccCirc = (+1.0 if self.dir>0 else -1.0) * (latAccCircPD + latAccCircCtr)

        # switch capture vs circle (capture only if outside)
        latAcc = latAccCap if (xtrackErrCirc > 0.0 and
                               (+1.0 if self.dir>0 else -1.0)*latAccCap <
                               (+1.0 if self.dir>0 else -1.0)*latAccCirc) else latAccCirc

        phi_cmd = math.atan2(latAcc, 9.81)
        return float(np.clip(phi_cmd, -self.bank_lim, self.bank_lim))

# ---------------- Wind function --------------------------------------
def wind_at(t: float, state: Tuple[float,float,float,float] | None = None) -> Tuple[float,float]:
    """Return (wn, we) at time t [s] according to global wind_* settings."""
    if wind_mode == "constant":
        return wind_mean

    if wind_mode == "rotating":
        vmag = math.hypot(*wind_mean)
        theta0 = math.atan2(wind_mean[1], wind_mean[0])
        theta  = theta0 + math.radians(wind_rot_deg_s) * t
        return (vmag*math.cos(theta), vmag*math.sin(theta))

    if wind_mode == "gust":
        base = np.array(wind_mean, float)
        vmag = np.linalg.norm(base) + gust_amp*math.sin(2*math.pi*t/gust_T)
        theta = math.atan2(base[1], base[0])
        return (vmag*math.cos(theta), vmag*math.sin(theta))

    if wind_mode == "randomwalk":
        theta0 = math.atan2(wind_mean[1], wind_mean[0])
        theta = theta0 + math.radians(rw_sigma_deg_s) * math.sin(0.7*t)
        vmag  = min(wind_max, max(0.0, np.linalg.norm(wind_mean) + rw_sigma_mps_s*math.sin(1.3*t+0.4)))
        return (vmag*math.cos(theta), vmag*math.sin(theta))

    if wind_mode == "custom":
        if state is None:
            n,e,chi,mu = (0.0,0.0,0.0,0.0)
        else:
            n,e = state[0], state[1]
        toC = np.array(circle_C) - np.array([n,e]); ang = math.atan2(toC[1], toC[0])
        vmag = min(wind_max, 6.0 + 3.0*math.sin(0.3*t))
        return (vmag*math.cos(ang), vmag*math.sin(ang))

    return wind_mean

def wind_mode_str() -> str:
    return {
        "constant":  "konstan",
        "rotating":  f"rotating (ω={wind_rot_deg_s:.1f}°/s)",
        "gust":      f"gust (A={gust_amp:.1f} m/s, T={gust_T:.0f} s)",
        "randomwalk":"random-walk",
        "custom":    "custom",
    }.get(wind_mode, str(wind_mode))

# ---------------- LTV-MPC (QP with OSQP) -----------------------------
class LTVMPC_OSQP:
    """
    MPC model uses x=[n,e,chi,mu], input u=mu_cmd with speed Va (from smoothed V_meas).
    Plant has V dynamics; throttle is handled by outer PI.
    """
    def __init__(self, Ts: float, N: int, Va_init: float, tau_mu: float,
                 bank_limit_deg: float, slew_limit_deg_s: float,
                 weights: MPCWeights, path: CirclePath,
                 w_du: float=80.0, w_mu_Term: float=4.0,
                 use_groundspeed_mu_ss: bool=True,
                 Va_min: float=6.0, Va_max: float=60.0,
                 Va_lp_tau: float=0.4, Va_nominal: float=17.0,
                 use_w_du_scaling: bool=False):
        self.Ts   = float(Ts)
        self.N    = int(N)
        self.Va_model = float(Va_init)    # smoothed for linearization
        self.Va_min = float(Va_min); self.Va_max = float(Va_max)
        self.Va_lp_tau = float(Va_lp_tau)
        self.Va_nominal = float(Va_nominal)
        self.tau  = float(tau_mu)
        self.path = path
        self.W    = weights
        self.w_du = float(w_du)
        self.use_w_du_scaling = bool(use_w_du_scaling)
        self.w_mu_Term = float(w_mu_Term)
        self.mu_max = np.radians(bank_limit_deg)
        self.du_max = np.radians(slew_limit_deg_s) * self.Ts
        self.use_groundspeed_mu_ss = bool(use_groundspeed_mu_ss)
        self.nx, self.nu = 4, 1
        self.E_mu = np.array([0.0, 0.0, 0.0, 1.0]).reshape(1,4)
        # warm-start cache
        self._z_prev = None
        self._y_prev = None
        # OSQP workspace & templates
        self._prob = None
        self._P_tpl = None; self._A_tpl = None
        self._P = None; self._A = None

    # ----- model -----
    def _f(self, x: np.ndarray, u: float, wind: Tuple[float,float]) -> np.ndarray:
        n,e,chi,mu = x; wn,we = wind; g=9.81
        Va = self.Va_model
        return np.array([
            Va*np.cos(chi)+wn,
            Va*np.sin(chi)+we,
            (g/max(1.0,Va))*np.tan(mu),
            -(mu - u)/self.tau
        ], dtype=float)

    def _rk4(self, x: np.ndarray, u: float, wind: Tuple[float,float]) -> np.ndarray:
        Ts=self.Ts
        k1=self._f(x,u,wind)
        k2=self._f(x+0.5*Ts*k1,u,wind)
        k3=self._f(x+0.5*Ts*k2,u,wind)
        k4=self._f(x+Ts*k3,u,wind)
        return x + (Ts/6.0)*(k1+2*k2+2*k3+k4)

    def _AB_discrete(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n,e,chi,mu = x
        Va=self.Va_model; g=9.81; tau=self.tau
        J = np.zeros((4,4))
        J[0,2] = -Va*np.sin(chi)
        J[1,2] =  Va*np.cos(chi)
        J[2,3] = (g/max(1.0,Va))*(1/np.cos(mu))**2
        J[3,3] = -1.0/tau
        Bc = np.zeros((4,1)); Bc[3,0] = 1.0/tau
        A = np.eye(4) + self.Ts*J
        B = self.Ts*Bc
        return A, B

    def _et_echi_and_C(self, x: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        n,e,chi,mu = x
        Cc = np.asarray(self.path.C)
        rvec = np.array([n, e]) - Cc
        r   = float(np.linalg.norm(rvec)) + 1e-9
        et0 = r - self.path.R
        alpha = math.atan2(rvec[1], rvec[0])
        chi_d = alpha - (math.pi/2) if self.path.cw else alpha + (math.pi/2)
        echi0 = wrap(chi_d - chi)
        rhat = rvec / r
        dalpha = np.array([-np.sin(alpha)/r, np.cos(alpha)/r])
        dechi  = np.zeros(4); dechi[0:2] = dalpha; dechi[2] = -1.0
        det    = np.zeros(4); det[0:2]   = rhat
        dmu    = np.array([0,0,0,1], dtype=float)
        C = np.vstack([det, dechi, dmu]).astype(float)  # 3x4
        y0 = np.array([et0, echi0, mu], dtype=float)
        return et0, echi0, C, y0

    def _mu_ss(self, x: np.ndarray, wind: Tuple[float,float]) -> float:
        Va = self.Va_model
        if self.use_groundspeed_mu_ss:
            vg = np.array([Va*np.cos(x[2])+wind[0], Va*np.sin(x[2])+wind[1]])
            Vg = max(5.0, float(np.linalg.norm(vg)))
            mu_ss_mag = math.atan2(Vg*Vg, 9.81*self.path.R)
        else:
            mu_ss_mag = math.atan2(Va*Va, 9.81*self.path.R)
        return (-mu_ss_mag) if self.path.cw else (+mu_ss_mag)

    def _gather_data_from_dense(self, M_full: np.ndarray, template_csc: sp.csc_matrix) -> np.ndarray:
        """Pick entries of dense matrix M_full in the positions of template_csc (column-wise)."""
        data = template_csc.data
        out = data.copy()
        indptr = template_csc.indptr
        indices = template_csc.indices
        for j in range(template_csc.shape[1]):
            start, end = indptr[j], indptr[j+1]
            rows = indices[start:end]
            out[start:end] = M_full[rows, j]
        return out

    # ----- solver step -----
    def step(self, x_cur: np.ndarray, wind: Tuple[float,float], u_prev: float, V_meas: float | None=None) -> Tuple[float, Dict[str, Any]]:
        # Update Va_model via low-pass of measured V
        if V_meas is not None:
            v = float(np.clip(V_meas, self.Va_min, self.Va_max))
            alpha = self.Ts / max(self.Va_lp_tau, self.Ts)
            self.Va_model = (1.0 - alpha)*self.Va_model + alpha*v

        N=self.N; nx=self.nx; nu=self.nu

        # nominal trajectory with u_nom = u_prev
        x_nom = np.zeros((N+1,nx)); x_nom[0,:] = x_cur
        for k in range(N):
            x_nom[k+1,:] = self._rk4(x_nom[k,:], u_prev, wind)

        # linearize
        Aks=[]; Bks=[]
        for k in range(N):
            A,B = self._AB_discrete(x_nom[k,:])
            Aks.append(A); Bks.append(B)

        # cost (dense)
        Wst = np.diag([self.W.w_et, self.W.w_echi, self.W.w_mu])
        WT  = np.diag([self.W.w_et_T, self.W.w_echi_T, 0.0])

        nzx = (N+1)*nx
        nzu = N*nu
        nz  = nzx + nzu
        Q = np.zeros((nz, nz), dtype=float)
        q = np.zeros(nz, dtype=float)

        def sx(k): return slice(k*nx, (k+1)*nx)
        def su(k): return slice(nzx + k*nu, nzx + (k+1)*nu)

        # stage costs
        for k in range(N):
            _, _, Ck, y0k = self._et_echi_and_C(x_nom[k,:])
            Q[sx(k), sx(k)] += Ck.T @ Wst @ Ck
            q[sx(k)]        += 2.0 * (Ck.T @ Wst @ y0k)
            Q[su(k), su(k)] += self.W.w_u * np.eye(nu)
            q[su(k)]        += 2.0 * self.W.w_u * u_prev

        # terminal costs
        _, _, CT, y0T = self._et_echi_and_C(x_nom[N,:])
        Q[sx(N), sx(N)] += CT.T @ WT @ CT
        q[sx(N)]        += 2.0 * (CT.T @ WT @ y0T)

        mu_ss = self._mu_ss(x_nom[N,:], wind)
        mu_nom_N = x_nom[N,3]
        E_mu = self.E_mu
        Q[sx(N), sx(N)] += self.w_mu_Term * (E_mu.T @ E_mu)
        q[sx(N)]        += (2.0 * self.w_mu_Term * E_mu.T * (mu_nom_N - mu_ss)).ravel()

        # Δu penalty (tri-diagonal on u-block), optionally scale with speed^2
        w_du_eff = self.w_du * ((self.Va_model/self.Va_nominal)**2 if self.use_w_du_scaling else 1.0)
        if w_du_eff > 0.0 and N>0:
            for k in range(N):
                Q[su(k), su(k)] += w_du_eff * np.eye(nu)
                if k>0:
                    Q[su(k-1), su(k-1)] += w_du_eff * np.eye(nu)
                    Q[su(k), su(k-1)]   -= w_du_eff * np.eye(nu)
                    Q[su(k-1), su(k)]   -= w_du_eff * np.eye(nu)

        # constraints: A z = b (dynamics) + bounds
        rows = []; l = []; u = []

        # initial δx_0 = 0
        A0 = sp.lil_matrix((nx, nz))
        A0[:, sx(0)] = sp.eye(nx)
        rows.append(A0); l.extend(np.zeros(nx)); u.extend(np.zeros(nx))

        # dynamics: δx_{k+1} = A_k δx_k + B_k δu_k
        for k in range(N):
            Ad, Bd = Aks[k], Bks[k]
            Am = sp.lil_matrix((nx, nz))
            Am[:, sx(k+1)] = sp.eye(nx)
            Am[:, sx(k)]   = -Ad
            Am[:, su(k)]   = -Bd
            rows.append(Am); l.extend(np.zeros(nx)); u.extend(np.zeros(nx))

        # |mu_actual_k| ≤ mu_max
        for k in range(N+1):
            Em = sp.lil_matrix((1, nz))
            Em[0, sx(k)] = E_mu
            mu_nom = x_nom[k,3]
            rows.append(Em); l.append(-self.mu_max - mu_nom); u.append(self.mu_max - mu_nom)

        # |u_nom + δu_k| ≤ mu_max
        for k in range(N):
            Um = sp.lil_matrix((1, nz)); Um[0, su(k)] = 1.0
            rows.append(Um); l.append(-self.mu_max - u_prev); u.append(self.mu_max - u_prev)

        # slew: |δu_k - δu_{k-1}| ≤ du_max
        Sm = sp.lil_matrix((1, nz)); Sm[0, su(0)] = 1.0
        rows.append(Sm); l.append(-self.du_max); u.append(self.du_max)
        for k in range(1, N):
            Sm = sp.lil_matrix((1, nz)); Sm[0, su(k)] = 1.0; Sm[0, su(k-1)] = -1.0
            rows.append(Sm); l.append(-self.du_max); u.append(self.du_max)

        A_full = sp.vstack(rows, format="csc")
        l = np.array(l, dtype=float)
        u = np.array(u, dtype=float)

        # ---------- build/update OSQP problem ----------
        P_full = sp.csc_matrix((Q + Q.T) * 0.5)
        P_full = 2.0 * P_full  # OSQP uses 0.5 z^T P z + q^T z
        P_num = P_full.toarray()
        P_num = np.triu(P_num)  # upper-triangular only

        if self._prob is None:
            # P template from nonzero pattern of first upper-tri P
            mask = (np.abs(P_num) > 1e-12).astype(float)
            self._P_tpl = sp.csc_matrix(mask)
            P_data = self._gather_data_from_dense(P_num, self._P_tpl)
            self._P = self._P_tpl.copy(); self._P.data[:] = P_data

            # A template uses exact sparsity of current A_full
            self._A_tpl = A_full.copy()
            self._A = self._A_tpl.copy()  # same structure, data will be overwritten
            self._A.data[:] = A_full.data  # copy numeric data

            self._prob = osqp.OSQP()
            self._prob.setup(P=self._P, q=q, A=self._A, l=l, u=u,
                             verbose=False, eps_abs=1e-4, eps_rel=1e-4,
                             polish=True, max_iter=20000, adaptive_rho=True)
            try:
                if self._z_prev is not None:
                    if self._y_prev is not None:
                        self._prob.warm_start(x=self._z_prev, y=self._y_prev)
                    else:
                        self._prob.warm_start(x=self._z_prev)
            except Exception:
                pass
        else:
            # Update numeric values in P and A with the template pattern kept constant
            P_data = self._gather_data_from_dense(P_num, self._P_tpl)
            self._P.data[:] = P_data
            # A: we assume structure unchanged, copy values only
            self._A.data[:] = A_full.data
            self._prob.update(Px=self._P.data, q=q, Ax=self._A.data, l=l, u=u)
            try:
                if self._z_prev is not None:
                    if self._y_prev is not None:
                        self._prob.warm_start(x=self._z_prev, y=self._y_prev)
                    else:
                        self._prob.warm_start(x=self._z_prev)
            except Exception:
                pass

        res = self._prob.solve()
        info = {"status": res.info.status, "fallback": False,
                "iter": getattr(res.info, "iter", np.nan),
                "obj": getattr(res.info, "obj_val", np.nan)}
        if res.info.status_val not in (1,2):
            # infeasible or other failure → hold previous u
            info["fallback"] = True
            return float(u_prev), info

        self._z_prev = res.x.copy()
        try:
            self._y_prev = res.y.copy()
        except Exception:
            self._y_prev = None

        z = res.x
        du0 = z[(N+1)*nx + 0] if N>0 else 0.0
        u_cmd = float(np.clip(u_prev + du0, -self.mu_max, self.mu_max))
        return u_cmd, info

# ---------------------- Utilities ------------------------------------
def make_start_state(pos, heading_type, bank0, Va_init, thr_init):
    n0, e0 = pos
    if isinstance(heading_type, (int, float)):
        chi0 = float(heading_type)
    else:
        if heading_type == "to_center":
            chi0 = math.atan2(circle_C[1]-e0, circle_C[0]-n0)
        elif heading_type == "east":
            chi0 = 0.0
        elif heading_type == "north":
            chi0 = math.pi/2.0
        else:
            raise ValueError("Unknown start_heading_type")
    V0 = float(Va_init)
    return np.array([n0, e0, chi0, bank0, V0, float(thr_init)], dtype=float)

def crosstrack_series(path: CirclePath, n_series, e_series):
    C = np.asarray(path.C)
    return np.array([np.linalg.norm(np.array([ni,ei])-C) - path.R for ni,ei in zip(n_series, e_series)])

def orbit_metrics(n, e, chi, C, R, ccw=True):
    r = np.array([n, e]) - np.asarray(C, float)
    rho = float(np.linalg.norm(r)) + 1e-9
    theta = math.atan2(r[1], r[0])
    s = +1.0 if ccw else -1.0
    chi_tan = theta + s * (math.pi/2.0)
    echi = wrap(chi_tan - chi)
    return rho, abs(rho - R), chi_tan, echi

# ====== 1) Controllers & path =========================================
def build_controllers() -> Tuple[CirclePath, L1Loiter, L1Loiter, LTVMPC_OSQP]:
    path = CirclePath(circle_C, circle_R, cw=cw)
    l1_dir = (-1 if cw else +1)
    l1_green  = L1Loiter(L1_period, L1_damping, bank_limit_deg, l1_dir)
    l1_orange = L1Loiter(L1_period, L1_damping, bank_limit_deg, l1_dir)
    Wg = MPCWeights(w_et, w_echi, w_mu, w_u, w_et_T, w_echi_T)
    mpc = LTVMPC_OSQP(Ts=Ts, N=N_horizon, Va_init=Va_ref, tau_mu=tau_mu,
                      bank_limit_deg=bank_limit_deg, slew_limit_deg_s=slew_limit_deg_s,
                      weights=Wg, path=path, w_du=w_du, w_mu_Term=w_mu_Term,
                      use_groundspeed_mu_ss=True, Va_nominal=Va_nominal,
                      use_w_du_scaling=use_w_du_scaling)
    return path, l1_green, l1_orange, mpc

# ====== 2) Simulasi plant + hybrid + logging ==========================
def simulate(path: CirclePath, l1_green: L1Loiter, l1_orange: L1Loiter, mpc: LTVMPC_OSQP
             ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any], Dict[str, list]]:
    steps = int(T_end / Ts)
    L1log  = {"n":[], "e":[], "chi":[], "mu":[], "V":[], "thr":[], "u_cmd":[]}
    MPClog = {"n":[], "e":[], "chi":[], "mu":[], "V":[], "thr":[], "u_cmd":[]}
    HYBlog = {"t":[], "mode":[], "blend":[], "rerr":[], "echi_deg":[]}
    QPlog  = {"iter":[], "obj":[], "status":[]}

    x1 = make_start_state(start_pos, start_heading_type, start_bank, Va_ref, thr0)
    x2 = x1.copy()
    u_prev = 0.0
    integ1 = 0.0; integ2 = 0.0

    mode = "CAPTURE_L1" if use_hybrid_l1_mpc else "MPC_TRACK"
    ok_cnt = 0
    blend  = 0.0
    blend_rate = Ts / max(Ts, float(blend_seconds))

    for k in range(steps):
        t = k*Ts
        wn, we = wind_at(t, state=x2); wk = (wn, we)

        # --- PI throttle (dua plant)
        V1 = x1[4]; err1 = Va_ref - V1
        integ1 += err1*Ts
        thr_cmd1 = float(np.clip(thr0 + Kp_thr*err1 + Ki_thr*integ1, 0.0, 1.0))
        if (thr_cmd1==0.0 and err1<0) or (thr_cmd1==1.0 and err1>0): integ1 -= err1*Ts

        V2 = x2[4]; err2 = Va_ref - V2
        integ2 += err2*Ts
        thr_cmd2 = float(np.clip(thr0 + Kp_thr*err2 + Ki_thr*integ2, 0.0, 1.0))
        if (thr_cmd2==0.0 and err2<0) or (thr_cmd2==1.0 and err2>0): integ2 -= err2*Ts

        # --- perintah bank
        u1 = l1_green.command(x1, path, Va_ref, wk)
        uL1_cap = l1_orange.command(x2, path, Va_ref, wk)

        # MPC uses measured V (smoothed inside)
        uMPC, qpinfo = mpc.step(x2[:4], wk, u_prev, V_meas=x2[4])
        QPlog["iter"].append(qpinfo.get("iter", np.nan))
        QPlog["obj"].append(qpinfo.get("obj", np.nan))
        QPlog["status"].append(qpinfo.get("status", ""))

        # --- hybrid L1→MPC
        if use_hybrid_l1_mpc:
            _, rerr, _, echi = orbit_metrics(x2[0], x2[1], x2[2], circle_C, circle_R, ccw=(not cw))
            if mode == "CAPTURE_L1":
                cond_r   = (rerr < r_err_enter_frac * circle_R)
                cond_ang = (abs(np.degrees(echi)) < head_align_deg)
                ok_cnt = ok_cnt + 1 if (cond_r and cond_ang) else max(0, ok_cnt-1)
                if ok_cnt >= hold_enter_steps: mode = "MPC_TRACK"
            else:
                if (rerr > r_err_exit_frac*circle_R) or (abs(np.degrees(echi)) > head_exit_deg):
                    mode = "CAPTURE_L1"; ok_cnt = 0
            target = 1.0 if mode=="MPC_TRACK" else 0.0
            blend += (blend_rate if target>blend else -blend_rate)
            blend = float(np.clip(blend,0.0,1.0))
            u2 = (1.0-blend)*uL1_cap + blend*uMPC
        else:
            rerr = 0.0; echi = 0.0; u2 = uMPC

        u2_applied = (1.0-alpha_cmd)*u_prev + alpha_cmd*u2 if use_cmd_filter else u2

        # --- integrasi plant 6-state
        x1 = rk4_step_long(x1, u1,        thr_cmd1, Ts, wk, tau_mu, m_aircraft, k_drag, k_thrust, tau_thr)
        x2 = rk4_step_long(x2, u2_applied,thr_cmd2, Ts, wk, tau_mu, m_aircraft, k_drag, k_thrust, tau_thr)

        # --- logging
        for (log, xk, uk) in ((L1log,x1,u1),(MPClog,x2,u2_applied)):
            log["n"].append(xk[0]); log["e"].append(xk[1]); log["chi"].append(xk[2])
            log["mu"].append(xk[3]); log["V"].append(xk[4]); log["thr"].append(xk[5]); log["u_cmd"].append(uk)
        HYBlog["t"].append(t); HYBlog["mode"].append(mode); HYBlog["blend"].append(blend)
        HYBlog["rerr"].append(rerr); HYBlog["echi_deg"].append(float(np.degrees(echi)))

        u_prev = u2_applied

    # list → np.array
    for log in (L1log, MPClog):
        for k in log: log[k] = np.array(log[k])
    return L1log, MPClog, HYBlog, QPlog

# ====== 3) Metrik & penyimpanan CSV ==================================
def prepare_metrics_and_save(path: CirclePath, L1log: Dict[str,np.ndarray], MPClog: Dict[str,np.ndarray]) -> Dict[str, Any]:
    et_L1 = crosstrack_series(path, L1log["n"], L1log["e"])
    et_M  = crosstrack_series(path, MPClog["n"], MPClog["e"])
    def rms(a): return float(np.sqrt(np.mean(a**2)))
    def settle_time(e, tol=5.0):
        for i, v in enumerate(e):
            if abs(v) <= tol: return i*Ts
        return T_end

    metrics = {
        "RMS_e_t_L1": rms(et_L1),
        "RMS_e_t_MPC": rms(et_M),
        "settle_s_L1(|e_t|<5m)": settle_time(et_L1, 5.0),
        "settle_s_MPC(|e_t|<5m)": settle_time(et_M, 5.0),
        "RMS_bank_L1_deg": rms(np.degrees(L1log["mu"])),
        "RMS_bank_MPC_deg": rms(np.degrees(MPClog["mu"])),
        "Mean_V_L1": float(np.mean(L1log["V"])),
        "Mean_V_MPC": float(np.mean(MPClog["V"])),
    }
    pd.DataFrame([metrics]).to_csv(savepath("metrics_compare.csv"), index=False)

    t_arr = np.arange(len(MPClog["n"])) * Ts
    pd.DataFrame({
        "t_s": t_arr,
        "e_t_MPC": et_M,
        "e_t_L1": et_L1,
        "V_MPC": MPClog["V"],
        "V_L1": L1log["V"],
        "thr_MPC": MPClog["thr"],
        "thr_L1": L1log["thr"],
        "abs_e_t_MPC": np.abs(et_M),
        "abs_e_t_L1": np.abs(et_L1),
    }).to_csv(savepath("et_series.csv"), index=False)

    IAE_M = float(np.sum(np.abs(et_M)) * Ts)
    IAE_L = float(np.sum(np.abs(et_L1)) * Ts)
    MAX_M = float(np.max(np.abs(et_M)))
    MAX_L = float(np.max(np.abs(et_L1)))
    pd.DataFrame([{
        "RMS_e_t_MPC": rms(et_M),
        "IAE_e_t_MPC": IAE_M,
        "MAX_e_t_MPC": MAX_M,
        "RMS_e_t_L1": rms(et_L1),
        "IAE_e_t_L1": IAE_L,
        "MAX_e_t_L1": MAX_L,
    }]).to_csv(savepath("metrics_crosstrack.csv"), index=False)

    return {"et_L1": et_L1, "et_MPC": et_M, "t": t_arr, "metrics": metrics}

# ====== 4) Plot statik (traj + e_t + bank) ===========================
def make_static_plots(path: CirclePath, L1log: Dict[str,np.ndarray], MPClog: Dict[str,np.ndarray], aux: Dict[str, Any]) -> None:
    et_L1, et_M, t_arr = aux["et_L1"], aux["et_MPC"], aux["t"]
    # --- TRAJ + wind label
    fig, ax = plt.subplots(figsize=(6,6))
    al = np.linspace(0, 2*np.pi, 400)
    ax.plot(path.C[0] + path.R*np.cos(al), path.C[1] + path.R*np.sin(al),
            '--', lw=1.3, color='tab:gray', alpha=alpha_circle,
            label=f"Circle ref ({'CW' if path.cw else 'CCW'})")

    all_n = np.concatenate([MPClog["n"], L1log["n"]])
    all_e = np.concatenate([MPClog["e"], L1log["e"]])
    nxmin, nxmax = float(np.min(all_n)), float(np.max(all_n))
    exmin, exmax = float(np.min(all_e)), float(np.max(all_e))
    pad_n = 0.1*(nxmax-nxmin + 1e-6); pad_e = 0.1*(exmax-exmin + 1e-6)
    arrow_n = nxmin + pad_n; arrow_e = exmax - pad_e

    t_samp = np.linspace(0, min(10.0, T_end), 30)
    wind_samples = np.array([wind_at(t) for t in t_samp])
    w_mean = wind_samples.mean(axis=0)
    wn_hist = np.array([wind_at(i*Ts)[0] for i in range(len(MPClog["chi"]))])
    we_hist = np.array([wind_at(i*Ts)[1] for i in range(len(MPClog["chi"]))])
    gs_hist = np.hypot(MPClog["V"]*np.cos(MPClog["chi"]) + wn_hist,
                       MPClog["V"]*np.sin(MPClog["chi"]) + we_hist)

    ax.quiver(arrow_n, arrow_e, w_mean[0], w_mean[1],
              angles='xy', scale_units='xy', scale=1, color='tab:blue', alpha=alpha_quiver, width=0.005)

    ax.text(0.02, 0.98,
            f"Wind ≈ {np.linalg.norm(w_mean):.1f} m/s\n"
            f"UAV speed avg ≈ {float(np.mean(gs_hist)):.1f} m/s\n"
            f"Wind mode: {wind_mode_str()}",
            transform=ax.transAxes, ha='left', va='top', fontsize=9, color='k')

    ax.plot(MPClog["n"], MPClog["e"], label="MPC", color="tab:orange", alpha=alpha_trails)
    ax.plot(L1log["n"], L1log["e"], label="L1",  color="tab:green",  alpha=alpha_trails)
    ax.set_aspect("equal"); ax.set_xlabel("North [m]"); ax.set_ylabel("East [m]")
    ax.grid(True, alpha=0.4); leg = ax.legend(loc="upper right"); leg.set_alpha(1.0)
    savefig_here("traj_compare_with_wind.png")

    # --- |e_t| vs step
    plt.figure()
    plt.plot(et_M, label="MPC e_t", alpha=0.9)
    plt.plot(et_L1, label="L1 e_t", alpha=0.9)
    plt.xlabel("Step"); plt.ylabel("Cross-track error [m]")
    plt.legend(); plt.grid(True); savefig_here("et_compare.png")

    # --- bank
    plt.figure()
    plt.plot(np.degrees(MPClog["u_cmd"]), label="MPC mu_ref [deg]", alpha=0.9)
    plt.plot(np.degrees(MPClog["mu"]),    label="MPC mu [deg]",     alpha=0.9)
    plt.plot(np.degrees(L1log["u_cmd"]), "--", label="L1 mu_ref [deg]", alpha=0.8)
    plt.plot(np.degrees(L1log["mu"]),   "--", label="L1 mu [deg]",     alpha=0.8)
    plt.xlabel("Step"); plt.ylabel("Bank angle [deg]")
    plt.legend(); plt.grid(True); savefig_here("bank_compare.png")

    # --- e_t vs time
    plt.figure()
    plt.plot(t_arr, et_M, label="MPC e_t", alpha=0.9)
    plt.plot(t_arr, et_L1, label="L1 e_t", alpha=0.9)
    plt.axhline(0, ls="--", lw=0.8)
    plt.xlabel("Time [s]"); plt.ylabel("Cross-track error e_t [m]")
    plt.legend(); plt.grid(True); savefig_here("et_series_vs_time.png")

# ====== 5) Animation ==================================================
def make_aircraft_patch(ax, size=10.0, color="tab:orange", zorder=5):
    base = np.array([[+1.2, 0.0], [-1.0, +0.6], [-0.6, 0.0], [-1.0, -0.6]], dtype=float) * size
    patch = Polygon(base, closed=True, facecolor=color, edgecolor="k", lw=0.8, zorder=zorder, animated=True, antialiased=True)
    patch._base_shape = base.copy()
    ax.add_patch(patch)
    return patch

def set_pose_patch(patch, n, e, chi):
    base = patch._base_shape
    R = np.array([[np.cos(chi), -np.sin(chi)],
                  [np.sin(chi),  np.cos(chi)]], dtype=float)
    verts = base @ R.T + np.array([n, e], dtype=float)
    patch.set_xy(verts)

def animate_loiter(path: CirclePath, MPClog: Dict[str,np.ndarray], L1log: Dict[str,np.ndarray], Ts: float,
                   every_k: int=2, plane_size: float=12.0, duration_s: float=20.0,
                   save_gif: bool=True, save_mp4: bool=False) -> None:
    # Auto-subsample to target ~600 frames
    total = len(MPClog["n"])
    target_frames = max(300, int(duration_s*20))  # aim ~20 fps visual density
    every_k = max(1, total // target_frames) if total > target_frames else max(1, every_k)

    idx = np.arange(0, total, every_k)
    n_nm, e_nm, chi_nm, mu_nm, V_nm = MPClog["n"][idx], MPClog["e"][idx], MPClog["chi"][idx], MPClog["mu"][idx], MPClog["V"][idx]
    n_l1, e_l1, chi_l1 = L1log["n"][idx], L1log["e"][idx], L1log["chi"][idx]
    frames = len(idx)
    fps = min(30, max(5, int(frames / max(1, duration_s))))

    fig, ax = plt.subplots(figsize=(6,6))
    al = np.linspace(0, 2*np.pi, 400)
    ax.plot(path.C[0] + path.R*np.cos(al), path.C[1] + path.R*np.sin(al),
            "--", lw=1.3, label="Circle ref ({})".format("CW" if path.cw else "CCW"), color="tab:gray", alpha=alpha_circle)
    trail_nm, = ax.plot([], [], "-", lw=2.0, color="tab:orange", alpha=alpha_trails, label="MPC", animated=True)
    trail_l1, = ax.plot([], [], "-", lw=2.0, color="tab:green",  alpha=alpha_trails, label="L1",  animated=True)

    plane_nm = plane_l1 = None
    if show_planes:
        plane_nm = make_aircraft_patch(ax, size=plane_size, color="tab:orange", zorder=6)
        plane_l1 = make_aircraft_patch(ax, size=plane_size, color="tab:green",  zorder=6)

    all_n = np.concatenate([MPClog["n"], L1log["n"]])
    all_e = np.concatenate([MPClog["e"], L1log["e"]])
    pad = 0.2 * max(path.R, np.std(all_n) + np.std(all_e))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(all_n))-pad, float(np.max(all_n))+pad)
    ax.set_ylim(float(np.min(all_e))-pad, float(np.max(all_e))+pad)
    ax.set_xlabel("North [m]"); ax.set_ylabel("East [m]")
    ax.grid(True, alpha=0.4); ax.legend(loc="upper right")

    # --- Dynamic wind indicator (arrow + text) ---
    nxmin, nxmax = ax.get_xlim()
    exmin, exmax = ax.get_ylim()
    arrow_n = nxmin + 0.10*(nxmax-nxmin)
    arrow_e = exmax - 0.10*(exmax-exmin)
    wn0, we0 = wind_at(0.0, state=(n_nm[0], e_nm[0], chi_nm[0], mu_nm[0]))
    vg0_n = V_nm[0]*np.cos(chi_nm[0]) + wn0
    vg0_e = V_nm[0]*np.sin(chi_nm[0]) + we0
    gs0 = float(np.hypot(vg0_n, vg0_e))

    wind_arrow = ax.quiver(arrow_n, arrow_e, wn0, we0, angles='xy', scale_units='xy', scale=1,
                           color='tab:blue', alpha=alpha_quiver, width=0.006, zorder=7, animated=True)
    mode_str = wind_mode_str()
    wind_text = ax.text(0.02, 0.98, f"Wind = {np.hypot(wn0,we0):.1f} m/s\nUAV speed = {gs0:.1f} m/s\nWind mode: {mode_str}",
                        transform=ax.transAxes, ha='left', va='top', fontsize=9, color='k')
    wind_text.set_animated(True)

    def init():
        trail_nm.set_data([], []); trail_l1.set_data([], [])
        if show_planes:
            set_pose_patch(plane_nm, n_nm[0], e_nm[0], chi_nm[0])
            set_pose_patch(plane_l1, n_l1[0], e_l1[0], chi_l1[0])
            return trail_nm, trail_l1, wind_arrow, wind_text, plane_nm, plane_l1
        return trail_nm, trail_l1, wind_arrow, wind_text

    def update(i):
        nonlocal wind_arrow
        trail_nm.set_data(n_nm[:i+1], e_nm[:i+1])
        trail_l1.set_data(n_l1[:i+1], e_l1[:i+1])
        if show_planes:
            set_pose_patch(plane_nm, n_nm[i], e_nm[i], chi_nm[i])
            set_pose_patch(plane_l1, n_l1[i], e_l1[i], chi_l1[i])
        t_i = float(idx[i]) * Ts
        wn, we = wind_at(t_i, state=(n_nm[i], e_nm[i], chi_nm[i], mu_nm[i]))
        try:
            wind_arrow.set_UVC(wn, we)
        except Exception:
            try:
                wind_arrow.remove()
            except Exception:
                pass
            wind_arrow = ax.quiver(arrow_n, arrow_e, wn, we, angles='xy', scale_units='xy', scale=1,
                                   color='tab:blue', alpha=alpha_quiver, width=0.006, zorder=7, animated=True)
        vg_n = V_nm[i]*np.cos(chi_nm[i]) + wn
        vg_e = V_nm[i]*np.sin(chi_nm[i]) + we
        gs = float(np.hypot(vg_n, vg_e))
        wind_text.set_text(f"Wind = {np.hypot(wn,we):.1f} m/s\nUAV speed = {gs:.1f} m/s\nWind mode: {mode_str}")
        if show_planes:
            return trail_nm, trail_l1, wind_arrow, wind_text, plane_nm, plane_l1
        return trail_nm, trail_l1, wind_arrow, wind_text

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=frames, interval=1000/fps, blit=True)
    if save_gif:
        try: anim.save(savepath("uav_loiter_anim.gif"), writer="pillow", fps=fps)
        except Exception as e: print("GIF save failed:", e)
    if save_mp4:
        try: anim.save(savepath("uav_loiter_anim.mp4"), writer="ffmpeg", fps=fps, dpi=160)
        except Exception as e: print("MP4 save failed:", e)
    plt.close(fig)

# ---------------------- Main run -------------------------------------
def main():
    path, l1_green, l1_orange, mpc = build_controllers()
    L1log, MPClog, HYBlog, QPlog = simulate(path, l1_green, l1_orange, mpc)
    aux = prepare_metrics_and_save(path, L1log, MPClog)
    # Save QP stats
    pd.DataFrame(QPlog).to_csv(savepath("qp_stats.csv"), index=False)
    make_static_plots(path, L1log, MPClog, aux)

    if make_animation:
        animate_loiter(path, MPClog, L1log, Ts,
                       every_k=anim_every_kstep,
                       plane_size=plane_size_m,
                       duration_s=anim_duration_s,
                       save_gif=save_gif, save_mp4=save_mp4)

    print(f"Artifacts saved in: {run_tag}")
    print(" - traj_compare_with_wind.png, et_compare.png, bank_compare.png, et_series_vs_time.png")
    print(" - metrics_compare.csv, metrics_crosstrack.csv, et_series.csv, qp_stats.csv")
    if make_animation:
        print(" - uav_loiter_anim.gif{}".format(" and uav_loiter_anim.mp4" if save_mp4 else ""))

if __name__ == "__main__":
    main()
