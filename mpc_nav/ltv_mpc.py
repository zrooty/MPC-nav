"""LTV-MPC for orbit tracking, solved as a QP with OSQP.

Two models are supported:

* When a ``LateralLQR`` is supplied: 5-state x = [n, e, chi, mu, p] with
  ZOH discretisation that includes the closed-loop roll dynamics.
* Otherwise: 4-state x = [n, e, chi, mu] with explicit Euler (kept for
  backward compatibility).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
import osqp

from .geometry import CirclePath, wrap

_G = 9.81


@dataclass
class MPCWeights:
    w_et: float
    w_echi: float
    w_mu: float
    w_u: float
    w_et_T: float
    w_echi_T: float


class LTVMPC_OSQP:
    def __init__(self, Ts: float, N: int, Va_init: float, tau_mu: float,
                 bank_limit_deg: float, slew_limit_deg_s: float,
                 weights: MPCWeights, path: CirclePath,
                 w_du: float = 80.0, w_mu_Term: float = 4.0,
                 use_groundspeed_mu_ss: bool = True,
                 Va_min: float = 6.0, Va_max: float = 60.0,
                 Va_lp_tau: float = 0.4, Va_nominal: float = 17.0,
                 use_w_du_scaling: bool = False,
                 lateral_lqr=None, a_p_roll: float = 0.0, b_p_roll: float = 0.0):
        self.Ts = float(Ts)
        self.N = int(N)
        self.Va_model = float(Va_init)        # smoothed for linearisation
        self.Va_min = float(Va_min)
        self.Va_max = float(Va_max)
        self.Va_lp_tau = float(Va_lp_tau)
        self.Va_nominal = float(Va_nominal)
        self.tau = float(tau_mu)
        self.path = path
        self.W = weights
        self.w_du = float(w_du)
        self.use_w_du_scaling = bool(use_w_du_scaling)
        self.w_mu_Term = float(w_mu_Term)
        self.mu_max = np.radians(bank_limit_deg)
        self.du_max = np.radians(slew_limit_deg_s) * self.Ts
        self.use_groundspeed_mu_ss = bool(use_groundspeed_mu_ss)
        self.nu = 1

        # ----- Pick 4- or 5-state model -----
        if lateral_lqr is not None and b_p_roll != 0.0:
            K_phi_ail = lateral_lqr.K[0, 0] * lateral_lqr._aileron_scale
            K_p_ail = lateral_lqr.K[0, 1] * lateral_lqr._aileron_scale
            # Closed-loop LQR roll dynamics:
            #   p_dot = -K_phi_cl*(mu - u) + a_eff_cl * p
            self._K_phi_cl = b_p_roll * K_phi_ail              # ≈ 178.9
            self._a_eff_cl = a_p_roll - b_p_roll * K_p_ail     # ≈ -48.8
            self.nx = 5
            # Pick RK4 substep so |λ_fast*dt| ≤ 1.5 (well inside the |λ*dt|≤2.79 stability bound)
            disc = self._a_eff_cl ** 2 - 4.0 * self._K_phi_cl
            lam_fast = (self._a_eff_cl - math.sqrt(max(0.0, disc))) / 2.0
            self._rk4_sub = max(1, math.ceil(abs(lam_fast) * self.Ts / 1.5))
            print(f"[LTVMPC] 5-state model: lam_fast={lam_fast:.1f} rad/s -> rk4_sub={self._rk4_sub}")
        else:
            self._K_phi_cl = None
            self._a_eff_cl = None
            self.nx = 4
            self._rk4_sub = 1

        # E_mu selects the bank state out of x.
        self.E_mu = np.zeros((1, self.nx))
        self.E_mu[0, 3] = 1.0

        # Warm-start cache
        self._z_prev = None
        self._y_prev = None
        self._u_opt_seq = None

        # OSQP workspace & templates
        self._prob = None
        self._P_tpl = None
        self._A_tpl = None
        self._P = None
        self._A = None

    # ------------------------------------------------------------------
    # Continuous-time dynamics and discretisation
    # ------------------------------------------------------------------
    def _f(self, x: np.ndarray, u: float, wind: Tuple[float, float]) -> np.ndarray:
        wn, we = wind
        Va = self.Va_model
        n, e, chi, mu = x[0], x[1], x[2], x[3]
        if self.nx == 5:
            p = x[4]
            return np.array([
                Va * np.cos(chi) + wn,
                Va * np.sin(chi) + we,
                (_G / max(1.0, Va)) * np.tan(mu),
                p,
                -self._K_phi_cl * mu + self._a_eff_cl * p + self._K_phi_cl * u,
            ], dtype=float)
        # 4-state Euler-friendly model
        return np.array([
            Va * np.cos(chi) + wn,
            Va * np.sin(chi) + we,
            (_G / max(1.0, Va)) * np.tan(mu),
            -(mu - u) / self.tau,
        ], dtype=float)

    def _rk4(self, x: np.ndarray, u: float, wind: Tuple[float, float]) -> np.ndarray:
        dt = self.Ts / self._rk4_sub
        for _ in range(self._rk4_sub):
            k1 = self._f(x, u, wind)
            k2 = self._f(x + 0.5 * dt * k1, u, wind)
            k3 = self._f(x + 0.5 * dt * k2, u, wind)
            k4 = self._f(x + dt * k3, u, wind)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    def _AB_discrete(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Discrete-time A,B at the linearisation point ``x``.

        5-state: exact ZOH via matrix exponential.
        4-state: explicit Euler (backward compatible).
        """
        chi, mu = x[2], x[3]
        Va = self.Va_model
        nx, nu = self.nx, self.nu
        Jc = np.zeros((nx, nx))
        Jc[0, 2] = -Va * np.sin(chi)
        Jc[1, 2] = Va * np.cos(chi)
        Jc[2, 3] = (_G / max(1.0, Va)) * (1 / np.cos(mu)) ** 2
        Bc = np.zeros((nx, nu))

        if nx == 5:
            Jc[3, 4] = 1.0
            Jc[4, 3] = -self._K_phi_cl
            Jc[4, 4] = self._a_eff_cl
            Bc[4, 0] = self._K_phi_cl
            M = np.zeros((nx + nu, nx + nu))
            M[:nx, :nx] = Jc
            M[:nx, nx:] = Bc
            eM = expm(M * self.Ts)
            return eM[:nx, :nx], eM[:nx, nx:]

        # 4-state: Euler
        Jc[3, 3] = -1.0 / self.tau
        Bc[3, 0] = 1.0 / self.tau
        return np.eye(nx) + self.Ts * Jc, self.Ts * Bc

    # ------------------------------------------------------------------
    # Output mapping y = [e_t, e_chi, mu] and steady-state bank
    # ------------------------------------------------------------------
    def _et_echi_and_C(self, x: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        n, e, chi, mu = x[0], x[1], x[2], x[3]
        Cc = np.asarray(self.path.C)
        rvec = np.array([n, e]) - Cc
        r = float(np.linalg.norm(rvec)) + 1e-9
        et0 = r - self.path.R
        alpha = math.atan2(rvec[1], rvec[0])
        chi_d = alpha - (math.pi / 2) if self.path.cw else alpha + (math.pi / 2)
        echi0 = wrap(chi_d - chi)

        rhat = rvec / r
        dalpha = np.array([-np.sin(alpha) / r, np.cos(alpha) / r])
        nx = self.nx
        det = np.zeros(nx)
        det[0:2] = rhat
        dechi = np.zeros(nx)
        dechi[0:2] = dalpha
        dechi[2] = -1.0
        dmu = np.zeros(nx)
        dmu[3] = 1.0
        C = np.vstack([det, dechi, dmu]).astype(float)   # 3 × nx
        y0 = np.array([et0, echi0, mu], dtype=float)
        return et0, echi0, C, y0

    def _mu_ss(self, x: np.ndarray, wind: Tuple[float, float]) -> float:
        Va = self.Va_model
        if self.use_groundspeed_mu_ss:
            vg = np.array([Va * np.cos(x[2]) + wind[0], Va * np.sin(x[2]) + wind[1]])
            Vg = max(5.0, float(np.linalg.norm(vg)))
            mu_ss_mag = math.atan2(Vg * Vg, _G * self.path.R)
        else:
            mu_ss_mag = math.atan2(Va * Va, _G * self.path.R)
        return (-mu_ss_mag) if self.path.cw else (+mu_ss_mag)

    # ------------------------------------------------------------------
    # OSQP sparsity helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _gather_data_from_dense(M_full: np.ndarray, template_csc: sp.csc_matrix) -> np.ndarray:
        """Pick entries of ``M_full`` at the nonzero positions of ``template_csc``."""
        indptr = template_csc.indptr
        indices = template_csc.indices
        out = template_csc.data.copy()
        for j in range(template_csc.shape[1]):
            start, end = indptr[j], indptr[j + 1]
            rows = indices[start:end]
            out[start:end] = M_full[rows, j]
        return out

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, x_cur: np.ndarray, wind: Tuple[float, float],
             u_prev: float, V_meas: float | None = None) -> Tuple[float, Dict[str, Any]]:
        self._update_va_model(V_meas)

        N, nx, nu = self.N, self.nx, self.nu
        x_nom, u_nom_seq = self._build_nominal(x_cur, u_prev, wind)

        # Linearisation around the nominal trajectory
        Aks, Bks = [], []
        for k in range(N):
            A_k, B_k = self._AB_discrete(x_nom[k, :])
            Aks.append(A_k)
            Bks.append(B_k)

        P_num, q = self._build_cost(x_nom, u_prev, wind)
        A_full, l_vec, u_vec = self._build_constraints(x_nom, u_prev, Aks, Bks)

        u_cmd, info = self._solve_qp(P_num, q, A_full, l_vec, u_vec, u_prev)
        return u_cmd, info

    # ----- step() helpers --------------------------------------------
    def _update_va_model(self, V_meas: float | None) -> None:
        if V_meas is None:
            return
        v = float(np.clip(V_meas, self.Va_min, self.Va_max))
        alpha = self.Ts / max(self.Va_lp_tau, self.Ts)
        self.Va_model = (1.0 - alpha) * self.Va_model + alpha * v

    def _build_nominal(self, x_cur: np.ndarray, u_prev: float,
                       wind: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Nominal x-trajectory used for linearisation.

        Reuse the previous optimal u-sequence (shifted) only when the last
        solution was well inside the bank envelope — during aggressive
        capture an extremal linearisation point can destabilise the QP.
        """
        N, nx = self.N, self.nx
        x_nom = np.zeros((N + 1, nx))
        x_nom[0, :] = x_cur

        use_shifted = (self._u_opt_seq is not None
                       and len(self._u_opt_seq) == N
                       and np.max(np.abs(self._u_opt_seq)) < 0.7 * self.mu_max)
        u_nom_seq = (np.append(self._u_opt_seq[1:], self._u_opt_seq[-1])
                     if use_shifted else np.full(N, u_prev))

        for k in range(N):
            x_nom[k + 1, :] = self._rk4(x_nom[k, :], u_nom_seq[k], wind)
        return x_nom, u_nom_seq

    def _build_cost(self, x_nom: np.ndarray, u_prev: float,
                    wind: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Build upper-triangular P (×2 for OSQP convention) and linear term q."""
        N, nx, nu = self.N, self.nx, self.nu
        Wst = np.diag([self.W.w_et, self.W.w_echi, self.W.w_mu])
        WT = np.diag([self.W.w_et_T, self.W.w_echi_T, 0.0])

        nzx = (N + 1) * nx
        nzu = N * nu
        nz = nzx + nzu
        Q = np.zeros((nz, nz), dtype=float)
        q = np.zeros(nz, dtype=float)

        def sx(k): return slice(k * nx, (k + 1) * nx)
        def su(k): return slice(nzx + k * nu, nzx + (k + 1) * nu)

        # Stage costs: state tracking + control magnitude
        for k in range(N):
            _, _, Ck, y0k = self._et_echi_and_C(x_nom[k, :])
            Q[sx(k), sx(k)] += Ck.T @ Wst @ Ck
            q[sx(k)] += 2.0 * (Ck.T @ Wst @ y0k)
            Q[su(k), su(k)] += self.W.w_u * np.eye(nu)
            q[su(k)] += 2.0 * self.W.w_u * u_prev

        # Terminal output cost
        _, _, CT, y0T = self._et_echi_and_C(x_nom[N, :])
        Q[sx(N), sx(N)] += CT.T @ WT @ CT
        q[sx(N)] += 2.0 * (CT.T @ WT @ y0T)

        # Terminal bank penalty (mu_N - mu_ss)^2
        mu_ss = self._mu_ss(x_nom[N, :], wind)
        mu_nom_N = x_nom[N, 3]
        E_mu = self.E_mu
        Q[sx(N), sx(N)] += self.w_mu_Term * (E_mu.T @ E_mu)
        q[sx(N)] += (2.0 * self.w_mu_Term * E_mu.T * (mu_nom_N - mu_ss)).ravel()

        # Δu penalty (tri-diagonal on u-block), optionally Va-scaled
        w_du_eff = self.w_du * ((self.Va_model / self.Va_nominal) ** 2
                                if self.use_w_du_scaling else 1.0)
        if w_du_eff > 0.0 and N > 0:
            I_nu = np.eye(nu)
            for k in range(N):
                Q[su(k), su(k)] += w_du_eff * I_nu
                if k > 0:
                    Q[su(k - 1), su(k - 1)] += w_du_eff * I_nu
                    Q[su(k), su(k - 1)] -= w_du_eff * I_nu
                    Q[su(k - 1), su(k)] -= w_du_eff * I_nu

        P_full = (Q + Q.T) * 0.5
        P_full = 2.0 * P_full              # OSQP uses 0.5 z^T P z + q^T z
        P_num = np.triu(P_full)
        return P_num, q

    def _build_constraints(self, x_nom: np.ndarray, u_prev: float,
                           Aks, Bks) -> Tuple[sp.csc_matrix, np.ndarray, np.ndarray]:
        N, nx, nu = self.N, self.nx, self.nu
        nzx = (N + 1) * nx
        nz  = nzx + N * nu

        # Constraint row counts: initial + dynamics + bank_limits + input_limits + slew
        n_rows = nx + N * nx + (N + 1) + N + N
        l_vec  = np.empty(n_rows)
        u_vec  = np.empty(n_rows)

        # Pre-allocate COO arrays (upper bound on nnz)
        max_nnz = nx + N * (nx + nx * nx + nx) + (N + 1) + N + (1 + (N - 1) * 2)
        ri = np.empty(max_nnz, dtype=np.int32)
        ci = np.empty(max_nnz, dtype=np.int32)
        vd = np.empty(max_nnz, dtype=float)
        ptr = 0  # COO fill pointer

        r = 0  # current constraint row

        # --- Initial: δx_0 = 0 ---
        ri[ptr:ptr+nx] = np.arange(nx); ci[ptr:ptr+nx] = np.arange(nx); vd[ptr:ptr+nx] = 1.0
        ptr += nx
        l_vec[r:r+nx] = 0.0; u_vec[r:r+nx] = 0.0
        r += nx

        # --- Dynamics: δx_{k+1} = Ak δx_k + Bk δu_k (vectorised over k) ---
        Aks_arr = np.array(Aks)  # (N, nx, nx)
        Bks_arr = np.array(Bks).reshape(N, nx)  # (N, nx)

        # Identity block for δx_{k+1}: nx entries per step
        for k in range(N):
            base_r = r + k * nx
            base_c = (k + 1) * nx
            ri[ptr:ptr+nx] = np.arange(base_r, base_r + nx)
            ci[ptr:ptr+nx] = np.arange(base_c, base_c + nx)
            vd[ptr:ptr+nx] = 1.0
            ptr += nx

        # -Ak blocks: nx*nx entries per step (full; Ak is always dense for this model)
        row_base = r + np.repeat(np.arange(N) * nx, nx * nx) + np.tile(np.arange(nx).repeat(nx), N)
        col_base = np.tile(np.arange(nx * nx) % nx, N) + np.repeat(np.arange(N) * nx, nx * nx)
        n_Ak = N * nx * nx
        ri[ptr:ptr+n_Ak] = row_base
        ci[ptr:ptr+n_Ak] = col_base
        vd[ptr:ptr+n_Ak] = -Aks_arr.reshape(-1)
        ptr += n_Ak

        # -Bk columns: nx entries per step
        for k in range(N):
            base_r = r + k * nx
            ri[ptr:ptr+nx] = np.arange(base_r, base_r + nx)
            ci[ptr:ptr+nx] = nzx + k * nu
            vd[ptr:ptr+nx] = -Bks_arr[k]
            ptr += nx

        l_vec[r:r+N*nx] = 0.0; u_vec[r:r+N*nx] = 0.0
        r += N * nx

        # --- Bank limits: E_mu δx_k ---
        mu_col = 3
        for k in range(N + 1):
            ri[ptr] = r; ci[ptr] = k * nx + mu_col; vd[ptr] = 1.0; ptr += 1
            l_vec[r] = -self.mu_max - x_nom[k, 3]
            u_vec[r] =  self.mu_max - x_nom[k, 3]
            r += 1

        # --- Input limits: δu_k ---
        for k in range(N):
            ri[ptr] = r; ci[ptr] = nzx + k * nu; vd[ptr] = 1.0; ptr += 1
            l_vec[r] = -self.mu_max - u_prev
            u_vec[r] =  self.mu_max - u_prev
            r += 1

        # --- Slew ---
        ri[ptr] = r; ci[ptr] = nzx; vd[ptr] = 1.0; ptr += 1
        l_vec[r] = -self.du_max; u_vec[r] = self.du_max
        r += 1
        for k in range(1, N):
            ri[ptr] = r; ci[ptr] = nzx + k * nu;       vd[ptr] =  1.0; ptr += 1
            ri[ptr] = r; ci[ptr] = nzx + (k-1) * nu;   vd[ptr] = -1.0; ptr += 1
            l_vec[r] = -self.du_max; u_vec[r] = self.du_max
            r += 1

        A_full = sp.csc_matrix((vd[:ptr], (ri[:ptr], ci[:ptr])), shape=(n_rows, nz))
        return A_full, l_vec, u_vec

    def _solve_qp(self, P_num: np.ndarray, q: np.ndarray,
                  A_full: sp.csc_matrix, l_vec: np.ndarray, u_vec: np.ndarray,
                  u_prev: float) -> Tuple[float, Dict[str, Any]]:
        N, nx = self.N, self.nx

        # Reset solver if the A-pattern changed (e.g. sin(chi)→0 at chi≈0/π)
        if self._prob is not None and len(A_full.data) != len(self._A.data):
            self._prob = None

        if self._prob is None:
            # Template: nonzero pattern of the current upper-triangular P
            mask = (np.abs(P_num) > 1e-12).astype(float)
            self._P_tpl = sp.csc_matrix(mask)
            P_data = self._gather_data_from_dense(P_num, self._P_tpl)
            self._P = self._P_tpl.copy()
            self._P.data[:] = P_data

            self._A_tpl = A_full.copy()
            self._A = self._A_tpl.copy()
            self._A.data[:] = A_full.data

            self._prob = osqp.OSQP()
            self._prob.setup(P=self._P, q=q, A=self._A, l=l_vec, u=u_vec,
                             verbose=False, eps_abs=1e-4, eps_rel=1e-4,
                             polish=False, max_iter=4000, adaptive_rho=True)
            self._try_warm_start()
        else:
            P_data = self._gather_data_from_dense(P_num, self._P_tpl)
            self._P.data[:] = P_data
            self._A.data[:] = A_full.data           # assumes structure unchanged
            self._prob.update(Px=self._P.data, q=q, Ax=self._A.data, l=l_vec, u=u_vec)
            self._try_warm_start()

        res = self._prob.solve()
        info = {
            "status": res.info.status,
            "fallback": False,
            "iter": getattr(res.info, "iter", np.nan),
            "obj": getattr(res.info, "obj_val", np.nan),
        }

        if res.info.status_val not in (1, 2):
            # Infeasible / failed → hold previous u
            info["fallback"] = True
            return float(u_prev), info

        self._z_prev = res.x.copy()
        try:
            self._y_prev = res.y.copy()
        except Exception:
            self._y_prev = None

        z = res.x
        self._u_opt_seq = np.array([u_prev + float(z[(N + 1) * nx + k]) for k in range(N)])
        du0 = z[(N + 1) * nx + 0] if N > 0 else 0.0
        u_cmd = float(np.clip(u_prev + du0, -self.mu_max, self.mu_max))
        return u_cmd, info

    def _try_warm_start(self) -> None:
        if self._z_prev is None:
            return
        try:
            if self._y_prev is not None:
                self._prob.warm_start(x=self._z_prev, y=self._y_prev)
            else:
                self._prob.warm_start(x=self._z_prev)
        except Exception:
            pass
