#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArduPlane FBWB lateral driver (RC override CH1) with **all parameters in this file**.
- Choose LATERAL_MODE: "l1" | "mpc" | "blend"
- L1 and MPC are instantiated using parameters below (no reliance on MPC_15 constants).
- Path, weights, limits, horizon, wind feed, and shaping are all configured here.
"""

import os, math, time, csv, threading, importlib, datetime as _dt
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from pymavlink import mavutil

# ==== MPC module resolution (set path here if needed) ====
MPC_MODULE_NAME = "MPC_v2"
MPC_MODULE_PATH = ""  # e.g., r"C:\\Users\\you\\project\\MPC_15.py" or "/home/you/project/MPC_15.py"

def _load_mpc_module():
    import importlib, importlib.util, sys, os
    name = MPC_MODULE_NAME
    if MPC_MODULE_PATH:
        mp = MPC_MODULE_PATH
        if not os.path.exists(mp):
            raise FileNotFoundError(f"[INIT] MPC module not found at MPC_MODULE_PATH='{mp}'. "
                                    f"Set MPC_MODULE_PATH to the absolute path of MPC_15.py, "
                                    f"or place MPC_15.py in the same folder as this driver.")
        spec = importlib.util.spec_from_file_location(name, mp)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        print(f"[INIT] Loaded MPC module from explicit path: {mp}")
        return mod
    try:
        mod = importlib.import_module(name)
        print(f"[INIT] Imported MPC module by name: {name}")
        return mod
    except ModuleNotFoundError:
        here = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
        alt = os.path.join(here, name + ".py")
        if os.path.exists(alt):
            spec = importlib.util.spec_from_file_location(name, alt)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            print(f"[INIT] Loaded MPC module from local file: {alt}")
            return mod
        raise FileNotFoundError(f"[INIT] Cannot find MPC_15.py. Put MPC_15.py in the same folder as this driver "
                                f"or set MPC_MODULE_PATH to its absolute path.")

# ====================== USER PARAMETERS (EDIT HERE) ====================
# MAVLink
MAVLINK_URL  = "udp:127.0.0.1:14550"

# Mode: "l1" (capture only), "mpc" (tracker only), "blend" (L1→MPC with gating+easing)
LATERAL_MODE = "blend"

# Timing
Ts               = 0.1     # controller step [s]
TX_HZ            = 15.0     # RC override send rate [Hz]
STALE_TIMEOUT    = 0.7      # state freshness guard [s]

# Path (circle) — used by both L1 & MPC
LOITER_C   = (0.0, 0.0)     # (N,E) [m]
LOITER_R   = 300.0          # [m]
LOITER_CW  = True           # True: clockwise

# L1 settings
L1_PERIOD   = 17.0          # [s]
L1_DAMPING  = 0.75          # [-]
BANK_LIMIT_DEG = 50.0       # [deg] bank limit seen by both L1 & MPC

# MPC model & solver settings
MPC_TAU_MU          = 0.6    # [s] bank time constant (LOES)
MPC_HORIZON         = 30     # steps
MPC_SLEW_DEG_S      = 120.0  # [deg/s] Δu/Δt limit inside MPC
MPC_USE_GS_FOR_MUss = True   # centripetal bank uses groundspeed if True
MPC_W_ET            = 120.0   # stage weight on cross-track radius error e_t
MPC_W_ECHI          = 10.0    # stage weight on heading-to-tangent error e_chi
MPC_W_MU            = 0.25   # stage weight on actual mu (keep small)
MPC_W_U             = 10.0    # stage weight on command u
MPC_W_ET_T          = 20.0   # terminal weights...
MPC_W_ECHI_T        = 3.0
MPC_W_DU            = 400.0   # rate penalty on Δu (total variation)
MPC_W_MU_TERM       = 4.0    # terminal penalty on (mu_N - mu_ss)^2
MPC_VA_INIT         = 21.0   # [m/s] initial speed used for linearization
MPC_VA_MIN_MAX      = (6.0, 30.0)
MPC_VA_LP_TAU       = 0.4    # [s] low-pass on measured Va used by MPC
MPC_USE_W_DU_SCALING= False  # scale w_du with Va^2 / Va_nom^2
MPC_VA_NOMINAL      = 21.0   # [m/s] for scaling (if enabled)

# Command shaping (outside MPC, applied to roll command before RC)
# SLEW_BANK_DPS    = 80.0      # [deg/s] external slew (driver level)
MAX_BANK_DEG     = BANK_LIMIT_DEG  # clamp for RC mapping

# Wind usage
USE_WIND_FEED = False         # feed (WIND) to L1 & MPC if available

# Hybrid gating (only when LATERAL_MODE == "blend")
BLEND_SECONDS    = 15.0
R_ERR_ENTER_FRAC = 0.35
HEAD_ALIGN_DEG   = 45.0
HOLD_ENTER_STEPS = 8
R_ERR_EXIT_FRAC  = 0.90
HEAD_EXIT_DEG    = 80.0

# MPC-only capture assist (optional; helps when far/heading off)
ASSIST_IN_MPC        = False
ASSIST_MAX_BLEND     = 0.45   # max mix of L1 into MPC when far
ASSIST_RERR_START    = 0.25   # start assist if |rho-R|/R > 0.25
ASSIST_HEAD_START_DEG= 35.0   # or |echi| > this [deg]
ASSIST_BOOTSTRAP_STEPS = 20   # force assist for first K steps

# --- Zig-Zag Killer: MPC feedforward warm-start ---
FF_ENABLE        = True      # aktif/nonaktifkan
FF_ALPHA         = 0.60      # porsi feedforward ke u_prev_warm (0..1) 0.60
FF_R_BIAS        = 0.25      # bias radial untuk dorong kembali ke R 0.25
FF_HYST_DEG      = 8.0       # histeresis tanda saat |eχ| kecil (deg) 8

# opsional: slew adaptif (lebih ketat saat sudah dekat tangent)
SLEW_NEAR_ECHI_DEG = 12.0    # ambang |eχ| untuk ketatkan slew
SLEW_BANK_DPS_NEAR = 40.0    # dps saat |eχ| < ambang


# RC mapping
PWM_TRIM = 1500
PWM_MIN  = 1000
PWM_MAX  = 2000

# Logging
RUN_TAG = _dt.datetime.now().strftime("logs_run_%Y%m%d-%H%M%S")
CSV_PATH = f"{RUN_TAG}/mavlink_driver_allparams_{RUN_TAG}.csv"

# ======================= Helper functions ============================
BOOT_T0 = time.monotonic()
def _wrap(a: float) -> float: return math.atan2(math.sin(a), math.cos(a))
def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x)); return x*x*(3.0 - 2.0*x)

def orbit_metrics(n, e, chi, C, R, ccw=True):
    r = np.array([n, e]) - np.asarray(C, float)
    rho = float(np.linalg.norm(r)) + 1e-9
    theta = math.atan2(r[1], r[0])
    s = +1.0 if ccw else -1.0
    chi_tan = theta + s*(math.pi/2.0)
    echi = _wrap(chi_tan - chi)
    rerr = abs(rho - R)
    return rho, rerr, chi_tan, echi

def mu_ff_circle(Vg, R, ccw, rerr, echi, prev_sign=None):
    # bank orbit ideal + bias radial halus + histeresis tanda
    g = 9.80665
    s = +1.0 if ccw else -1.0          # ccw True → +1
    mu_orbit = math.atan2(Vg*Vg, g*max(R, 1.0))  # >0
    # bias radial: dorong balik ke lingkaran (tanda mengikuti arah putar)
    bias = FF_R_BIAS * math.tanh(rerr / max(1e-6, 0.25*R))
    # histeresis tanda di sekitar tangent agar tidak flip
    sign = s
    if prev_sign is not None and abs(math.degrees(echi)) < FF_HYST_DEG:
        sign = prev_sign
    return sign*mu_orbit + sign*bias, sign


def rate_limit(u_now: float, u_prev: float, dps: float, Ts: float) -> float:
    du_max = math.radians(dps) * Ts
    du = max(-du_max, min(du_max, u_now - u_prev))
    return u_prev + du

def roll_deg_to_pwm_linear(roll_deg: float) -> int:
    # linear map: -BANK_LIMIT_DEG..+BANK_LIMIT_DEG → PWM_MIN..PWM_MAX (sekitar trim=1500)
    roll_deg = max(-BANK_LIMIT_DEG, min(BANK_LIMIT_DEG, roll_deg))
    k = (PWM_MAX - PWM_TRIM) / max(1.0, BANK_LIMIT_DEG)
    pwm = int(round(PWM_TRIM + k * roll_deg))
    return min(PWM_MAX, max(PWM_MIN, pwm))

def connect_mavlink(url: str) -> mavutil.mavfile:
    m = mavutil.mavlink_connection(url, autoreconnect=True, dialect="ardupilotmega")
    m.wait_heartbeat(timeout=30)
    print(f"[MAV] Heartbeat ok (sys={m.target_system} comp={m.target_component})")
    return m

_PLANE_MODES = {0:"MANUAL",1:"CIRCLE",2:"STABILIZE",5:"FBWA",6:"FBWB",7:"CRUISE",10:"AUTO",11:"RTL",12:"LOITER",15:"GUIDED"}
def decode_plane_mode(custom_mode: int) -> str: return _PLANE_MODES.get(int(custom_mode), f"MODE_{custom_mode}")
def set_mode(m: mavutil.mavfile, name: str) -> None:
    try: m.set_mode_apm(name)
    except Exception:
        m.mav.command_long_send(m.target_system, m.target_component,
                                mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                6 if name.upper()=="FBWB" else 0, 0,0,0,0,0)
    print(f"[MAV] set_mode request: {name}")

# ======================= State & shared ===============================
@dataclass
class NavState:
    N: float=0.0; E: float=0.0
    vN: float=0.0; vE: float=0.0
    Vg: float=0.0
    Va: float=0.0
    roll: float=0.0; pitch: float=0.0; yaw: float=0.0
    psi: float=0.0
    cog_deg: Optional[float]=None
    alt: float=0.0; vz: float=0.0
    throttle: float=0.0
    nav_pitch_deg: Optional[float]=None
    ts: float=0.0
    mode: str="UNKNOWN"
    wind_spd:float=float('nan'); wind_dir_rad:float=float('nan')

class Shared:
    def __init__(self):
        self.state = NavState(); self.state_lock = threading.Lock()
        self.cmd_roll = 0.0; self.cmd_lock = threading.Lock()
        self.stop = threading.Event()

        self.lat_mode = LATERAL_MODE
        self.hybrid_phase = "CAPTURE_L1"; self._blend_raw = 0.0; self.blend = 0.0
        self.uL1_last = 0.0; self.uMPC_last = 0.0

        self.mu_ff_sign = None

        self._mpc = None; self._path = None; self._l1 = None
        self.hz_rx = 0.0; self.hz_ctl = 0.0; self.hz_tx = 0.0

        self.qp_status=""; self.qp_iter=0; self.qp_fallback=False
        self.rho=float('nan'); self.rerr=float('nan'); self.echi_deg=float('nan')

# ======================= Threads ======================================
def state_reader(m: mavutil.mavfile, S: Shared):
    t0 = time.monotonic(); cnt = 0
    psi_lp = 0.0; psi_init = False; psi_ts = None
    while not S.stop.is_set():
        msg = m.recv_match(blocking=True, timeout=0.3)
        if msg is None: continue
        cnt += 1; tnow = time.monotonic(); tname = msg.get_type()
        with S.state_lock:
            st = S.state
            if tname=="HEARTBEAT":
                try: st.mode = decode_plane_mode(msg.custom_mode)
                except Exception: pass
                st.ts = tnow
            elif tname=="LOCAL_POSITION_NED":
                st.N=float(msg.x); st.E=float(msg.y)
                st.vN=float(msg.vx); st.vE=float(msg.vy)
                st.Vg=float(math.hypot(st.vN,st.vE))
                st.alt=float(-msg.z); st.vz=float(msg.vz); st.ts=tnow
            elif tname=="GLOBAL_POSITION_INT":
                try: st.cog_deg = float(msg.hdg/100.0) if msg.hdg!=65535 else st.cog_deg
                except Exception: pass
                st.alt=float(msg.alt)/1000.0; st.ts=tnow
            elif tname=="VFR_HUD":
                st.Va=float(msg.airspeed)
                try: st.throttle=float(msg.throttle)/100.0
                except Exception: pass
                st.ts=tnow
            elif tname=="WIND":
                try:
                    st.wind_spd=float(getattr(msg,"speed", float('nan')))
                    st.wind_dir_rad=math.radians(float(getattr(msg,"direction", 0.0)))
                except Exception: pass
            elif tname=="ATTITUDE":
                st.roll=float(msg.roll); st.pitch=float(msg.pitch); st.yaw=float(msg.yaw); st.ts=tnow
            elif tname=="NAV_CONTROLLER_OUTPUT":
                try: st.nav_pitch_deg=float(msg.nav_pitch)
                except Exception: pass
                st.ts=tnow

            # ψ estimate
            Vg_here = st.Vg if st.Vg else math.hypot(st.vN,st.vE)
            if Vg_here >= 0.5: psi_meas = math.atan2(st.vE, st.vN)
            elif st.cog_deg is not None: psi_meas = math.radians(st.cog_deg)
            else: psi_meas = st.yaw

            if not psi_init: psi_lp=psi_meas; psi_init=True; psi_ts=tnow
            # LPF berbasis dt aktual (discrete exact)
            dt  = max(1e-3, (tnow - psi_ts) if psi_ts is not None else Ts)
            tau = 0.15  # konstanta waktu filter (s)
            alpha = 1.0 - math.exp(-dt/tau)
            psi_lp = _wrap(psi_lp + alpha * _wrap(psi_meas - psi_lp))
            psi_ts = tnow

            st.psi = psi_lp
        if (time.monotonic()-t0)>=1.0:
            S.hz_rx = cnt / (time.monotonic()-t0); t0=time.monotonic(); cnt=0

def ctl_worker(m: mavutil.mavfile, S: Shared):
    MPC = _load_mpc_module()

    # ----- Build path & controllers from DRIVER PARAMETERS -----
    path = MPC.CirclePath(LOITER_C, LOITER_R, cw=LOITER_CW)
    l1_dir = (-1 if LOITER_CW else +1)
    l1 = MPC.L1Loiter(L1_PERIOD, L1_DAMPING, BANK_LIMIT_DEG, l1_dir)

    W = MPC.MPCWeights(MPC_W_ET, MPC_W_ECHI, MPC_W_MU, MPC_W_U, MPC_W_ET_T, MPC_W_ECHI_T)
    mpc = MPC.LTVMPC_OSQP(
        Ts=Ts, N=MPC_HORIZON, Va_init=MPC_VA_INIT, tau_mu=MPC_TAU_MU,
        bank_limit_deg=BANK_LIMIT_DEG, slew_limit_deg_s=MPC_SLEW_DEG_S,
        weights=W, path=path, w_du=MPC_W_DU, w_mu_Term=MPC_W_MU_TERM,
        use_groundspeed_mu_ss=MPC_USE_GS_FOR_MUss,
        Va_min=MPC_VA_MIN_MAX[0], Va_max=MPC_VA_MIN_MAX[1],
        Va_lp_tau=MPC_VA_LP_TAU, Va_nominal=MPC_VA_NOMINAL,
        use_w_du_scaling=MPC_USE_W_DU_SCALING
    )

    S._mpc = mpc; S._path = path; S._l1 = l1
    S.hybrid_phase = "CAPTURE_L1"; S._blend_raw=0.0; S.blend=0.0

    u_prev=0.0; phi_prev=0.0
    next_t=time.monotonic(); t0=time.monotonic(); cnt=0
    blend_rate = Ts / max(Ts, float(BLEND_SECONDS))
    k_step=0

    while not S.stop.is_set():
        next_t += Ts; time.sleep(max(0.0, next_t-time.monotonic())); cnt+=1; k_step+=1
        with S.state_lock: st = NavState(**vars(S.state))
        if (time.monotonic()-st.ts) > STALE_TIMEOUT: continue

        if USE_WIND_FEED and isinstance(st.wind_spd,float) and st.wind_spd==st.wind_spd:
            wind_NE = (st.wind_spd*math.cos(st.wind_dir_rad), st.wind_spd*math.sin(st.wind_dir_rad))
        else:
            wind_NE = (0.0, 0.0)

        Vg = st.Vg if st.Vg>0.05 else math.hypot(st.vN,st.vE)

        # --- ORBIT METRICS lebih dulu (untuk FF & logging) ---
        rho, rerr, _, echi = orbit_metrics(st.N, st.E, st.psi, path.C, path.R, ccw=(not path.cw))
        S.rho = rho; S.rerr = rerr; S.echi_deg = math.degrees(echi)

        # --- WARM-START untuk MPC: campur u_prev dengan feedforward bank orbit ---
        if FF_ENABLE:
            mu_ff, new_sign = mu_ff_circle(Vg, path.R, ccw=(not path.cw), rerr=rerr, echi=echi, prev_sign=S.mu_ff_sign)
            mu_ff = float(np.clip(mu_ff, -math.radians(MAX_BANK_DEG), math.radians(MAX_BANK_DEG)))
            S.mu_ff_sign = new_sign
            u_prev_warm = (1.0 - FF_ALPHA)*u_prev + FF_ALPHA*mu_ff
        else:
            u_prev_warm = u_prev

        # --- L1 & MPC commands (L1 biasa; MPC pakai u_prev_warm) ---
        xL1 = np.array([st.N, st.E, st.psi, u_prev, Vg, 0.5], float)
        uL1 = l1.command(xL1, path, Va_for_ctrl=Vg, wind=wind_NE)

        xM  = np.array([st.N, st.E, st.psi, u_prev], float)  # state tetap u_prev
        try:
            uMPC, info = mpc.step(
                xM, wind_NE, u_prev_warm,
                V_meas=st.Va if st.Va>0.1 else Vg
            )
            S.qp_status  = info.get("status","")
            S.qp_iter    = info.get("iter",0)
            S.qp_fallback= info.get("fallback",False)
        except Exception:
            uMPC = u_prev
            S.qp_status = "EXC"
            S.qp_fallback = True



        # === Select mixing ===
        if S.lat_mode == "l1":
            mu_raw = uL1; S.hybrid_phase="L1_ONLY"; S.blend=0.0
        elif S.lat_mode == "mpc":
            mu_raw = uMPC; S.hybrid_phase="MPC_ONLY"; S.blend=1.0
            if ASSIST_IN_MPC:
                far_r = rerr/max(1.0, path.R); far_h = abs(S.echi_deg)/max(1.0, ASSIST_HEAD_START_DEG)
                trigger = max(far_r-ASSIST_RERR_START, far_h-1.0)
                assist = 0.0 if trigger<=0.0 else min(1.0, trigger/1.0)
                if k_step <= max(1,int(ASSIST_BOOTSTRAP_STEPS)): assist = max(assist, 0.75)
                if S.qp_fallback: assist = max(assist, 0.85)
                w = _smoothstep(assist) * ASSIST_MAX_BLEND
                mu_raw = (1.0 - w)*uMPC + w*uL1
        else:
            # blend mode
            if S.hybrid_phase=="CAPTURE_L1":
                cond_r = (rerr < R_ERR_ENTER_FRAC*path.R)
                cond_h = (abs(S.echi_deg) < HEAD_ALIGN_DEG)
                ok_cnt = getattr(S,"_ok_cnt",0)
                ok_cnt = ok_cnt+1 if (cond_r and cond_h) else max(0, ok_cnt-1)
                S._ok_cnt=ok_cnt
                if ok_cnt>=HOLD_ENTER_STEPS: S.hybrid_phase="MPC_TRACK"
            else:
                if (rerr > R_ERR_EXIT_FRAC*path.R) or (abs(S.echi_deg) > HEAD_EXIT_DEG):
                    S.hybrid_phase="CAPTURE_L1"; S._ok_cnt=0
            target = 1.0 if S.hybrid_phase=="MPC_TRACK" else 0.0
            S._blend_raw += (blend_rate if target>S._blend_raw else -blend_rate)
            S._blend_raw = float(np.clip(S._blend_raw,0.0,1.0))
            s = _smoothstep(S._blend_raw); S.blend=s
            mu_raw = (1.0 - s)*uL1 + s*uMPC

        S.uL1_last=uL1; S.uMPC_last=uMPC

        # langsung pakai hasil mixing (limit sudah dijaga QP & L1)
        mu_cmd = float(mu_raw)
        u_prev = mu_cmd

        with S.cmd_lock: S.cmd_roll = mu_cmd

        if (time.monotonic()-t0)>=1.0:
            S.hz_ctl = cnt / (time.monotonic()-t0); t0=time.monotonic(); cnt=0

def att_sender(m: mavutil.mavfile, S: Shared, hz: float=TX_HZ):
    period = 1.0 / max(1e-3, hz)
    next_t=time.monotonic(); t0=time.monotonic(); cnt=0
    try:
        set_mode(m, "FBWB")
    except Exception:
        pass
    while not S.stop.is_set():
        next_t += period; time.sleep(max(0.0, next_t-time.monotonic())); cnt+=1
        with S.cmd_lock: roll_cmd = float(S.cmd_roll)
        with S.state_lock: st = NavState(**vars(S.state))
        if (time.monotonic()-st.ts) > STALE_TIMEOUT: continue
        # mapping linear dari BANK_LIMIT_DEG → PWM (tanpa shaping tambahan)
        pwm = roll_deg_to_pwm_linear(math.degrees(roll_cmd))       
        m.mav.rc_channels_override_send(
            m.target_system, m.target_component,
            pwm, 1500, 1500, 0, 0, 0, 0, 0     
        )
        if (time.monotonic()-t0)>=1.0:
            S.hz_tx = cnt / (time.monotonic()-t0); t0=time.monotonic(); cnt=0

def logger_thread(S: Shared, csv_path: str=CSV_PATH):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fields = [
        "t","mode_fc","lat_mode","hyb_phase","hyb_blend",
        "hz_rx","hz_ctl","hz_tx",
        "N","E","vN","vE","Vg","Va",
        "roll_deg","pitch_deg","yaw_deg","psi_deg",
        "cog_deg","alt","vz","thr",
        "rho","rerr","echi_deg",
        "cmd_roll_deg","uL1_deg","uMPC_deg",
        "qp_status","qp_iter","qp_fallback"
    ]
    with open(csv_path,"w",newline="") as f:
        w = csv.writer(f); w.writerow(fields)
        while not S.stop.is_set():
            time.sleep(0.05)
            with S.state_lock: st = NavState(**vars(S.state))
            with S.cmd_lock: cmd_roll = float(S.cmd_roll)
            row = [
                f"{time.monotonic()-BOOT_T0:.3f}", st.mode, S.lat_mode, S.hybrid_phase, f"{S.blend:.3f}",
                f"{S.hz_rx:.1f}", f"{S.hz_ctl:.1f}", f"{S.hz_tx:.1f}",
                f"{st.N:.2f}", f"{st.E:.2f}", f"{st.vN:.2f}", f"{st.vE:.2f}", f"{st.Vg:.2f}", f"{st.Va:.2f}",
                f"{math.degrees(st.roll):.2f}", f"{math.degrees(st.pitch):.2f}", f"{math.degrees(st.yaw):.2f}", f"{math.degrees(st.psi):.2f}",
                f"{st.cog_deg if st.cog_deg is not None else ''}", f"{st.alt:.1f}", f"{st.vz:.2f}", f"{st.throttle:.2f}",
                f"{S.rho:.2f}", f"{S.rerr:.2f}", f"{S.echi_deg:.1f}",
                f"{math.degrees(cmd_roll):.2f}", f"{math.degrees(S.uL1_last):.2f}", f"{math.degrees(S.uMPC_last):.2f}",
                S.qp_status, S.qp_iter, S.qp_fallback
            ]
            w.writerow(row)

# ======================= Main ========================================
def main():
    print("[RUN] URL:", MAVLINK_URL, " | MODE:", LATERAL_MODE.upper())
    print("[RUN] Expect MPC_15.py to be resolvable. If not, set MPC_MODULE_PATH inside this file.")
    m = connect_mavlink(MAVLINK_URL)
    S = Shared()
    th_rx = threading.Thread(target=state_reader, args=(m,S), daemon=True)
    th_ct = threading.Thread(target=ctl_worker,   args=(m,S), daemon=True)
    th_tx = threading.Thread(target=att_sender,   args=(m,S,TX_HZ), daemon=True)
    th_lg = threading.Thread(target=logger_thread, args=(S,CSV_PATH), daemon=True)
    th_rx.start(); th_ct.start(); th_tx.start(); th_lg.start()
    print(f"[RUN] Logging to: {CSV_PATH}")
    print("[RUN] FBWB + RC override CH1 (roll-only). Longitudinal by TECS on FC.")

    try:
        while True:
            time.sleep(1.0)
            with S.state_lock: st = NavState(**vars(S.state))
            with S.cmd_lock: u = float(S.cmd_roll)
            print(f"[STAT] fc={st.mode:>6s} | lat={S.lat_mode} phase={S.hybrid_phase} blend={S.blend:.2f} | "
                  f"RX={S.hz_rx:4.1f} CTL={S.hz_ctl:4.1f} TX={S.hz_tx:4.1f} | "
                  f"psi={math.degrees(st.psi):6.1f}° Va={st.Va:5.1f} Vg={st.Vg:5.1f} | "
                  f"rerr={S.rerr:.1f} echi={S.echi_deg:+5.1f}° | "
                  f"roll_cmd={math.degrees(u):+5.1f}° | "
                  f"qp:{S.qp_status} iters={S.qp_iter} fb={S.qp_fallback}")
    except KeyboardInterrupt:
        try: set_mode(m, "RTL")
        except Exception: pass
        print("\n[RUN] Stopping..."); S.stop.set(); time.sleep(0.3)

if __name__ == "__main__":
    main()
