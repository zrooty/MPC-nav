"""Static plots and animation for L1 vs MPC comparison."""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon

from . import config
from .io_utils import savefig_here, savepath
from .wind import wind_at, wind_mode_str


# ----------------------------------------------------------------------
# Static plots
# ----------------------------------------------------------------------
def make_static_plots(path, L1log: Dict[str, np.ndarray], MPClog: Dict[str, np.ndarray],
                      PIlog: Dict[str, np.ndarray], PIDlog: Dict[str, np.ndarray],
                      aux: Dict[str, Any]) -> None:
    et_L1, et_M, et_PID, t_arr = (aux["et_L1"], aux["et_MPC"],
                                    aux["et_PID"], aux["t"])

    # --- Trajectory with wind annotation
    fig, ax = plt.subplots(figsize=(6, 6))
    al = np.linspace(0, 2 * np.pi, 400)
    ax.plot(path.C[0] + path.R * np.cos(al),
            path.C[1] + path.R * np.sin(al),
            "--", lw=1.3, color="tab:gray", alpha=config.alpha_circle,
            label=f"Circle ref ({'CW' if path.cw else 'CCW'})")

    all_n = np.concatenate([MPClog["n"], L1log["n"], PIDlog["n"]])
    all_e = np.concatenate([MPClog["e"], L1log["e"], PIDlog["e"]])
    nxmin, nxmax = float(np.min(all_n)), float(np.max(all_n))
    exmin, exmax = float(np.min(all_e)), float(np.max(all_e))
    pad_n = 0.1 * (nxmax - nxmin + 1e-6)
    pad_e = 0.1 * (exmax - exmin + 1e-6)
    arrow_n = nxmin + pad_n
    arrow_e = exmax - pad_e

    t_samp = np.linspace(0, min(10.0, config.T_end), 30)
    wind_samples = np.array([wind_at(t) for t in t_samp])
    w_mean = wind_samples.mean(axis=0)
    wn_hist = np.array([wind_at(i * config.Ts)[0] for i in range(len(MPClog["chi"]))])
    we_hist = np.array([wind_at(i * config.Ts)[1] for i in range(len(MPClog["chi"]))])
    gs_hist = np.hypot(MPClog["V"] * np.cos(MPClog["chi"]) + wn_hist,
                       MPClog["V"] * np.sin(MPClog["chi"]) + we_hist)

    ax.quiver(arrow_n, arrow_e, w_mean[0], w_mean[1],
              angles="xy", scale_units="xy", scale=1,
              color="tab:blue", alpha=config.alpha_quiver, width=0.005)
    ax.text(0.02, 0.98,
            f"Wind ≈ {np.linalg.norm(w_mean):.1f} m/s\n"
            f"UAV speed avg ≈ {float(np.mean(gs_hist)):.1f} m/s\n"
            f"Wind mode: {wind_mode_str()}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color="k")

    ax.plot(MPClog["n"], MPClog["e"], label="MPC", color="tab:orange", alpha=config.alpha_trails)
    ax.plot(L1log["n"], L1log["e"], label="L1", color="tab:green", alpha=config.alpha_trails)
    ax.plot(PIDlog["n"], PIDlog["e"], label="PID", color="tab:purple", alpha=config.alpha_trails)
    ax.set_aspect("equal")
    ax.set_xlabel("North [m]"); ax.set_ylabel("East [m]")
    ax.grid(True, alpha=0.4)
    leg = ax.legend(loc="upper right"); leg.set_alpha(1.0)
    savefig_here("traj_compare_with_wind.png")

    # --- |e_t| vs step
    plt.figure()
    plt.plot(et_M, label="MPC e_t", color="tab:orange", alpha=0.9)
    plt.plot(et_L1, label="L1 e_t", color="tab:green", alpha=0.9)
    plt.plot(et_PID, label="PID e_t", color="tab:purple", alpha=0.9)
    plt.xlabel("Step"); plt.ylabel("Cross-track error [m]")
    plt.legend(); plt.grid(True)
    savefig_here("et_compare.png")

    # --- Bank angles
    plt.figure()
    plt.plot(np.degrees(MPClog["u_cmd"]), label="MPC mu_ref [deg]", color="tab:orange", alpha=0.9)
    plt.plot(np.degrees(MPClog["mu"]),    label="MPC mu [deg]",     color="tab:orange", ls=":", alpha=0.9)
    plt.plot(np.degrees(L1log["u_cmd"]), "--", label="L1 mu_ref [deg]", color="tab:green", alpha=0.8)
    plt.plot(np.degrees(L1log["mu"]),   ":", label="L1 mu [deg]",     color="tab:green", alpha=0.8)
    plt.plot(np.degrees(PIDlog["u_cmd"]), "--", label="PID mu_ref [deg]", color="tab:purple", alpha=0.8)
    plt.plot(np.degrees(PIDlog["mu"]),   ":", label="PID mu [deg]",     color="tab:purple", alpha=0.8)
    plt.xlabel("Step"); plt.ylabel("Bank angle [deg]")
    plt.legend(); plt.grid(True)
    savefig_here("bank_compare.png")

    # --- e_t vs time
    plt.figure()
    plt.plot(t_arr, et_M, label="MPC e_t", color="tab:orange", alpha=0.9)
    plt.plot(t_arr, et_L1, label="L1 e_t", color="tab:green", alpha=0.9)
    plt.plot(t_arr, et_PID, label="PID e_t", color="tab:purple", alpha=0.9)
    plt.axhline(0, ls="--", lw=0.8)
    plt.xlabel("Time [s]"); plt.ylabel("Cross-track error e_t [m]")
    plt.legend(); plt.grid(True)
    savefig_here("et_series_vs_time.png")


# ----------------------------------------------------------------------
# Animation helpers
# ----------------------------------------------------------------------
def _make_aircraft_patch(ax, size: float = 10.0, color: str = "tab:orange", zorder: int = 5):
    base = np.array([[+1.2, 0.0], [-1.0, +0.6], [-0.6, 0.0], [-1.0, -0.6]], dtype=float) * size
    patch = Polygon(base, closed=True, facecolor=color, edgecolor="k", lw=0.8,
                    zorder=zorder, animated=True, antialiased=True)
    patch._base_shape = base.copy()
    ax.add_patch(patch)
    return patch


def _set_pose_patch(patch, n: float, e: float, chi: float) -> None:
    base = patch._base_shape
    R = np.array([[np.cos(chi), -np.sin(chi)],
                  [np.sin(chi),  np.cos(chi)]], dtype=float)
    verts = base @ R.T + np.array([n, e], dtype=float)
    patch.set_xy(verts)


def animate_loiter(path, MPClog: Dict[str, np.ndarray], L1log: Dict[str, np.ndarray],
                   PIlog: Dict[str, np.ndarray], PIDlog: Dict[str, np.ndarray],
                   Ts: float, every_k: int = 2, plane_size: float = 12.0,
                   duration_s: float = 20.0, save_gif: bool = True,
                   save_mp4: bool = False) -> None:
    # Auto subsample to ~600 frames
    total = len(MPClog["n"])
    target_frames = max(300, int(duration_s * 20))  # ~20 fps visual density
    every_k = max(1, total // target_frames) if total > target_frames else max(1, every_k)

    idx = np.arange(0, total, every_k)
    n_nm, e_nm, chi_nm = MPClog["n"][idx], MPClog["e"][idx], MPClog["chi"][idx]
    mu_nm, p_nm, V_nm = MPClog["mu"][idx], MPClog["p"][idx], MPClog["V"][idx]
    n_l1, e_l1, chi_l1 = L1log["n"][idx], L1log["e"][idx], L1log["chi"][idx]
    n_pd, e_pd, chi_pd = PIDlog["n"][idx], PIDlog["e"][idx], PIDlog["chi"][idx]
    frames = len(idx)
    fps = min(30, max(5, int(frames / max(1, duration_s))))

    fig, ax = plt.subplots(figsize=(6, 6))
    al = np.linspace(0, 2 * np.pi, 400)
    ax.plot(path.C[0] + path.R * np.cos(al),
            path.C[1] + path.R * np.sin(al),
            "--", lw=1.3,
            label="Circle ref ({})".format("CW" if path.cw else "CCW"),
            color="tab:gray", alpha=config.alpha_circle)
    trail_nm, = ax.plot([], [], "-", lw=2.0, color="tab:orange",
                        alpha=config.alpha_trails, label="MPC", animated=True)
    trail_l1, = ax.plot([], [], "-", lw=2.0, color="tab:green",
                        alpha=config.alpha_trails, label="L1", animated=True)
    trail_pd, = ax.plot([], [], "-", lw=2.0, color="tab:purple",
                        alpha=config.alpha_trails, label="PID", animated=True)

    plane_nm = plane_l1 = plane_pd = None
    if config.show_planes:
        plane_nm = _make_aircraft_patch(ax, size=plane_size, color="tab:orange", zorder=6)
        plane_l1 = _make_aircraft_patch(ax, size=plane_size, color="tab:green", zorder=6)
        plane_pd = _make_aircraft_patch(ax, size=plane_size, color="tab:purple", zorder=6)

    all_n = np.concatenate([MPClog["n"], L1log["n"], PIDlog["n"]])
    all_e = np.concatenate([MPClog["e"], L1log["e"], PIDlog["e"]])
    pad = 0.2 * max(path.R, np.std(all_n) + np.std(all_e))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(all_n)) - pad, float(np.max(all_n)) + pad)
    ax.set_ylim(float(np.min(all_e)) - pad, float(np.max(all_e)) + pad)
    ax.set_xlabel("North [m]"); ax.set_ylabel("East [m]")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right")

    # Dynamic wind indicator (arrow + text)
    nxmin, nxmax = ax.get_xlim()
    exmin, exmax = ax.get_ylim()
    arrow_n = nxmin + 0.10 * (nxmax - nxmin)
    arrow_e = exmax - 0.10 * (exmax - exmin)
    wn0, we0 = wind_at(0.0, state=(n_nm[0], e_nm[0], chi_nm[0], mu_nm[0]))
    vg0_n = V_nm[0] * np.cos(chi_nm[0]) + wn0
    vg0_e = V_nm[0] * np.sin(chi_nm[0]) + we0
    gs0 = float(np.hypot(vg0_n, vg0_e))

    wind_arrow = ax.quiver(arrow_n, arrow_e, wn0, we0,
                           angles="xy", scale_units="xy", scale=1,
                           color="tab:blue", alpha=config.alpha_quiver, width=0.006,
                           zorder=7, animated=True)
    mode_str = wind_mode_str()
    wind_text = ax.text(0.02, 0.98,
                        f"Wind = {np.hypot(wn0, we0):.1f} m/s\n"
                        f"UAV speed = {gs0:.1f} m/s\n"
                        f"Wind mode: {mode_str}",
                        transform=ax.transAxes, ha="left", va="top", fontsize=9, color="k")
    wind_text.set_animated(True)

    def init():
        trail_nm.set_data([], [])
        trail_l1.set_data([], [])
        trail_pd.set_data([], [])
        if config.show_planes:
            _set_pose_patch(plane_nm, n_nm[0], e_nm[0], chi_nm[0])
            _set_pose_patch(plane_l1, n_l1[0], e_l1[0], chi_l1[0])
            _set_pose_patch(plane_pd, n_pd[0], e_pd[0], chi_pd[0])
            return (trail_nm, trail_l1, trail_pd, wind_arrow, wind_text,
                    plane_nm, plane_l1, plane_pd)
        return trail_nm, trail_l1, trail_pd, wind_arrow, wind_text

    def update(i):
        nonlocal wind_arrow
        trail_nm.set_data(n_nm[:i + 1], e_nm[:i + 1])
        trail_l1.set_data(n_l1[:i + 1], e_l1[:i + 1])
        trail_pd.set_data(n_pd[:i + 1], e_pd[:i + 1])
        if config.show_planes:
            _set_pose_patch(plane_nm, n_nm[i], e_nm[i], chi_nm[i])
            _set_pose_patch(plane_l1, n_l1[i], e_l1[i], chi_l1[i])
            _set_pose_patch(plane_pd, n_pd[i], e_pd[i], chi_pd[i])
        t_i = float(idx[i]) * Ts
        wn, we = wind_at(t_i, state=(n_nm[i], e_nm[i], chi_nm[i], mu_nm[i]))
        try:
            wind_arrow.set_UVC(wn, we)
        except Exception:
            try:
                wind_arrow.remove()
            except Exception:
                pass
            wind_arrow = ax.quiver(arrow_n, arrow_e, wn, we,
                                   angles="xy", scale_units="xy", scale=1,
                                   color="tab:blue", alpha=config.alpha_quiver,
                                   width=0.006, zorder=7, animated=True)
        vg_n = V_nm[i] * np.cos(chi_nm[i]) + wn
        vg_e = V_nm[i] * np.sin(chi_nm[i]) + we
        gs = float(np.hypot(vg_n, vg_e))
        wind_text.set_text(
            f"Wind = {np.hypot(wn, we):.1f} m/s\n"
            f"UAV speed = {gs:.1f} m/s\n"
            f"Wind mode: {mode_str}"
        )
        if config.show_planes:
            return (trail_nm, trail_l1, trail_pd, wind_arrow, wind_text,
                    plane_nm, plane_l1, plane_pd)
        return trail_nm, trail_l1, trail_pd, wind_arrow, wind_text

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=frames, interval=1000 / fps, blit=True)
    if save_gif:
        try:
            anim.save(savepath("uav_loiter_anim.gif"), writer="pillow", fps=fps)
        except Exception as e:
            print("GIF save failed:", e)
    if save_mp4:
        try:
            anim.save(savepath("uav_loiter_anim.mp4"), writer="ffmpeg", fps=fps, dpi=160)
        except Exception as e:
            print("MP4 save failed:", e)
    plt.close(fig)
