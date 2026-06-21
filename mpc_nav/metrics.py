"""Comparison metrics between L1 and MPC controllers + CSV exports."""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd

from . import config
from .geometry import CirclePath, crosstrack_series
from .io_utils import savepath


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a ** 2)))


def _settle_time(e: np.ndarray, tol: float = 5.0) -> float:
    for i, v in enumerate(e):
        if abs(v) <= tol:
            return i * config.Ts
    return config.T_end


# ----------------------------------------------------------------------
# Markdown report
# ----------------------------------------------------------------------
def _controller_row(et: np.ndarray, log: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Per-controller metric set used by the markdown report."""
    ae = np.abs(et)
    n_trans = min(len(et), int(round(40.0 / config.Ts)))   # first 40 s
    n_ss = min(len(et), int(round(80.0 / config.Ts)))      # last 80 s
    bank_rate = np.diff(np.degrees(log["u_cmd"])) / config.Ts if len(log["u_cmd"]) > 1 else np.array([0.0])
    return {
        "RMS": _rms(et),
        "MAX": float(ae.max()),
        "IAE": float(np.sum(ae) * config.Ts),
        "settle": _settle_time(et, 5.0),
        "trans": _rms(et[:n_trans]),
        "ss": float(np.mean(ae[-n_ss:])),
        "bank_rms": _rms(np.degrees(log["mu"])),
        "bank_rate": _rms(bank_rate),
        "meanV": float(np.mean(log["V"])),
    }


def _fmt_table(rows: Dict[str, Dict[str, float]]) -> str:
    """Render a markdown table; bold the best (lowest) value per metric column.

    ``rows`` maps controller name -> metric dict. Controllers whose RMS is
    non-finite or absurd (divergent, e.g. naive PI) are excluded from the
    'best' comparison so they never win a column.
    """
    cols = [
        ("RMS",       "RMS e_t [m]"),
        ("MAX",       "Peak abs error [m]"),
        ("IAE",       "IAE [m·s]"),
        ("settle",    "Settle <5m [s]"),
        ("trans",     "Transient RMS 0–40s [m]"),
        ("ss",        "Steady mean abs-err last 80s [m]"),
        ("bank_rms",  "Bank RMS [°]"),
        ("bank_rate", "Bank-rate RMS [°/s]"),
        ("meanV",     "Mean V [m/s]"),
    ]
    names = list(rows.keys())
    # 'best' = lowest, considering only non-divergent controllers (RMS < 100 m)
    stable = [n for n in names if rows[n]["RMS"] < 100.0]
    best = {key: (min(rows[n][key] for n in stable) if stable else None)
            for key, _ in cols}

    header = "| Metric | " + " | ".join(names) + " |"
    sep = "|" + "---|" * (len(names) + 1)
    lines = [header, sep]
    for key, label in cols:
        cells = []
        for n in names:
            v = rows[n][key]
            s = f"{v:.2f}"
            if best[key] is not None and n in stable and abs(v - best[key]) < 1e-9:
                s = f"**{s}**"
            cells.append(s)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_md_report(path: CirclePath, ets: Dict[str, np.ndarray],
                     logs: Dict[str, Dict[str, np.ndarray]]) -> None:
    from .wind import wind_mode_str
    rows = {name: _controller_row(ets[name], logs[name]) for name in logs}
    wn, we = config.wind_mean
    wind_spd = float(np.hypot(wn, we))
    md = f"""# Loiter Controller Comparison — Metrics Report

**Operating point**

| Parameter | Value |
|---|---|
| Cruise airspeed `Va_ref` | {config.Va_ref:.1f} m/s |
| Circle radius `R` | {config.circle_R:.0f} m |
| Direction | {"CW" if config.cw else "CCW"} |
| Wind | {wind_spd:.1f} m/s ({wind_mode_str()}) |
| Sim duration `T_end` | {config.T_end:.0f} s |
| Outer step `Ts` | {config.Ts:.2f} s |
| MPC horizon `N` | {config.N_horizon} ({config.N_horizon * config.Ts:.1f} s) |

**Comparison** (best stable value per row in **bold**; lower is better for all error metrics)

{_fmt_table(rows)}

**Legend**

- **RMS e_t** — root-mean-square cross-track error over the whole run.
- **Peak abs error** — peak cross-track error (capture overshoot shows up here).
- **IAE** — integral of absolute error (sum of abs error x Ts); overall tracking effort.
- **Settle <5m** — first time the error stays/falls below 5 m.
- **Transient / Steady** — RMS over the first 40 s vs mean abs error over the last 80 s.
- **Bank RMS / Bank-rate RMS** — control magnitude and smoothness (lower rate = smoother).
"""
    with open(savepath("metrics_report.md"), "w", encoding="utf-8") as f:
        f.write(md)


def prepare_metrics_and_save(path: CirclePath,
                             L1log: Dict[str, np.ndarray],
                             MPClog: Dict[str, np.ndarray],
                             PIlog: Dict[str, np.ndarray],
                             PIDlog: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Compute and persist L1-vs-MPC-vs-PI-vs-PID comparison metrics."""
    et_L1 = crosstrack_series(path, L1log["n"], L1log["e"])
    et_M = crosstrack_series(path, MPClog["n"], MPClog["e"])
    et_PI = crosstrack_series(path, PIlog["n"], PIlog["e"])
    et_PID = crosstrack_series(path, PIDlog["n"], PIDlog["e"])

    metrics = {
        "RMS_e_t_L1": _rms(et_L1),
        "RMS_e_t_MPC": _rms(et_M),
        "RMS_e_t_PI": _rms(et_PI),
        "RMS_e_t_PID": _rms(et_PID),
        "settle_s_L1(|e_t|<5m)": _settle_time(et_L1, 5.0),
        "settle_s_MPC(|e_t|<5m)": _settle_time(et_M, 5.0),
        "settle_s_PI(|e_t|<5m)": _settle_time(et_PI, 5.0),
        "settle_s_PID(|e_t|<5m)": _settle_time(et_PID, 5.0),
        "RMS_bank_L1_deg": _rms(np.degrees(L1log["mu"])),
        "RMS_bank_MPC_deg": _rms(np.degrees(MPClog["mu"])),
        "RMS_bank_PI_deg": _rms(np.degrees(PIlog["mu"])),
        "RMS_bank_PID_deg": _rms(np.degrees(PIDlog["mu"])),
        "Mean_V_L1": float(np.mean(L1log["V"])),
        "Mean_V_MPC": float(np.mean(MPClog["V"])),
        "Mean_V_PI": float(np.mean(PIlog["V"])),
        "Mean_V_PID": float(np.mean(PIDlog["V"])),
    }
    pd.DataFrame([metrics]).to_csv(savepath("metrics_compare.csv"), index=False)

    t_arr = np.arange(len(MPClog["n"])) * config.Ts
    pd.DataFrame({
        "t_s": t_arr,
        "e_t_MPC": et_M,
        "e_t_L1": et_L1,
        "e_t_PI": et_PI,
        "e_t_PID": et_PID,
        "V_MPC": MPClog["V"],
        "V_L1": L1log["V"],
        "V_PI": PIlog["V"],
        "V_PID": PIDlog["V"],
        "thr_MPC": MPClog["thr"],
        "thr_L1": L1log["thr"],
        "thr_PI": PIlog["thr"],
        "thr_PID": PIDlog["thr"],
        "p_MPC": MPClog["p"],
        "p_L1": L1log["p"],
        "p_PI": PIlog["p"],
        "p_PID": PIDlog["p"],
        "abs_e_t_MPC": np.abs(et_M),
        "abs_e_t_L1": np.abs(et_L1),
        "abs_e_t_PI": np.abs(et_PI),
        "abs_e_t_PID": np.abs(et_PID),
    }).to_csv(savepath("et_series.csv"), index=False)

    pd.DataFrame([{
        "RMS_e_t_MPC": _rms(et_M),
        "IAE_e_t_MPC": float(np.sum(np.abs(et_M)) * config.Ts),
        "MAX_e_t_MPC": float(np.max(np.abs(et_M))),
        "RMS_e_t_L1": _rms(et_L1),
        "IAE_e_t_L1": float(np.sum(np.abs(et_L1)) * config.Ts),
        "MAX_e_t_L1": float(np.max(np.abs(et_L1))),
        "RMS_e_t_PI": _rms(et_PI),
        "IAE_e_t_PI": float(np.sum(np.abs(et_PI)) * config.Ts),
        "MAX_e_t_PI": float(np.max(np.abs(et_PI))),
        "RMS_e_t_PID": _rms(et_PID),
        "IAE_e_t_PID": float(np.sum(np.abs(et_PID)) * config.Ts),
        "MAX_e_t_PID": float(np.max(np.abs(et_PID))),
    }]).to_csv(savepath("metrics_crosstrack.csv"), index=False)

    # Human-readable markdown report (4-way, with best-per-metric highlighting)
    _write_md_report(
        path,
        ets={"MPC": et_M, "PID": et_PID, "L1": et_L1, "PI": et_PI},
        logs={"MPC": MPClog, "PID": PIDlog, "L1": L1log, "PI": PIlog},
    )

    return {"et_L1": et_L1, "et_MPC": et_M, "et_PI": et_PI, "et_PID": et_PID,
            "t": t_arr, "metrics": metrics}
