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


def prepare_metrics_and_save(path: CirclePath,
                             L1log: Dict[str, np.ndarray],
                             MPClog: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Compute and persist L1-vs-MPC comparison metrics."""
    et_L1 = crosstrack_series(path, L1log["n"], L1log["e"])
    et_M = crosstrack_series(path, MPClog["n"], MPClog["e"])

    metrics = {
        "RMS_e_t_L1": _rms(et_L1),
        "RMS_e_t_MPC": _rms(et_M),
        "settle_s_L1(|e_t|<5m)": _settle_time(et_L1, 5.0),
        "settle_s_MPC(|e_t|<5m)": _settle_time(et_M, 5.0),
        "RMS_bank_L1_deg": _rms(np.degrees(L1log["mu"])),
        "RMS_bank_MPC_deg": _rms(np.degrees(MPClog["mu"])),
        "Mean_V_L1": float(np.mean(L1log["V"])),
        "Mean_V_MPC": float(np.mean(MPClog["V"])),
    }
    pd.DataFrame([metrics]).to_csv(savepath("metrics_compare.csv"), index=False)

    t_arr = np.arange(len(MPClog["n"])) * config.Ts
    pd.DataFrame({
        "t_s": t_arr,
        "e_t_MPC": et_M,
        "e_t_L1": et_L1,
        "V_MPC": MPClog["V"],
        "V_L1": L1log["V"],
        "thr_MPC": MPClog["thr"],
        "thr_L1": L1log["thr"],
        "p_MPC": MPClog["p"],
        "p_L1": L1log["p"],
        "abs_e_t_MPC": np.abs(et_M),
        "abs_e_t_L1": np.abs(et_L1),
    }).to_csv(savepath("et_series.csv"), index=False)

    pd.DataFrame([{
        "RMS_e_t_MPC": _rms(et_M),
        "IAE_e_t_MPC": float(np.sum(np.abs(et_M)) * config.Ts),
        "MAX_e_t_MPC": float(np.max(np.abs(et_M))),
        "RMS_e_t_L1": _rms(et_L1),
        "IAE_e_t_L1": float(np.sum(np.abs(et_L1)) * config.Ts),
        "MAX_e_t_L1": float(np.max(np.abs(et_L1))),
    }]).to_csv(savepath("metrics_crosstrack.csv"), index=False)

    return {"et_L1": et_L1, "et_MPC": et_M, "t": t_arr, "metrics": metrics}
