"""End-to-end run: build controllers → simulate → metrics → plots → animation."""
from __future__ import annotations
import pandas as pd

from . import config
from .io_utils import run_tag, savepath
from .simulation import build_controllers, simulate
from .metrics import prepare_metrics_and_save
from .plotting import make_static_plots, animate_loiter


def main() -> None:
    path, l1_green, l1_orange, mpc, lateral_lqr = build_controllers()
    L1log, MPClog, HYBlog, QPlog = simulate(path, l1_green, l1_orange, mpc, lateral_lqr)
    aux = prepare_metrics_and_save(path, L1log, MPClog)
    pd.DataFrame(QPlog).to_csv(savepath("qp_stats.csv"), index=False)
    make_static_plots(path, L1log, MPClog, aux)

    if config.make_animation:
        animate_loiter(path, MPClog, L1log, config.Ts,
                       every_k=config.anim_every_kstep,
                       plane_size=config.plane_size_m,
                       duration_s=config.anim_duration_s,
                       save_gif=config.save_gif, save_mp4=config.save_mp4)

    print(f"Artifacts saved in: {run_tag}")
    print(" - traj_compare_with_wind.png, et_compare.png, bank_compare.png, et_series_vs_time.png")
    print(" - metrics_compare.csv, metrics_crosstrack.csv, et_series.csv, qp_stats.csv")
    if config.make_animation:
        suffix = " and uav_loiter_anim.mp4" if config.save_mp4 else ""
        print(f" - uav_loiter_anim.gif{suffix}")


if __name__ == "__main__":
    main()
