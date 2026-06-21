# MPC-nav

UAV **loiter (circular orbit) guidance** study: several lateral controllers
track a circle under wind in a shared closed-loop simulation, plus a
MAVLink/SITL driver to fly the same controllers on ArduPilot.

## Install

```bash
pip install -e .                 # core simulation
pip install -e ".[hardware]"     # + pymavlink/psutil for the SITL driver
```

(Or skip install and use `PYTHONPATH=.` — see [CLAUDE.md](CLAUDE.md).)

## Run the simulation

```bash
python -m mpc_nav.main
```

Artefacts land in `logs/run_<timestamp>/` (cross-track metrics, `metrics_report.md`,
PNGs, and a GIF). For fast iteration, disable the GIF:

```bash
python -c "import mpc_nav.config as c; c.make_animation=False; import mpc_nav.main as m; m.main()"
```

## Layout

| Path | What |
|---|---|
| [mpc_nav/](mpc_nav/) | Core simulation library (controllers, plant, metrics, plotting) |
| [hardware/](hardware/) | MAVLink/SITL driver (`mavlink_driver.py`) and `rtl_monitor.py` |
| [scripts/](scripts/) | Tools/diagnostics: `sitl_auto_eval.py`, `tune_l1.py`, `plot_mavlog.py`, `lqr_roll_step.py` |
| [docs/](docs/) | Design/tuning notes; `docs/refs/` reference PDFs |
| [legacy/](legacy/) | Archived monolithic versions — do not edit |

Controllers compared: **L1** (baseline), **LTV-MPC** (main, OSQP),
**PI** (intentionally divergent negative baseline), **PID** (functional classical
baseline). See [CLAUDE.md](CLAUDE.md) for architecture, conventions, and gotchas.
