# CLAUDE.md

Guidance for working in this repo. UAV **loiter (circular orbit) guidance**
study: compares several lateral controllers tracking a circle under wind, in a
shared closed-loop simulation.

## Running

Install once in editable mode so the packages are importable from anywhere
(replaces the old `PYTHONPATH=.` ritual):

```bash
pip install -e .                 # core sim
pip install -e ".[hardware]"     # + pymavlink/psutil for the SITL driver
```

Then run the sim:

```bash
python -m mpc_nav.main
```

`PYTHONPATH=.` still works if you prefer not to install:

```bash
# Git Bash
PYTHONPATH=. python -m mpc_nav.main
```
```powershell
# PowerShell
$env:PYTHONPATH = "."; python -m mpc_nav.main
```

- Entry point: [mpc_nav/main.py](mpc_nav/main.py) → build controllers → simulate
  → metrics → plots → animation.
- Artefacts land in a fresh `logs/run_<timestamp>/` (created on import of
  [mpc_nav/io_utils.py](mpc_nav/io_utils.py)).
- **For a fast iteration, disable the GIF** (animation dominates runtime):
  ```bash
  python -c "import mpc_nav.config as c; c.make_animation=False; import mpc_nav.main as m; m.main()"
  ```

Dependencies: see [requirements.txt](requirements.txt) / [pyproject.toml](pyproject.toml)
(`numpy scipy osqp matplotlib pandas`; `pymavlink psutil` only for the
[hardware/](hardware/) SITL driver, not the sim package).

Roll inner-loop sanity test: `python tests/test_lqr_roll.py` (step bank response).

## Architecture (`mpc_nav/`)

The sim runs **4 controllers in parallel on 4 independent plants** from the same
initial state, in one time loop ([simulation.py](mpc_nav/simulation.py)):

| Plant | Controller | File | Role |
|---|---|---|---|
| x1 | L1 nonlinear guidance | [l1_loiter.py](mpc_nav/l1_loiter.py) | baseline |
| x2 | LTV-MPC (OSQP QP)      | [ltv_mpc.py](mpc_nav/ltv_mpc.py) | main controller |
| x3 | PI on radial error    | [pi_loiter.py](mpc_nav/pi_loiter.py) | **naive negative baseline (divergent by design)** |
| x4 | PID (PI + radial-velocity damping) | [pid_loiter.py](mpc_nav/pid_loiter.py) | functional classical baseline |

Shared pieces:
- **Plant**: 7-state `[n, e, chi, mu, p, V, thr]` (N, E, course, bank, roll-rate,
  airspeed, throttle), integrated in [dynamics.py](mpc_nav/dynamics.py)
  (`rk4_step_long`).
- **Inner loop**: every outer controller outputs a **bank-angle command
  `mu_ref`**; [lateral_lqr.py](mpc_nav/lateral_lqr.py) (`LateralLQR`) converts it
  to aileron. See sub-stepping gotcha below.
- **Wind**: [wind.py](mpc_nav/wind.py) (`constant`/`rotating`/`gust`/etc.).
- **Geometry**: [geometry.py](mpc_nav/geometry.py) (`CirclePath`, cross-track
  helpers, start-state).
- **Outputs**: [metrics.py](mpc_nav/metrics.py) (CSVs **+ human-readable
  `metrics_report.md`**), [plotting.py](mpc_nav/plotting.py) (static PNGs + GIF).

**All tunable parameters live in [config.py](mpc_nav/config.py)** — there are no
CLI flags. Change behaviour there.

## Conventions & gotchas (learned the hard way)

- **Inner-loop sub-stepping is required.** The LQR closed-loop roll pole is
  ~-49 rad/s; at `Ts=0.1 s` that is under-sampled (|s·dt|≈5 > RK4 limit), giving
  a bank-tracking bias. The plant is integrated in `config.inner_loop_substeps`
  (=5) sub-steps of `Ts/5`, recomputing the aileron each one
  (`simulation._step_plant`). Don't remove this.
- **Operating point is `Va_ref = 21 m/s`.** `k_thrust=60` is sized for it
  (`k_thrust=9` capped airspeed at 10.6 m/s — old value). Two MPC params are
  tied to this speed and were re-tuned for it: `N_horizon=40` (shorter overshoots
  capture on the R=90 m orbit) and `w_du=300` (smaller makes the bank command
  rough). See [docs/mpc_retune_va21.md](docs/mpc_retune_va21.md).
- **PI is intentionally unstable** — kept as a documented negative baseline, not
  a bug to fix. See [docs/pi_baseline_notes.md](docs/pi_baseline_notes.md).
- **Windows console is cp1252** — don't `print()` non-ASCII (λ, →, °). It raises
  `UnicodeEncodeError`. File writes use `encoding="utf-8"` and are fine.
- **Markdown tables**: never put a raw `|` in a cell (e.g. `|e_t|`) — it breaks
  the table. Rephrase (see `metrics._fmt_table`).
- `legacy/MPC_v1/v2/v3.py` are the **old monolithic versions**, superseded
  by the `mpc_nav/` package. Edit the package, not those.

## Repo layout

- [mpc_nav/](mpc_nav/) — core closed-loop simulation library (controllers, plant,
  metrics, plotting). Self-contained; this is what you usually edit.
- [hardware/](hardware/) — MAVLink/SITL interface: `mavlink_driver.py` (lateral
  guidance driver, reuses `mpc_nav` controllers) and `rtl_monitor.py`.
- [scripts/](scripts/) — runnable tools: `sitl_auto_eval.py` (SITL evaluation,
  imports `hardware.mavlink_driver`), `tune_l1.py`, `plot_mavlog.py`.
- [tests/](tests/) — `test_lqr_roll.py`.
- [docs/](docs/) — design/tuning notes; `docs/refs/` holds reference PDFs (gitignored).
- [legacy/](legacy/) — archived monolithic versions; do not edit.

## Docs

- [docs/mpc_retune_va21.md](docs/mpc_retune_va21.md) — MPC horizon + `w_du`
  re-tuning at 21 m/s (diagnosis, sweeps, results).
- [docs/pi_baseline_notes.md](docs/pi_baseline_notes.md) — why naive PI diverges.
- [docs/sitl_test_protocol.md](docs/sitl_test_protocol.md) — SITL test protocol.

## Tuning workflow

Sweep a parameter by reloading the sim with overridden `config` values (see the
diagnosis scripts referenced in `docs/`). General pattern: set
`config.make_animation=False`, override the param, `importlib.reload` the
`simulation` module, run `build_controllers()` + `simulate()`, score from
`crosstrack_series`. Write best values back into `config.py` with a comment
explaining why.
