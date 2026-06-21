"""Entry point for the MPC navigation simulation.

The implementation lives in the ``mpc_nav`` package; this file just keeps the
historical ``python MPC_v3.py`` invocation working.

Module layout
-------------
* ``mpc_nav.config``       — all tunable simulation parameters
* ``mpc_nav.geometry``     — angle wrapping, ``CirclePath``, orbit metrics
* ``mpc_nav.wind``         — wind field models
* ``mpc_nav.dynamics``     — 7-state plant + RK4 integrator
* ``mpc_nav.lateral_lqr``  — 6-state inner-loop LQR
* ``mpc_nav.l1_loiter``    — L1 loiter controller
* ``mpc_nav.ltv_mpc``      — LTV-MPC (OSQP) outer loop
* ``mpc_nav.simulation``   — controller wiring and time-stepping loop
* ``mpc_nav.metrics``      — RMS/IAE/MAX and CSV exports
* ``mpc_nav.plotting``     — static plots and loiter animation
* ``mpc_nav.io_utils``     — timestamped output folder helpers
* ``mpc_nav.main``         — end-to-end run
"""
from mpc_nav.main import main

if __name__ == "__main__":
    main()
