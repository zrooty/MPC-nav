# SITL Test Protocol — Lateral Loiter Controllers (ArduPilot Plane)

How to test the `hardware/mavlink_driver.py` controllers on ArduPilot SITL **one at a
time**, fairly and reproducibly. The sim runs 4 controllers in parallel on
identical plants; SITL can only run **one controller per flight**, so a
standardised scenario is what makes runs comparable.

## 0. Why the sim tuning is only a starting point

The driver's operating point differs from the sim, and the plant is the real
autopilot (not the sim's LQR inner loop). See [docs/mpc_retune_va21.md](mpc_retune_va21.md)
for the sim tuning. Key differences:

| | Sim (`config.py`) | Driver (SITL) |
|---|---|---|
| Radius R | 90 m | 180 m |
| Va | 21 m/s | ~22 m/s |
| Inner loop | LQR (`tau_mu=0.25`) | **Autopilot FBWB roll loop** |
| Wind | known exactly | **estimated** (`vg − va`, LPF) |
| Plants | 4 in parallel | 1 per flight |

The driver params (`L1_PERIOD`, `MPC_HORIZON`, `MPC_TAU_MU`, …) are **scaled
starting points**, all editable at the top of `hardware/mavlink_driver.py`. Re-tune here.

## 1. Launch SITL

```bash
# ArduPilot Plane SITL (from the ardupilot checkout)
sim_vehicle.py -v ArduPlane --console --map
# Driver connects to tcp:127.0.0.1:5760 (MAVLINK_URL in the driver).
```

Optional: set a repeatable wind so runs compare apples-to-apples:

```
# In the SITL/MAVProxy console:
param set SIM_WIND_SPD 6      # m/s
param set SIM_WIND_DIR 90     # deg
```

## 2. Standardise the scenario (do this for EVERY run)

So L1 vs MPC vs blend are comparable, fix:

- **Same start**: arm + takeoff (or AUTO to a fixed altitude), then begin guidance
  from roughly the same position/heading relative to the orbit
  (`LOITER_C=(0,0)`, `LOITER_R=180`).
- **Same wind**: identical `SIM_WIND_SPD`/`SIM_WIND_DIR`.
- **Same duration**: e.g. ≥4 full orbits after capture.
- **Same altitude**: TECS holds it; pick one FBWB altitude.

## 3. Identify the autopilot roll time constant (do ONCE, before MPC tuning)

`MPC_TAU_MU` must match the **FBWB roll closed-loop**, not the sim LQR. In FBWB,
command a bank step (e.g. via the driver in `l1` mode forcing a constant bank, or
a manual RC1 step) and read the `roll_deg` column in the CSV. Fit a first-order
response `phi(t) = phi_cmd·(1 − e^(−t/tau))`; the time to ~63% of the commanded
bank is `tau`. Set `MPC_TAU_MU` to that value. This is the single biggest lever
on MPC fidelity.

## 4. Test order (simplest → hardest)

Edit `LATERAL_MODE` at the top of the driver, restart, fly the standard scenario:

1. **`l1`** — capture + hold. Validates path geometry, RC1 sign, wind estimate.
   Confirm `rerr` settles and `roll_cmd` isn't saturating/chattering.
2. **`pid`** — functional classical baseline (radial PI + `rdot` damping). Should
   hold the circle once captured; compare its steady `rerr`/bank-rate to L1.
3. **`pi`** — naive radial PI. **Expected to diverge / limit-cycle** (heading-blind,
   no rate damping) — it's a documented negative baseline, not a bug. Fly it briefly
   only to confirm the divergence, then RTL. Keep an eye on `rerr` growing.
4. **`mpc`** — tracker only. Needs step 3 of §3 (tau ID) done first. Watch
   `qp_status` (should be `solved`), `qp_fallback` (should stay False), and
   `exec_time_ms` (must stay well under `Ts*1000` = 100 ms; else lower `MPC_HORIZON`).
5. **`blend`** — L1→MPC gated hand-off. Only after L1 and MPC are each solid.

All five controllers are logged every loop (`uL1_deg`, `uMPC_deg`, `uPI_deg`,
`uPID_deg`), so you can see what the non-active controllers *would* have commanded.
Only the selected one is sent to the FC.

## 5. First-flight sanity checklist

- [ ] Heartbeat OK, `fc` mode shows `FBWB` in the `[STAT]` line.
- [ ] **RC1 sign correct**: positive bank command → aircraft rolls the way that
      reduces `rerr` (turns toward the circle). If reversed, negate in
      `roll_deg_to_pwm_linear` or set `RC1_REVERSED`.
- [ ] `psi` (course estimate) tracks reasonably (driver derives it from velocity).
- [ ] Wind estimate (`wn_est/we_est`) converges, not wildly noisy.
- [ ] `exec_time_ms` < ~100 ms (for MPC/blend).
- [ ] On Ctrl-C the driver requests `RTL`.

## 6. Metrics (compare to the sim's, from the CSV)

The driver logs `rerr`, `echi_deg`, `cmd_roll_deg`, `qp_*`, `exec_time_ms`, etc.
Score each run with the same metrics the sim uses (see `mpc_nav/metrics.py`):

- **RMS `rerr`** over the steady window (last N orbits) — tracking accuracy.
- **Settle time** — first time `rerr` stays < 5 m.
- **Bank-rate RMS** — `diff(cmd_roll_deg)/Ts`; smoothness.
- **Peak `rerr`** — capture overshoot.

Tune one parameter at a time, write the winning value back into the driver with
a comment explaining why (same discipline as `config.py`).
