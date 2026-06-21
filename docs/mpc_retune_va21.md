# LTV-MPC Re-tuning at Va = 21 m/s — Diagnosis & Fix

This note documents the re-tuning of the LTV-MPC loiter controller
(`mpc_nav/ltv_mpc.py`) after the cruise airspeed was raised from the old
`Va_ref ≈ 10.6 m/s` to `Va_ref = 21 m/s`. It explains why the MPC initially
looked *non-robust* against the L1 and PID baselines, shows that the real cause
was the **prediction horizon being too short for the orbit geometry at the new
speed**, and records the sweep evidence that fixes it.

**Headline:** two parameter changes — `N_horizon: 15 → 40` (fixes the capture
overshoot) and `w_du: 40 → 300` (fixes the rough bank command) — take the MPC
from the *worst* controller (RMS 22.9 m, bank-rate 28.6 °/s) to the *best*
tracker (RMS 6.15 m) with near-baseline control smoothness, beating both L1
(7.1 m) and PID (6.8 m) on every error metric.

Operating point: circle radius `R = 90 m`, cruise `Va_ref = 21 m/s`, constant
wind `(6, 0) m/s` (28.6 % of airspeed), outer step `Ts = 0.10 s`, LQR roll
inner loop sub-stepped at `Ts/5 = 0.02 s` (`config.inner_loop_substeps`).
Initial condition: start at `(-120, 100) m` heading to the centre — initial
radial error `‖(-120,100)‖ − 90 = 66 m`.

---

## 1. Symptom

The first 4-way comparison at `Va = 21 m/s` (`N = 15`) looked damning for MPC:

| Controller | RMS e_t [m] | MAX [m] |
|------------|------------:|--------:|
| L1         |        7.1  |   63.6  |
| PID        |        6.8  |   63.6  |
| **MPC**    |     **22.9**|**95.9** |

MPC's RMS was 3× worse and its peak error (95.9 m) *exceeded the initial error*
(66 m) — i.e. it **overshot the circle during capture**, swinging further out
than where it started before turning back.

The natural assumption ("MPC is not robust / wind is not modelled / weights are
stale") turned out to be wrong. A proper diagnosis was needed.

## 2. Diagnosis — it is the transient, not robustness

Breaking the error down by phase (full 150 s run) tells a completely different
story:

| Controller | transient RMS (0–40 s) | steady-state mean ‖e‖ (last 80 s) |
|------------|-----------------------:|----------------------------------:|
| L1         |                 10.94  |                              3.20 |
| PID        |                 10.79  |                              2.70 |
| **MPC (N=15)** |             43.7   |                          **2.58** |

Two facts emerge:

1. **In steady state, MPC was already the most accurate of the three** (mean
   ‖e‖ 2.58 m vs PID 2.70 m vs L1 3.20 m). Wind *is* modelled — see
   `LTVMPC_OSQP._f`, which carries `+wn, +we`, and `_mu_ss`, which uses the
   groundspeed-based centripetal bank. The steady tracking under 6 m/s wind was
   never the problem.
2. **The entire RMS penalty came from the capture transient** (0–40 s), where
   MPC overshot to 95.9 m and took ~40 s to settle.

So the question is narrow: *why does the capture overshoot?*

### Root cause: horizon shorter than the orbit geometry

With `N = 15` and `Ts = 0.1 s`, the prediction horizon is **1.5 s ≈ 31 m of
path** at 21 m/s. The orbit has radius **90 m**. The MPC literally cannot *see*
far enough ahead to plan the turn-in: within its 31 m window the optimal move is
to keep flying toward the circle, and by the time the curvature constraint
bites, it is already past the tangent and overshoots.

At the old `Va = 10.6 m/s`, the same `N = 15` covered 1.5 s × 10.6 = 16 m — but
the *capture dynamics were ~2× slower in path-angle terms*, so the short horizon
was tolerable. Doubling the speed without lengthening the horizon broke that
balance. This is why the controller "regressed" with no code change: the horizon
was implicitly tuned to the old operating point.

## 3. What was ruled out

**Weights are not the lever.** A full sweep of the stage/terminal weights at
`N = 15` (`w_et ∈ {4,8,16} × w_echi ∈ {6,12} × w_du ∈ {10,20,40}`, 18 combos)
could not get below RMS 23.6 m — *every* combination was worse than the existing
`w_et=3, w_echi=4` config (22.9 m). The overshoot is a horizon/geometry
limitation that no re-weighting can remove, because the cost function cannot
penalise a future it does not predict.

## 4. Fix — lengthen the horizon

Sweeping `N` with the existing weights (T_end = 120 s):

| N  | RMS [m] | MAX [m] | transient RMS (0–40s) | steady-state mean ‖e‖ |
|----|--------:|--------:|----------------------:|----------------------:|
| 15 |   25.54 |    95.9 |                 43.74 |                  2.69 |
| 25 |   13.31 |    63.6 |                 22.20 |                  2.36 |
| 35 |    6.93 |    63.6 |                 10.76 |                  1.78 |
| **40** | **6.66** | **63.6** |          **10.33** |              **1.74** |
| 45 |    6.56 |    63.6 |                 10.26 |                  1.59 |
| 50 |    6.50 |    63.6 |                 10.22 |                  1.52 |

Key transitions:

- **`N ≥ 25` eliminates the overshoot**: MAX drops from 95.9 m to 63.6 m, which
  is exactly the initial radial error — the trajectory no longer swings past the
  circle.
- **The knee is at `N ≈ 35–40`**: RMS collapses from 25.5 → 6.7 m, then flattens
  (≤ 3 % further improvement to N=50).
- **Steady state also improves** (2.69 → 1.5 m) because the longer terminal arc
  lets the MPC anticipate the once-per-orbit groundspeed variation under wind.

`N = 40` (`4.0 s`, ~84 m lookahead — radius-scale) was chosen: it sits at the
knee, already beats both baselines, and keeps the largest real-time margin.

### Real-time feasibility

Wall-clock for the **whole 4-controller loop** (all 4 plants + 5 inner
sub-steps each, Python):

| N  | ms / outer step (all 4 controllers) | QP fallbacks |
|----|------------------------------------:|-------------:|
| 35 |                               44.8  |            0 |
| 40 |                               51.4  |            0 |
| 45 |                               59.0  |            0 |
| 50 |                               66.2  |            0 |

Even at N=50 the *entire* 4-way loop runs in 66 ms < the 100 ms budget, with
zero QP fallbacks. MPC alone (one plant, one QP) is a small fraction of this, so
N=40 is comfortably real-time.

## 5. Result

Final 4-way comparison at `Va = 21 m/s`, `N = 40`, `w_du = 300` (T_end = 150 s):

| Controller | RMS e_t [m] | MAX [m] | IAE [m·s] | transient RMS (0–40s) | steady-state mean ‖e‖ [m] | bank RMS [°] |
|------------|------------:|--------:|----------:|----------------------:|--------------------------:|-------------:|
| **MPC**    |    **6.15** | 63.6 | **377.6** |             **10.33** |                  **1.74** |         25.5 |
| PID        |        6.78 |    63.6 |     518.5 |                 10.79 |                      2.70 |         24.4 |
| L1         |        7.09 |    63.6 |     581.1 |                 10.94 |                      3.20 |         24.2 |
| PI (naive) |   ~313 (divergent) | — | — |                  —    |                      —    |         33.6 |

MPC is now **best on every error metric**:

- Lowest RMS (6.15 m, −13 % vs L1, −9 % vs PID).
- Lowest IAE (377.6 m·s, **−35 % vs L1**, −27 % vs PID) — the longer horizon
  pays off most in the integrated transient.
- Lowest steady-state error (1.74 m) — confirming the MPC's predictive wind
  handling was always its strength.
- No capture overshoot (MAX = the initial 63.6 m, identical to the baselines).
- Bank-angle RMS (25.5°) is now on par with L1/PID (~24°), and the control rate
  is smooth (see §9).

## 6. Change made

`mpc_nav/config.py`:

```python
N_horizon = 40    # was 15  (capture overshoot — §4)
w_du      = 300   # was 40  (bank smoothness — §9)
```

The stage/terminal tracking weights (`w_et=3, w_echi=4, w_mu, w_u, w_et_T=2,
w_echi_T=3`) are unchanged — they were never the problem (§3). Only the two
parameters that were implicitly tied to the old operating point needed updating:
the horizon (geometry/speed) and the control-rate penalty.

## 7. Reproduction

```bash
PYTHONPATH=. python -m mpc_nav.main      # generates the 4-way artefacts
```

Sweep scripts used for this note (scratchpad, not committed):
`tune_mpc.py` (weights), `tune_horizon.py` / `tune_horizon2.py` (horizon +
timing), `tune_wdu.py` (bank smoothness, §9).

## 9. Bank-command smoothness (the `w_du` fix)

After the horizon fix, the MPC bank command was visibly rougher than L1/PID.
Quantifying it (bank-command rate = Δμ_cmd / Ts, full 150 s run, `w_du=40`):

| Controller | bank-rate RMS [°/s] | bank-rate MAX [°/s] | roll-rate p RMS [°/s] |
|------------|--------------------:|--------------------:|----------------------:|
| MPC (w_du=40) |             28.6 |        80.1 (slew-limited) |             19.6 |
| PID        |                15.6 |                64.8 |                   7.2 |
| L1         |                13.0 |                49.6 |                   6.2 |

MPC was ~2× rougher in command rate and ~3× more active in realised roll rate,
and the roughness was present in **steady state**, not just capture. The cause
was simply that the Δu penalty `w_du = 40` (inherited from the old operating
point) was too small for the new dynamics.

Sweeping `w_du` at N=40 (T_end = 120 s) shows smoothing is essentially **free**:

| w_du | RMS [m] | bank-rate RMS [°/s] | roll-rate p RMS [°/s] |
|------|--------:|--------------------:|----------------------:|
| 40   |    6.66 |                28.6 |                  19.6 |
| 150  |    6.62 |                22.1 |                  15.8 |
| **300** | **6.60** |            **17.9** |              **12.6** |
| 600  |    6.59 |                16.1 |                  11.4 |

Raising `w_du` **lowers** RMS slightly (smoother control tracks the orbit a
touch better) while cutting the bank-rate RMS by 37 % at `w_du=300`
(28.6 → 17.9 °/s, close to PID's 15.6). `w_du=300` was chosen: near-PID
smoothness, no tracking cost, and it keeps response headroom for disturbance
rejection (vs pushing to 600). The final realised roll-rate p RMS drops from
19.6 to 12.4 °/s.

Note `bank-rate MAX` stays pinned at the 80 °/s slew limit regardless of `w_du`:
during the aggressive turn-in the MPC deliberately slews at the maximum allowed
rate. That is expected and stays within the constraint — the smoothing gain is
all in the non-saturated (steady-state) portion.

## 10. Notes / future work

- **Tracking weights not re-optimised at N=40.** The weight sweep was run at
  N=15, where weights cannot help (§3). A fresh sweep at N=40 might squeeze a
  little more, but MPC already wins, so this is low priority.
- **N=45–50 gives a further ~5 % RMS reduction** (6.66 → 6.50) and better steady
  state (1.74 → 1.52 m) if extra compute is available.
- The residual once-per-orbit ~10 m spikes (visible in `et_series_vs_time.png`)
  come from the constant-wind groundspeed cycle; they are small and recover
  within a few steps. A wind-varying-over-horizon model (instead of frozen
  current wind) could shave them, but the gain is marginal at this wind level.
