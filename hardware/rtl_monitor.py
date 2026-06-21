#!/usr/bin/env python3
"""
rtl_monitor_hardhome.py

RTL monitor that uses a hard-coded HOME latitude/longitude as the loiter center.

- Hard-coded HOME: lat=-7.903000, lon=110.291856 (user provided)
- Uses GLOBAL_POSITION_INT (preferred) to compute rho (meters) and rerr.
- Monitors HEARTBEAT mode and highlights when mode contains "RTL".
- Writes CSV (ts, mode, lat, lon, rho_m, rerr_m) and can optionally re-broadcast named_value_float 'RTL_RERR'.

Usage example:
  python3 /mnt/data/rtl_monitor_hardhome.py --conn udp:127.0.0.1:14550 --radius 100 --send-mavlink --log rtl_monitor_hardhome.csv
"""
import argparse
import csv
import math
import os
import time
from pymavlink import mavutil

# === USER HARD-CODED HOME (lat, lon) ===
HOME_LAT = -7.903000
HOME_LON = 110.291856

# approximate meters per degree latitude
M_PER_DEG_LAT = 111320.0

def meters_from_home(lat, lon, lat0=HOME_LAT, lon0=HOME_LON):
    """
    Convert lat/lon (degrees) to approximate local N (north) and E (east) offsets in meters
    relative to lat0, lon0 using a local equirectangular approximation.
    Good enough for distances under a few kilometers.
    """
    dlat = lat - lat0
    dlon = lon - lon0
    dN = dlat * M_PER_DEG_LAT
    dE = dlon * M_PER_DEG_LAT * math.cos(math.radians(lat0))
    return dN, dE

def safe_open_log(path):
    logdir = os.path.dirname(path) or "."
    os.makedirs(logdir, exist_ok=True)
    return open(path, "w", newline="")

def main():
    p = argparse.ArgumentParser(description="Monitor RTL rerr relative to hard-coded HOME lat/lon.")
    p.add_argument("--conn", "-c", required=True, help="MAVLink connection string (eg tcp:127.0.0.1:5760 or udp:127.0.0.1:14550)")
    p.add_argument("--radius", "-r", type=float, default=100.0, help="Desired loiter radius (meters) to compute rerr")
    p.add_argument("--log", "-o", default="rtl_monitor_hardhome.csv", help="CSV output path")
    p.add_argument("--send-mavlink", action="store_true", help="Send named_value_float RTL_RERR to the same link")
    args = p.parse_args()

    print(f"Using hard-coded HOME lat={HOME_LAT}, lon={HOME_LON}")
    print(f"Connecting to {args.conn} ...")
    m = mavutil.mavlink_connection(args.conn, autoreconnect=True)
    try:
        m.wait_heartbeat(timeout=10)
    except Exception:
        print("No heartbeat within timeout, continuing to listen for messages...")

    print("Listening for GLOBAL_POSITION_INT (preferred) and HEARTBEAT ...")

    last_named_send = 0.0
    last_print = 0.0
    current_mode = ""

    # CSV header
    f = safe_open_log(args.log)
    w = csv.writer(f)
    w.writerow(["ts", "mode", "lat_deg", "lon_deg", "rho_m", "rerr_m"])
    f.flush()

    try:
        while True:
            msg = m.recv_match(blocking=True, timeout=1.0)
            if msg is None:
                continue

            tnow = time.time()
            tname = msg.get_type()

            if tname == "HEARTBEAT":
                try:
                    current_mode = mavutil.mode_string_v10(msg)
                except Exception:
                    try:
                        current_mode = str(msg.custom_mode)
                    except Exception:
                        current_mode = ""

            # GLOBAL_POSITION_INT: use this for distance calc
            if tname == "GLOBAL_POSITION_INT":
                try:
                    lat = msg.lat / 1e7
                    lon = msg.lon / 1e7
                except Exception:
                    continue

                # convert to meters relative to hard-coded home
                dN, dE = meters_from_home(lat, lon)
                rho = math.hypot(dN, dE)
                rerr = rho - args.radius

                row_mode = current_mode or ""

                w.writerow([f"{tnow:.3f}", row_mode, f"{lat:.7f}", f"{lon:.7f}", f"{rho:.3f}", f"{rerr:.3f}"])
                f.flush()

                # print: highlight RTL
                if "RTL" in row_mode:
                    if (time.time() - last_print) >= 1.0:
                        print(f"[{time.strftime('%H:%M:%S')}] RTL | rho={rho:.2f} m | rerr={rerr:.2f} m | mode={row_mode}")
                        last_print = time.time()
                else:
                    if (time.time() - last_print) >= 2.0:
                        print(f"[{time.strftime('%H:%M:%S')}] mode={row_mode or 'UNKNOWN'} | rho={rho:.2f} m | rerr={rerr:.2f} m")
                        last_print = time.time()

                # send named_value_float every ~1s if requested
                if args.send_mavlink and (time.time() - last_named_send) >= 1.0:
                    try:
                        m.mav.named_value_float_send(
                            int((time.time()*1000) % 2**32),
                            b"RTL_RERR",
                            float(rerr)
                        )
                        last_named_send = time.time()
                    except Exception:
                        pass

            # If LOCAL_POSITION_NED messages are also available, optionally could be used,
            # but this script relies on GLOBAL_POSITION_INT because HOME is given in lat/lon.

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        try:
            f.close()
        except Exception:
            pass
        try:
            m.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
