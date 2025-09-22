"""Varian SITL/headless untuk pipeline MPC berbasis :mod:`MPC_v1`.

Modul ini mewarisi seluruh pipeline dari :mod:`MPC_v1` tetapi otomatis
menerapkan override ``SITL_CONFIG`` dari :mod:`config_sets`. Versi ini
menonaktifkan animasi dan ekspor GIF sehingga ringan saat dipakai bersama
SITL atau sistem headless. Jalankan ``python MPC_v2.py`` untuk menjalankan
simulasi dengan konfigurasi tersebut; gunakan ``python MPC_v1.py`` untuk
varian penuh dengan animasi.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from config_sets import SITL_CONFIG
from MPC_v1 import *  # noqa: F401,F403
from MPC_v1 import configure as _base_configure
from MPC_v1 import main as _base_main


def configure(overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Terapkan konfigurasi SITL dan override tambahan jika diperlukan."""

    merged = dict(SITL_CONFIG)
    if overrides:
        merged.update(overrides)
    return _base_configure(merged)


def main(config_overrides: Mapping[str, Any] | None = None):
    """Jalankan simulasi dengan konfigurasi SITL (+override opsional)."""

    merged = dict(SITL_CONFIG)
    if config_overrides:
        merged.update(config_overrides)
    return _base_main(merged)


# Terapkan konfigurasi SITL segera setelah impor agar nilai global konsisten.
configure()


__all__ = [name for name in globals().keys() if not name.startswith("_")]


if __name__ == "__main__":
    main()
