"""Kumpulan override konfigurasi untuk pipeline MPC."""

from __future__ import annotations

from typing import Any, Dict

# Konfigurasi simulasi Python murni menggunakan default dari pipeline.
PYTHON_SIM_CONFIG: Dict[str, Any] = {}

# Konfigurasi SITL/headless mematikan animasi & ekspor GIF.
SITL_CONFIG: Dict[str, Any] = {
    "make_animation": False,
    "save_gif": False,
}
