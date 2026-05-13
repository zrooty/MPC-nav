"""Output directory helpers. Creates a timestamped run folder on import."""
from __future__ import annotations
import os
from datetime import datetime
import matplotlib.pyplot as plt

run_tag = datetime.now().strftime("logs/run_%Y%m%d-%H%M%S")
os.makedirs(run_tag, exist_ok=True)


def savepath(name: str) -> str:
    return os.path.join(run_tag, name)


def savefig_here(name: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(run_tag, name), dpi=160)
