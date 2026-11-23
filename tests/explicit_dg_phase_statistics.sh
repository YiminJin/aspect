#!/usr/bin/env python3
"""Read pointwise extrema and integrated mass from the ordinary postprocessors."""
from pathlib import Path

lines = Path("output-explicit_dg_phase_statistics/statistics").read_text().splitlines()
names = [line.split(": ", 1)[1] for line in lines if line.startswith("#")]
columns = [names.index(name) for name in ("Time (seconds)", "Maximal temperature (K)",
                                         "Minimal value for composition damage", "Maximal value for composition damage",
                                         "Global mass for composition damage")]
for line in lines:
    if line.strip() and not line.startswith("#"):
        print(" ".join(f"{float(line.split()[i]):.8f}" for i in columns))
