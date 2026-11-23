#!/usr/bin/env python3
"""Compare the physical AT2 energies on each of the three limited phase fields."""
from pathlib import Path

lines = Path("output-explicit_dg_phase/statistics").read_text().splitlines()
indices = {line.split(": ",1)[1]: int(line.split(":")[0][1:])-1 for line in lines if line.startswith("#")}
columns = [indices[name] for name in ("Time (seconds)", "Crack surface energy local (J/m)",
                                      "Crack surface energy gradient (J/m)", "Crack surface energy total (J/m)")]
for line in lines:
    if line.strip() and not line.startswith("#"):
        values = line.split()
        print(" ".join(f"{float(values[i]):.10f}" for i in columns))
