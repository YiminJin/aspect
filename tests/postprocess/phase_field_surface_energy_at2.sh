#!/usr/bin/env python3
"""Compare the named statistics columns at 14 significant digits on both build types."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/phase_field_surface_energy"))
from check import statistics

values = statistics(Path("output-phase_field_surface_energy_at2/statistics"))
for term in ("local", "gradient", "total"):
    column = f"Crack surface energy {term} (J/m)"
    print(f"{column} {values[column]:.14g}")
