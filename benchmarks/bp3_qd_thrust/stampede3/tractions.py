"""Write mechanical and frictional shear (positive up dip) at the same accepted Q1 material IDs."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys


def frictional_shear(mechanical_pa, signed_rate, radiation):
    value = mechanical_pa - radiation * signed_rate
    if not math.isfinite(value):
        raise ValueError("Nonfinite frictional shear")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    path = parser.parse_args().profile
    settings = json.loads((path.parent / "parameters.json").read_text())
    radiation = float(settings["Material model"]["Phase field fault"]["Radiation damping"]["value"])
    writer = csv.writer(sys.stdout)
    writer.writerow(["accepted_clock_s", "id", "s_m", "rate_m_s", "shear_Pa", "frictional_shear_Pa",
                     "raw_compression_Pa", "friction_compression_Pa"])
    with path.open() as stream:
        clock = float(next(stream).split()[-1])
        next(stream)
        for row in csv.DictReader(stream, delimiter=" ", skipinitialspace=True):
            shear, rate = float(row["shear_Pa"]), float(row["rate_m_s"])
            writer.writerow([clock, row["id"], row["s_m"], rate, shear, frictional_shear(shear, rate, radiation),
                             row["raw_compression_Pa"], row["friction_compression_Pa"]])
