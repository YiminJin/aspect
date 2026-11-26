"""Exercise the installed historical comparison and the public ASCII table syntax."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("prefix", "plugin", "destination"):
        parser.add_argument(arg, type=lambda path: Path(path).resolve())
    parser.add_argument("--mpiexec", default="mpirun")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    source = args.prefix / "benchmarks/layered_moving_fault"
    env = os.environ.copy()
    env.pop("ASPECT_SOURCE_DIR", None)

    for case in ("documented", "changed-shear"):
        base = args.destination / case
        example = base / "benchmarks/layered_moving_fault"
        shutil.copytree(source, example)
        table = example / "layered_vertices.txt"
        records = (line.partition("#")[0].split() for line in table.read_text().splitlines())
        rows = [row for row in records if row]
        if case == "changed-shear":
            column = rows[0].index("background_shear_Pa")
            if rows[1][0] != "0":
                raise ValueError("Expected the first material ID to be zero")
            rows[1][column] = str(float(rows[1][column]) + .1)
        table.write_text("# SI units, persistent IDs\n\n" +
                         "\n".join("\t  ".join(row) + "  # vertex table" for row in rows) + "\n")
        run = base / "run"
        with (base / "check.log").open("w") as log:
            result = subprocess.run([sys.executable, "-B", example/"check_vertex_file.py",
                                     args.prefix/"bin/aspect-release", args.plugin, run,
                                     "--mpiexec", args.mpiexec], env=env, stdout=log, stderr=subprocess.STDOUT)
        public = run / "public-file/output"
        params = json.loads((public/"parameters.json").read_text())
        actual = params["Material model"]["Phase field fault"]["Fault vertex file"]["value"]
        if Path(actual) != table:
            raise ValueError(f"Public run read {actual}, expected {table}")
        if case == "documented":
            result.check_returncode()
            print("Commented table: historical inline, partition crossing and restart passed.", flush=True)
        else:
            message = (base/"check.log").read_text().splitlines()[-1]
            if result.returncode == 0 or not message.startswith("ValueError: V_m_s:"):
                raise ValueError(f"Expected an initial rate discrepancy; got {message}")
            rates = []
            for directory in (public, run/"historical-inline/output"):
                with (directory/"material-0-surface-0.csv").open() as stream:
                    row, = (r for r in csv.DictReader(stream) if r["i"] == "0")
                    rates.append(float(row["V_m_s"]))
            print(f"Changed ID 0 shear: {rates[0]:.15g} vs {rates[1]:.15g} m/s; {message}")


if __name__ == "__main__":
    main()
