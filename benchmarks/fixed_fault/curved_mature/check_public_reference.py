"""Compare accepted curved-fault fields and the public label files with the frozen serial run."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


REFERENCE = Path(__file__).resolve().with_name("public-reference")
STEPS = range(3)
# Public column name and units, paired with the independently sampled label quantity.
LABEL_COLUMNS = (("x(m)", "x"), ("y(m)", "y"), ("age(s)", "Theta_s"),
                 ("slip(m)", "slip"), ("rate(m/s)", "V_m_s"))


def keyed(rows, fields):
    """Return numerical columns in records indexed by label ID or by (CellId, q)."""
    result = {}
    for row in rows:
        if fields == ("cell", "q"):
            key = row["cell"], int(row["q"])
        else:
            key = int(row[fields[0]])
        if key in result:
            raise ValueError(f"Repeated {fields} identity {key}")
        result[key] = {col: float(value) for col, value in row.items() if col not in fields}
    if not result:
        raise ValueError(f"No rows with {fields}")
    return result


def csv_rows(path, keys):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Missing or duplicate CSV header in {path}")
        return keyed(reader, keys)


def public_rows(path):
    names = ["label", *(col for col, _ in LABEL_COLUMNS)]
    with path.open() as stream:
        if stream.readline().split() != ["#", *names]:
            raise ValueError(f"Incorrect public label header or units in {path}; expected {names}")
        return keyed((dict(zip(names, line.split(), strict=True)) for line in stream if line.strip()),
                     ("label",))


def compare(expected, actual, description):
    """Use the established mature cross-build budget on each named physical column."""
    if expected.keys() != actual.keys():
        raise ValueError(f"{description}: different physical identities")
    ids = sorted(expected)
    columns = list(expected[ids[0]])
    if expected[ids[0]].keys() != actual[ids[0]].keys():
        raise ValueError(f"{description}: different columns")
    for col in columns:
        a = np.array([expected[i][col] for i in ids])
        b = np.array([actual[i][col] for i in ids])
        error = np.abs(b-a)
        failed = (~np.isfinite(a) | ~np.isfinite(b) | (error > 1e-10 + 1e-8*np.abs(a)))
        if np.any(failed):
            i = np.flatnonzero(failed)[0]
            raise ValueError(f"{description}, ID {ids[i]}, {col}: expected {a[i]:.17g}, "
                             f"measured {b[i]:.17g}, absolute error {error[i]:.5g}")


def read_clocks(output):
    return np.loadtxt(output/"statistics", usecols=(0, 1, 2), ndmin=2)


def check_public_reference(output, public_output=None):
    manifest = json.loads((REFERENCE/"provenance.json").read_text())
    if manifest["source_revision"] != "ae676e24095909196f885073b1146babce50292b":
        raise ValueError("Unexpected revision of the curved-fault reference")
    for filename, digest in manifest["files_sha256"].items():
        if hashlib.sha256((REFERENCE/filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Damaged frozen reference file {filename}")
    clocks = np.loadtxt(REFERENCE/"clocks.csv", delimiter=",", skiprows=1, ndmin=2)
    ranks = range(len(list(output.glob("material-0-rank*.csv"))))
    if not ranks:
        raise ValueError(f"No native field shards in {output}")
    expected_shards = {f"material-{step}-{stem}{rank}.csv"
                       for step in STEPS for stem in ("rank", "surface-") for rank in ranks}
    if {p.name for p in output.glob("material-*.csv")} != expected_shards:
        raise ValueError(f"Unexpected set of native steps or rank shards in {output}")
    public_dirs = [output] if public_output is None or public_output == output else [output, public_output]
    names = {f"fault-labels-{step:05}.txt" for step in STEPS}
    for directory in public_dirs:
        candidate = read_clocks(directory)
        np.testing.assert_array_equal(candidate[:, 0], clocks[:, 0], err_msg="accepted step IDs")
        np.testing.assert_allclose(candidate[:, 1:], clocks[:, 1:], rtol=0., atol=1e-15,
                                   err_msg=f"accepted clocks in {directory}")
        if {p.name for p in directory.glob("fault-labels-*.txt")} != names:
            raise ValueError(f"Expected the three accepted public label files in {directory}")

    previous_slip = {}
    for step in STEPS:
        stem = f"material-{step}"
        clock = f"{clocks[step, 1]:g} s"
        labels = csv_rows(REFERENCE/f"{stem}-surface-0.csv", ("i",))
        native = csv_rows(REFERENCE/f"{stem}-rank0.csv.gz", ("cell", "q"))
        observed = {}
        for rank in ranks:
            surface = csv_rows(output/f"{stem}-surface-{rank}.csv", ("i",))
            compare(labels, surface, f"{clock}, native labels on rank {rank}")
            if rank == 0:
                public_quantities = {i: {col: row[native_col] for col, native_col in LABEL_COLUMNS}
                                     for i, row in surface.items()}
            shard = csv_rows(output/f"{stem}-rank{rank}.csv", ("cell", "q"))
            if next(iter(shard.values())).keys() != next(iter(native.values())).keys():
                raise ValueError(f"{clock}: different native columns on rank {rank}")
            if observed.keys() & shard.keys():
                raise ValueError(f"{clock}: duplicate owned native sample")
            observed.update(shard)
        compare(native, observed, f"{clock}, native fields")

        filename = f"fault-labels-{step:05}.txt"
        frozen = public_rows(REFERENCE/filename)
        for directory in public_dirs:
            values = public_rows(directory/filename)
            compare(public_quantities, values, f"{clock}, {directory/filename} vs native label")
            compare(frozen, values, f"{clock}, frozen public labels")
            # The rate is the accepted positive endpoint speed. Multiplying by
            # the physical duration gives the increment of accumulated distance.
            expected_slip = {i: {"slip(m)": previous_slip.get((directory, i), 0.) +
                                            clocks[step, 2]*row["rate(m/s)"]} for i, row in values.items()}
            compare(expected_slip, {i: {"slip(m)": row["slip(m)"]} for i, row in values.items()},
                    f"{clock}, integrated public speed")
            previous_slip.update({(directory, i): row["slip(m)"] for i, row in values.items()})
        print(f"{clock}: {len(labels)} material IDs, {len(native)} native (CellId, q) samples, "
              f"public header, units and all five physical columns passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="accepted output with moving fault samples")
    parser.add_argument("--public-output", type=Path, help="also check the public run without a plugin")
    args = parser.parse_args()
    check_public_reference(args.output, args.public_output)
