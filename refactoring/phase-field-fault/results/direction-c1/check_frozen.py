"""Verify the first frozen strip-8 turn and its accepted arclength on two ranks.

Use the donor replay, ordinary production output and both instrumented output
directories. The instrumented production and candidate runs differ only in
which score is allowed to update the plane. Hash the *inputs* to the first
decision before comparing either selector's accepted trajectory.
"""
import argparse
import csv
import hashlib
import math
import tarfile
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_common_files(before, after, names):
    for name in names:
        left = (before / name).read_bytes()
        require(left == (after / name).read_bytes(), f"Data differ in {name}")
        print(f"  {hashlib.sha256(left).hexdigest()}  {name}")


def check_historical_snapshot(archive_path, production):
    """Compare the earlier 5A debugger capture at the same two-rank first turn."""
    stage_count = point_count = 0
    with tarfile.open(archive_path) as archive:
        def archived_rows(filename):
            return [line.decode().split() for line in archive.extractfile(f"T2/{filename}")]

        for rank in range(2):
            # MidpointSystem stores the selected entries in increasing local
            # FESystem index. The accepted FE export records the corresponding
            # component and basis index in that order. Translation to those
            # keys avoids relying on the traversal order of cells or ranks.
            local_basis = defaultdict(list)
            for row in read_rows(production / f"fields-0000-r{rank}.csv"):
                component = int(row["component"])
                if component in (0, 1, 2, 7):  # Q2 displacement, Q1 pressure, Q2 phase in onecut.prm
                    local_basis[row["cell"]].append((component, int(row["local"])))
            historical = {(cell, int(c), int(j)): float(value)
                          for cell, c, j, value in archived_rows(f"stage-{rank}")}
            stage = {}
            for line in (production / f"turn-fields-{rank}").read_text().splitlines():
                cell, index, value = line.split()
                c, j = local_basis[cell][int(index)]
                stage[cell, c, j] = float(value)
            require(stage == historical, f"Historical stage differs on rank {rank}")
            stage_count += len(stage)

            def point_values(rows):
                return {(row[0], int(row[1])): tuple(map(float, row[2:8])) for row in rows}

            points = point_values(line.split() for line in
                                  (production / f"turn-points-{rank}").read_text().splitlines())
            require(points == point_values(archived_rows(f"points-{rank}")),
                    f"Six first-turn native inputs differ from historical rank {rank}")
            point_count += len(points)
            actual = (production / f"turn-sums-{rank}").read_text().splitlines()[1].split()
            raw, = archived_rows("live-81")
            centered, = archived_rows("live-27")
            require(list(map(float, actual)) == list(map(float, raw + centered)),
                    f"Rank {rank} differs from the historical MPI receive buffers")
    require((stage_count, point_count) == (992, 180), "Incomplete historical snapshot")
    print("Matched 992 stage values, six inputs at 180 keyed native points and twelve historical moments.")


def check_frozen(donor, ordinary, production, candidate, destination, historical):
    require(ordinary.is_dir() and production.is_dir() and candidate.is_dir(),
            "Provide the output directories of the three ordinary parameter-file runs")
    check_historical_snapshot(historical, production)
    print("Identical production and candidate accepted parent, solved midpoint and first-decision inputs")
    check_common_files(production, candidate, ["sections-0000.csv"] +
                       [name for rank in range(2) for name in (
                           f"fields-0000-r{rank}.csv", f"points-0000-r{rank}.csv",
                           f"turn-fields-{rank}", f"turn-points-{rank}")])
    for rank in range(2):
        # The first two rows store the reduction inputs/local sums and output.
        # Later rows contain evaluations by the selected scoring formula.
        name = f"turn-sums-{rank}"
        left, right = ((directory / name).read_bytes().splitlines()[:2] for directory in (production, candidate))
        require([len(row.split()) for row in left] == [9, 12] and left == right,
                f"Different or incomplete first-decision moments on rank {rank}")

    # The capture must observe the ordinary solve without moving its accepted
    # fields. Check all output rows on each owner, including the first parent.
    print("Identical accepted exports with and without production capture")
    check_common_files(ordinary, production, [
        f"{field}-{step:04d}-r{rank}.csv"
        for step in range(4) for rank in range(2) for field in ("points", "fields")
    ] + [f"sections-{step:04d}.csv" for step in range(4)] +
                       [f"balance-{step:04d}.csv" for step in (1, 2, 3)])

    incoming = read_rows(ordinary / "sections-0000.csv")
    require([int(row["section"]) for row in incoming] == list(range(9)), "Invalid seed strip IDs")
    require(all(float(row["beta"]) == 0 and int(row["formed"]) == 0 and float(row["slip"]) == 0
                and float(row["age"]) == 10 for row in incoming), "Invalid virgin seed history")
    width = float(incoming[8]["ds"])

    first_proposals = []
    for publish_policy, directory in enumerate((production, candidate)):
        decisions = read_rows(directory / "turn-decisions.csv")
        # Strip 7 is visited before strip 8. Although its two proposals
        # differ, even the whole positive peak score is below the minimum
        # required increase from the (nonnegative) incoming score.
        expected_visits = [(k, policy) for k in (7, 8) for policy in (0, 1)]
        require([(int(row["k"]), int(row["policy"])) for row in decisions] == expected_visits * 2 and
                all(int(row["step"]) == 1 for row in decisions),
                "Unexpected decision sequence before or after the turn")
        strip7 = [row for row in decisions if int(row["k"]) == 7]
        require(all(float(row["incoming"]) == 0 and row["accepted"] == "0" and
                    0 <= float(row["score"]) < float(row["update_threshold"]) for row in strip7),
                "An earlier strip-7 proposal could have changed the material plane")
        require(strip7[0]["angle"] != strip7[1]["angle"], "Expected distinct rejected strip-7 proposals")
        strip8 = [row for row in decisions if int(row["k"]) == 8]
        first = [row for row in strip8 if float(row["incoming"]) == 0.]
        require([(int(row["k"]), int(row["policy"])) for row in first] == [(8, 0), (8, 1)],
                "Unexpected order or repeated decision on the virgin plane")
        require(all(row["accepted"] == "1" and float(row["pressure"]) == 200. for row in first),
                "First decision did not update under constant 200 Pa compression")
        selected = first[publish_policy]
        # In any subsequent selection within the same nonlinear solve, the
        # consumed result must keep the original update.
        later = [row for row in strip8 if int(row["policy"]) == publish_policy
                 and float(row["incoming"]) != 0.]
        require(later and all(row["incoming"] == selected["angle"] and row["accepted"] == "0" for row in later),
                "Expected a second evaluation retaining the same angle")
        first_proposals.append(first)
    require(first_proposals[0] == first_proposals[1],
            "First-run and second-run selector results differ at the common state")

    results = []
    first_angle = {}
    donor_clocks = read_rows(donor / "tip-time-location.csv")
    for step in (1, 2, 3):
        expected_clock = 0.005 * step
        donor_sections = read_rows(donor / f"tip-planes-{step:04d}.csv")
        require([int(row["section"]) for row in donor_sections] == list(range(9)), "Invalid donor IDs")
        reference = donor_sections[8]
        require(float(reference["reference_dxi"]) == width, "Different undeformed strip widths")
        for method, run in (("donor", None), ("production", production), ("candidate", candidate)):
            if run is None:
                sections = donor_sections
                timestamp = donor_clocks[step-1]
                require(int(timestamp["step"]) == step, "Missing donor clock")
                clock = float(timestamp["endpoint_time"])
            else:
                balance, = read_rows(run / f"balance-{step:04d}.csv")
                clock = float(balance["time"])
                sections = read_rows(run / f"sections-{step:04d}.csv")
                require([int(row["section"]) for row in sections] == list(range(9)), "Invalid target IDs")
                require(all(row["formed"] == ref["formed"] for row, ref in zip(sections, donor_sections)),
                        "Formation differs")
                require(all(row["beta"] == ref["beta"] for row, ref in zip(sections[:8], donor_sections[:8])),
                        "Another strip turned")

            row = sections[8]
            require(clock == expected_clock, f"Unexpected accepted time for {method} step {step}")
            angle = float(row["beta"])
            ds = float(row["ds"])
            require(int(row["formed"]) == 1 and angle != 0, "First turn did not form strip 8")
            if step == 1:
                first_angle[method] = angle
                if run is not None:
                    decision = first_proposals[0][int(method == "candidate")]
                    require(float(decision["angle"]) == angle and float(decision["ds"]) == ds,
                            "The first selected angle or arclength differs from the published value")
            require(angle == first_angle[method], f"A formed angle changed at step {step}")
            predicted = width / math.cos(angle)
            require(abs(predicted - ds) <= math.ulp(ds), f"Arclength disagrees with geometry for {method}")
            reference_length = float(reference["ds"])
            reference_angle = float(reference["beta"])
            results.append(dict(step=step, time_s=clock, method=method, width_m=width,
                                angle_rad=angle, angle_minus_donor_rad=angle-reference_angle,
                                angle_allowance_rad=1e-7 + 1e-7*abs(reference_angle),
                                accepted_length_m=ds, geometric_length_m=predicted,
                                length_minus_donor_m=ds-reference_length,
                                length_allowance_m=1e-11 + 1e-7*abs(reference_length)))

    with destination.open("w", newline="") as stream:
        table = csv.DictWriter(stream, fieldnames=results[0], lineterminator="\n")
        table.writeheader()
        table.writerows(results)
    print(f"Arclength reproduction at all three clocks written to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for item in ("donor", "ordinary", "production", "candidate", "destination"):
        parser.add_argument(item, type=Path)
    parser.add_argument("--historical", type=Path,
                        default=Path(__file__).resolve().parents[2]/"strip8"/"evidence.tar.gz")
    check_frozen(**vars(parser.parse_args()))
