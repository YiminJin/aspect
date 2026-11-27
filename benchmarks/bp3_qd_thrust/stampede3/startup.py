"""Check the historical 60-degree full-grid run before continuing past step 3."""
import csv
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET

from history import read_statistics, slip_budget

OUTPUT = Path("historical-output")
VERTICES = Path("historical-vertices.txt")
PROFILE_HEADER = "id s_m x_m y_m rate_m_s age_s slip_m shear_Pa raw_compression_Pa friction_compression_Pa normal_integral_m"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checkpoint():
    restart = OUTPUT / "restart"
    slot = int((restart / "last_good_checkpoint.txt").read_text())
    require(slot > 0, "Invalid native checkpoint pointer")
    directory = restart / f"{slot:02d}"
    for name in ("mesh", "resume.z"):
        require((directory / name).stat().st_size > 0, f"Empty {directory/name}")
    info = dict(line.split(None, 1) for line in (directory / "metadata.txt").read_text().splitlines())
    step, clock = int(info["timestep_number"]), float(info["time"])
    require(step > 0 and math.isfinite(clock), f"Invalid checkpoint metadata in {directory}")
    return step, clock


def read_profile(step, clock, expected_ids):
    path = OUTPUT / f"fault-profile-{step:07d}.txt"
    with path.open() as stream:
        stamp = stream.readline().split()
        require(stamp[:2] == ["#", "accepted_clock_s"] and float(stamp[2]) == clock, f"Clock mismatch in {path}")
        stream.readline()
        require(stream.readline().split() == PROFILE_HEADER.split(), f"Unexpected profile columns in {path}")
        rows = [tuple(map(float, line.split())) for line in stream if line.strip()]
    require(len(rows) == expected_ids, f"Expected all {expected_ids} IDs in {path}")
    for i, row in enumerate(rows):
        require(len(row) == 11 and all(map(math.isfinite, row)) and row[0] == i
                and min(row[4], row[5], row[9], row[10]) > 0, f"Invalid material ID {i} in {path}")
    return rows


def check_cuts(top):
    path = OUTPUT / "startup-cuts.txt"
    with path.open() as stream:
        cuts = {}
        for row in csv.DictReader(stream, delimiter=" ", skipinitialspace=True):
            values = {k: float(v) for k, v in row.items()}
            require(all(map(math.isfinite, values.values())), f"Nonfinite startup cut in {path}")
            cuts.setdefault(values["y_m"], []).append((values["r_m"], values["chi_per_m"]))
    require(sorted(cuts) == [0, top/2, top], "Missing contact or mid-depth cut")
    for y, cut in cuts.items():
        require(len(cut) == 513 and cut[0][0] < -39.529 and cut[-1][0] > 39.529, f"Incomplete cut at y={y}")
        require(all(a[0] < b[0] for a, b in zip(cut, cut[1:])), f"Disordered cut at y={y}")
        mass = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(cut, cut[1:]))
        print(f"y={y:g} m, integrated slip density={mass:.8g}")
        require(abs(mass - 1) < .01, f"Cut integral outside the resolved-mesh 1% budget at y={y}")


def check_native(clocks):
    frames = ET.parse(OUTPUT / "solution.pvd").findall(".//DataSet")
    require(len(frames) == len(clocks), "Expected one bulk frame at each startup clock")
    paths = set()
    for frame, clock in zip(frames, clocks):
        # The PVD time label can be rounded by deal.II. Profiles retain 17 digits.
        require(math.isclose(float(frame.attrib["timestep"]), clock, rel_tol=5e-6), f"Wrong bulk clock near {clock}")
        pvtu = OUTPUT / frame.attrib["file"]
        require(pvtu not in paths, f"Repeated bulk frame {pvtu}")
        paths.add(pvtu)
        grid = ET.parse(pvtu)
        fields = {a.get("Name") for a in grid.findall(".//PPointData/PDataArray")}
        require({"p", "velocity", "phase_field", "fault_tau_xx", "fault_tau_xy", "fault_tau_yy"} <= fields,
                f"Missing native fields in {pvtu}")
        pieces = [pvtu.parent / p.attrib["Source"] for p in grid.findall(".//Piece")]
        require(pieces and len(set(pieces)) == len(pieces) and all(p.stat().st_size > 0 for p in pieces),
                f"Missing or empty native pieces for {pvtu}")


def check_mesh(path, counts, domain):
    with path.open() as stream:
        row, = csv.DictReader(stream, delimiter=" ")
    require({int(row["cells"])} == counts, f"Mesh inventory disagrees with statistics: {counts}")
    print(f"Actual run mesh has {row['cells']} cells")
    bounds = tuple(float(row[k]) for k in ("xmin", "xmax", "ymin", "ymax"))
    require(bounds == domain and float(row["fine_m"]) == 3.90625, f"Wrong domain or resolution: {row}")
    require(abs(float(row["area_m2"])-(bounds[1]-bounds[0])*(bounds[3]-bounds[2])) < .01,
            "Mesh does not cover the box")
    for region in ("support", "base", "surface"):
        require(int(row[region]) > 0 and int(row["bad_"+region]) == 0, f"Underresolved or missing {region}")
    require(int(row["bad_coarse"]) == 0, "Cells exceed the outer grid spacing")


def verify(target, domain=(-60000., 90000., 0., 50000.), expected_ids=5776):
    records = list(read_statistics(OUTPUT / "statistics"))
    steps = [int(row["Time step number"]) for row in records]
    require(steps == list(range(target + 1)), f"Expected accepted steps 0..{target}, got {steps}")
    clocks = [float(row["BP3 clock (s)"]) for row in records]
    require(clocks[0] == 0 and all(a < b for a, b in zip(clocks, clocks[1:])), "Invalid accepted clocks")
    counts = {int(row["Number of mesh cells"]) for row in records}
    check_mesh(OUTPUT / "bp3-mesh.txt", counts, domain)
    next_step, next_time = checkpoint()
    # ASPECT advances time and step number before writing the restart metadata.
    require(next_step == target + 1 and next_time > clocks[-1], "Checkpoint does not continue the last accepted state")
    with VERTICES.open() as stream:
        next(stream)
        next(stream)
        reference = [tuple(map(float, line.split())) for line in stream if line.strip()]
    require(len(reference) == expected_ids, "Incomplete historical vertex table")
    settings = json.loads((OUTPUT / "parameters.json").read_text())
    dc = float(settings["Material model"]["Phase field fault"]["Slip distance"]["value"])
    control = settings["Time stepping"]["Fault RSF"]
    age_cap = float(control["Maximum logarithmic state change"]["value"])
    previous = None
    for step, (record, clock) in enumerate(zip(records, clocks)):
        rows = read_profile(step, clock, expected_ids)
        peak = max(rows, key=lambda r: r[4])
        require(peak[4] == float(record["BP3 peak speed (m/s)"]) and
                abs(peak[1] - float(record["BP3 peak distance (m)"])) < 1e-7, f"Peak mismatch at step {step}")
        dt = float(record["BP3 interval (s)"])
        if step:
            require(dt > 0 and math.isclose(clocks[step-1] + dt, clock, rel_tol=2e-15),
                    f"Clock disagrees with accepted interval at step {step}")
        max_log = max_slip = 0.
        for i, row in enumerate(rows):
            identity, x, y, age0 = reference[i]
            require(identity == i and abs(row[1]-math.hypot(x, y-domain[3])) < 1e-7, f"Invalid station {i}")
            rate, age, slip = row[4:7]
            require(abs(row[2] - x) < 1e-7 and abs(row[3] - y) < 1e-7, f"Moved ID {i} at step {step}")
            if previous is None:
                require(slip == 0 and math.isclose(age, age0, rel_tol=1e-13), f"Incorrect initial history at ID {i}")
            else:
                old = previous[i]
                z = -dt * rate / dc
                age_expected = old[5] * math.exp(z) - dc / rate * math.expm1(z)
                require(math.isclose(age, age_expected, rel_tol=5e-15) and
                        math.isclose(slip, old[6] + dt * rate, rel_tol=5e-15),
                        f"Age/slip update mismatch at step {step}, ID {i}")
                max_log = max(max_log, abs(math.log(age / old[5])))
                increment = dt * abs(rate) / dc
                budget = slip_budget(settings, row[1])
                require(increment <= budget*(1+1e-10), f"Slip fraction {increment} exceeds {budget} at step {step}, ID {i}")
                max_slip = max(max_slip, increment)
        require(not age_cap or max_log <= age_cap * (1 + 1e-10), f"Accepted RSF budget exceeded at step {step}")
        print(f"step={step} t={clock:.10g} s dt={dt:.10g} s peak={peak[4]:.10g} m/s "
              f"max slip/Dc={max_slip:.8g} max |dln(age)|={max_log:.8g}")
        for name, r in (("top", rows[0]), ("base", rows[-1])):
            print(f"  {name}: V={r[4]:.9g} m/s, shear={r[7]:.12g}, raw={r[8]:.12g}, filtered={r[9]:.12g} Pa")
        previous = rows
    check_cuts(domain[3])
    check_native(clocks)
