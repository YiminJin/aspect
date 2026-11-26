"""Compare fresh, retried and checkpoint-only mature fault trajectories by physical ID."""
import argparse
import csv
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


def fault(settings):
    return f"subsection Material model\n subsection Phase field fault\n{settings}\n end\nend\n"


SAVE = """subsection Checkpointing
  set Steps between checkpoint = 1
  set Number of checkpoints to keep = 4
end
"""
OBSERVE = """subsection Postprocess
  set List of postprocessors = moving fault samples, mature fault statistics
end
"""


def run(name, base, plugins, *, root, prefix, settings="", ranks=1, checkpoint=None,
        diagnostic=None, mpiexec="mpirun"):
    case = root / name
    case.mkdir()
    out = case / "output"
    if checkpoint:
        # Bring only the generation containing the accepted 12.5 ms parent.
        shutil.copytree(checkpoint / "restart/02", out / "restart/02")
        settings += "set Resume computation = true\nsubsection Checkpointing\n set Resume checkpoint = 2\nend\n"
    prm = case / "case.prm"
    prm.write_text(f"include {base}\nset Output directory = {out}\n"
                   f"set Additional shared libraries = {', '.join(map(str, plugins))}\n" + settings)
    command = [prefix / "bin/aspect-release", prm]
    if ranks > 1:
        command = [mpiexec, "-np", str(ranks), *command]
    env = os.environ | dict(ASPECT_SOURCE_DIR=str(prefix), OMP_NUM_THREADS="1",
                            OPENBLAS_NUM_THREADS="1", DEAL_II_NUM_THREADS="1")
    log = case / "log"
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=case, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if diagnostic:
        if result.returncode == 0 or diagnostic not in " ".join(log.read_text().split()):
            raise ValueError(f"{case}: expected {diagnostic}")
        if list(out.glob("material-*-surface-0.csv")):
            raise ValueError(f"{case}: incompatible state reached publication")
        print(name, "rejected:", diagnostic, flush=True)
    else:
        result.check_returncode()
    return out


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def accepted_frame(out, step, ranks):
    surface = read_rows(out / f"material-{step}-surface-0.csv")
    ids = [int(r["i"]) for r in surface]
    if ids != list(range(len(surface))) or len(ids) < 2:
        raise ValueError(f"Incomplete material identities in {out}")
    bulk = {}
    for rank in range(ranks):
        if read_rows(out / f"material-{step}-surface-{rank}.csv") != surface:
            raise ValueError(f"Replicated accepted labels differ on rank {rank}")
        rows = read_rows(out / f"material-{step}-rank{rank}.csv")
        if not any(float(row["chi"]) > 0 for row in rows):
            raise ValueError(f"Rank {rank} has no active normal support")
        for row in rows:
            key = (row["cell"], int(row["q"]))
            if key in bulk:
                raise ValueError(f"Duplicate native sample {key}")
            bulk[key] = row
    return {int(row["i"]): row for row in surface}, bulk


def check_clocks(out, steps):
    with (out / "statistics").open() as stream:
        clocks = [tuple(map(float, row.split()[:3])) for row in stream
                  if row.strip() and not row.startswith("#")]
    expected = [(i, i * .0125, .0125 if i else 0.) for i in range(max(steps)+1)]
    np.testing.assert_allclose(clocks, expected, rtol=0, atol=1e-15)
    published = sorted(int(p.name.split("-")[1]) for p in out.glob("material-*-surface-0.csv"))
    if published != list(steps):
        raise ValueError(f"{out}: accepted steps {published}, expected {list(steps)}")


def internal_shear_work(bulk):
    """Pair the accepted DG stress with the virtual displacement (y,0), in J/m."""
    by_cell = {}
    for (cell, q), sample in bulk.items():
        by_cell.setdefault(cell, {})[q] = sample
    # On each rectangular cell, the outer Gauss points span sqrt(3/5) of
    # each edge. Recover physical JxW independently from the recorded points.
    weights = [5/18, 4/9, 5/18]
    terms = []
    for cell in sorted(by_cell):
        samples = by_cell[cell]
        if set(samples) != set(range(9)):
            raise ValueError(f"Incomplete Q3 quadrature in {cell}")
        x = [float(samples[q]["x"]) for q in range(9)]
        y = [float(samples[q]["y"]) for q in range(9)]
        area = (max(x)-min(x)) * (max(y)-min(y)) / .6
        for q in range(9):
            terms.append(weights[q % 3] * weights[q // 3] * area * float(samples[q]["tauxy"]))
    return math.fsum(terms)


def compare(reference, measured, step, ranks=1, reference_ranks=1, exact=True):
    a, u = accepted_frame(reference, step, reference_ranks)
    b, v = accepted_frame(measured, step, ranks)
    worst = {}
    atol, rtol = (0., 0.) if exact else (1e-10, 1e-8)
    for left, right, keys in [(a, b, ("i",)), (u, v, ("cell", "q"))]:
        if left.keys() != right.keys() or set(next(iter(left.values()))) != set(next(iter(right.values()))):
            raise ValueError("Native sample, material ID or column set mismatch")
        ids = sorted(left)
        for col in left[ids[0]].keys() - set(keys):
            ref, obs = (np.array([float(data[k][col]) for k in ids]) for data in (left, right))
            if not np.all(np.isfinite([ref, obs])):
                raise ValueError(f"Nonfinite {col}")
            np.testing.assert_allclose(obs, ref, rtol=rtol, atol=atol, err_msg=col)
            worst[col] = max(worst.get(col, 0.), max(abs(obs-ref)))
    w_ref, w_obs = (internal_shear_work(bulk) for bulk in (u, v))
    if (step > 0 and abs(w_ref) < 1.e-8) or not math.isclose(w_ref, w_obs, rel_tol=rtol, abs_tol=atol):
        raise ValueError(f"Unresolved or mismatched internal shear work: {w_ref} / {w_obs}")
    print(f"{measured.parent.name}, step {step}: {len(a)} IDs, {len(u)} native samples; "
          f"max |Δx,y| {max(worst['x'],worst['y']):.4g} m, ΔI {worst['I_m']:.4g} m, "
          f"Δage {worst['Theta_s']:.4g} s, Δslip {worst['slip']:.4g} m, ΔV {worst['V_m_s']:.4g} m/s, "
          f"Δstress {max(worst[k] for k in ('tauxx','tauyy','tauxy')):.4g} Pa; "
          f"work {w_obs:.15g} J/m, |Δwork| {abs(w_obs-w_ref):.4g} J/m", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prefix", "layered_plugin", "retry_plugin", "destination"):
        parser.add_argument(name, type=lambda p: Path(p).resolve())
    parser.add_argument("--mpiexec", default="mpirun")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    example = args.prefix / "benchmarks/fixed_fault/curved_mature"
    base = example / "curved_mature.prm"
    common = dict(root=args.destination, prefix=args.prefix, mpiexec=args.mpiexec)
    plugins = [args.layered_plugin]

    clean = run("curved-clean", base, plugins, settings=OBSERVE, **common)
    parallel = run("curved-clean-two", base, plugins, settings=OBSERVE, ranks=2, **common)
    seed = run("curved-seed", base, plugins, settings=OBSERVE + SAVE + "set End time = .0125\n", **common)
    check_clocks(seed, [0, 1])
    for folder in (clean, parallel):
        check_clocks(folder, [0, 1, 2])
    # Copy the physical table to a new filename to exercise a path-independent signature.
    table = args.destination / "renamed-table.txt"
    shutil.copyfile(example / "vertices.txt", table)
    for ranks, reference in [(1, clean), (2, parallel)]:
        resumed = run(f"curved-resume-{ranks}", base, plugins, settings=OBSERVE + fault(f"set Fault vertex file = {table}"),
                      ranks=ranks, checkpoint=seed, **common)
        check_clocks(resumed, [2])
        compare(reference, resumed, 2, ranks, ranks, exact=ranks == 1)
        if ranks == 2:
            compare(clean, resumed, 2, ranks, exact=False)
    for step in range(3):
        compare(clean, parallel, step, 2, exact=False)
    retried = run("curved-retry", base, [*plugins, args.retry_plugin], settings=OBSERVE + """subsection Time stepping
  set List of model names = function, reject long first fault step
  subsection Function
    set Function expression = if(time==0,0.025,0.0125)
  end
end
""", **common)
    if (retried.parent / "log").read_text().count("Repeating the current time step") != 1:
        raise ValueError("Expected exactly one solved and rejected first interval")
    check_clocks(retried, [0, 1, 2])
    for step in range(3):
        compare(clean, retried, step)

    run("changed-sense", base, plugins, settings=OBSERVE + fault("set Mature slip sense = along tangent"),
        checkpoint=seed, diagnostic="Mature fault slip sense differs from the checkpoint", **common)
    changed = []
    for line in table.read_text().splitlines():
        cols = line.split()
        if cols and cols[0] == "2":
            cols[2] = str(float(cols[2]) + .01)
            line = " ".join(cols)
        changed.append(line)
    shifted = args.destination / "displaced-vertex.txt"
    shifted.write_text("\n".join(changed) + "\n")
    run("changed-vertex", base, plugins, settings=OBSERVE + fault(f"set Fault vertex file = {shifted}"),
        checkpoint=seed, diagnostic="ordered prescribed vertices differ from the checkpoint", **common)
    # Persistent loading belongs to its physical ID, even at identical geometry.
    columns = [line.split("#")[0].split() for line in table.read_text().splitlines()]
    header, *rows = [c for c in columns if c]
    shear = header.index("background_shear_Pa")
    rows[2][shear] = str(float(rows[2][shear]) + .1)
    changed_load = args.destination / "changed-loading.txt"
    changed_load.write_text("\n".join(" ".join(r) for r in [header, *rows]) + "\n")
    run("changed-loading", base, plugins, settings=OBSERVE + fault(f"set Fault vertex file = {changed_load}"),
        checkpoint=seed, diagnostic="Persistent vertex shear tractions differ from the checkpoint", **common)
    run("changed-friction", base, plugins, settings=OBSERVE + fault("set Radiation damping = 2"),
        checkpoint=seed, diagnostic="Mature fault resistance parameters differ from the checkpoint", **common)

    # Fit a straight, displaced seed to the same FE band. Each ID is compared
    # to its virgin supplied position before continuing from the measured fit.
    positions = np.linspace([0., .65], [1., .28], 5)
    inline = "; ".join(f"{x:.17g},{y:.17g}" for x, y in positions)
    fit = OBSERVE + fault("set Fit trace to phase = true\nset Fault vertex file =\n"
                          f"set Fault vertices = {inline}")
    fitted = run("fitted-clean", base, plugins, settings=fit + SAVE, **common)
    initial = read_rows(fitted / "material-0-surface-0.csv")
    distances = [np.linalg.norm([float(row["x"])-x, float(row["y"])-y])
                 for (x,y), row in zip(positions, initial)]
    if len(initial) != len(positions) or max(distances) < .01:
        raise ValueError("Initial fit must displace a station by more than 1 cm")
    print("Initial fit displacements in material order (m):", *distances, flush=True)
    resumed = run("fitted-resume", base, plugins, settings=fit, checkpoint=fitted, **common)
    check_clocks(resumed, [2])
    compare(fitted, resumed, 2)


if __name__ == "__main__":
    main()
