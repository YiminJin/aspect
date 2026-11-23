"""Check the two-cell DG stress fixture, its partition face and a discarded attempt."""
import argparse
import math
from pathlib import Path
import os
import re
import subprocess
from tempfile import TemporaryDirectory


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_record(values, key):
    numbers = tuple(map(float, values))
    require(all(math.isfinite(v) for v in numbers), f"nonfinite record {key}")
    return numbers


def run(args, name, ranks=1, extra="", expected_failure=None, fixture="dg_stress_transport"):
    root = args.directory / name
    root.mkdir(parents=True)
    output = root / "output"
    text = (args.source / "tests" / f"{fixture}.prm").read_text()
    mode = args.aspect.name.split("-")[-1]
    libraries = [args.plugins / f"lib{fixture}.{mode}.so"]
    if name == "retry":
        libraries.append(args.plugins / f"libfixed_fault.{mode}.so")
    prm = root / "input.prm"
    prm.write_text(text + "\nset Output directory = " + str(output) +
                   "\nset Additional shared libraries = " + ", ".join(map(str, libraries)) + "\n" + extra)
    launch = [args.aspect, prm] if ranks == 1 else [args.mpiexec, "-n", str(ranks), args.aspect, prm]
    with (root / "run.log").open("w") as log:
        status = subprocess.run(launch, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, ASPECT_SOURCE_DIR=str(args.source),
                                         DEAL_II_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")).returncode
    message = (root / "run.log").read_text()
    if expected_failure:
        require(status and expected_failure in " ".join(message.split()), f"{name} missed its diagnostic")
    else:
        require(status == 0, f"{name} failed; see {root / 'run.log'}")
    return output, message


def read_spatial(output, ranks):
    data = {}
    partition = {}
    for rank in range(ranks):
        for line in (output / f"spatial-{rank}").read_text().splitlines():
            test, t, cell, node, component, x, y, value = line.split()
            t, x, y, value = finite_record((t, x, y, value), (test, t, cell, node, component))
            key = (int(test), t, cell, int(node), int(component))
            require(key not in data, "duplicate owned DG node")
            data[key] = (x, y, value)
            require(partition.setdefault(cell, rank) == rank, "a cell has multiple owners")
    require(len(data) == 8 * 2 * 9 * 3, "incomplete spatial output")
    require(len(partition) == 2 and len(set(partition.values())) == ranks,
            "the two cells must straddle the MPI partition")
    return data


def jump_polynomial(duration):
    # Strong form on the unit interval. The lifting of the left trace in P2
    # is 1 - 3 P1(2s-1) + 5 P2(2s-1). Advance its polynomial coefficients
    # by the cubic Taylor polynomial of the affine semidiscrete operator.
    def rhs(p, inflow):
        a, b, c = p
        jump = inflow - a
        return (16*(9*jump-b), 16*(-36*jump-2*c), 16*30*jump)

    p = (0., 0., 0.)
    steps = max(1, math.ceil(duration * 16 / .1))
    h = duration / steps
    for _ in range(steps):
        first = rhs(p, 1)
        second = rhs(first, 0)
        third = rhs(second, 0)
        p = tuple(a + h*b + h**2*c/2 + h**3*d/6 for a, b, c, d in zip(p, first, second, third))
    return p


def spatial_errors(data):
    errors = [[0.]*3 for _ in range(5)]
    mass = {}
    for (test, t, cell, node, c), (x, y, actual) in data.items():
        if test == 3:
            # Right and left DG traces have distinct identities at x=1/2.
            right = max(row[0] for key, row in data.items() if key[2] == cell) > .5
            reference = (c+1)*(1+y+y*y)
            if right:
                a, b, d = jump_polynomial(t)
                xi = 2*x-1
                reference *= a + xi*(b + xi*d)
                # Tensor Simpson integrates the receiver DGQ2 polynomial exactly.
                weight = (.5/36)*(4 if x == .75 else 1)*(4 if y == .5 else 1)
                mass[t,c] = mass.get((t,c), 0.) + weight*actual
        elif test == 4:
            reference = c+1
        else:
            speed = (8., -8., 0.)[test]
            reference = (.3+c)*(x-speed*t)+(.7-.2*c)*y-.4*c
        error = abs(actual-reference)
        require(error < 2.e-12, f"spatial error {error} at {(test,t,cell,node,c)}")
        errors[test][c] = max(errors[test][c], error)
    # Eliminate the quadratic and cubic temporal terms of a single SSPRK
    # step to extract the signed initial mass transfer. The downstream
    # outflow trace begins at zero, so only the shared face contributes.
    delta = .001
    transfer = [(18*mass[delta,c]-9*mass[2*delta,c]+2*mass[3*delta,c])/(6*delta)
                for c in range(3)]
    # Three-point Gauss rule on [0,1].
    offset = math.sqrt(3/5)/2
    flux = sum(w*8*(1+y+y*y) for w,y in ((5/18,.5-offset), (4/9,.5), (5/18,.5+offset)))
    defects = [value-(c+1)*flux for c,value in enumerate(transfer)]
    require(all(abs(e) < 2.e-11 for e in defects), f"incorrect face transfer {defects}")
    print("xx yy xy max nodal errors (Pa), by +u, -u, zero, jump, div(u):", errors)
    print("signed interior face transfer (Pa m²/s)", transfer, "errors", defects)


def accepted(output, ranks):
    labels, coefficients = {}, {}
    clocks = [(0.,0.),(.0125,.0125),(.025,.0125)]
    for r in range(ranks):
        published = (output / f"published-{r}").read_text().splitlines()
        require([finite_record(line.split(), "publication") for line in published] == clocks,
                "unexpected publication clocks")
        require(len(list(output.glob(f"accepted-*-{r}"))) == len(clocks), "unexpected publication")
        for step, clock in enumerate(clocks):
            rows = (output / f"accepted-{step}-{r}").read_text().splitlines()
            require(finite_record(rows[0].split(), step) == clock, "unexpected accepted clock")
            for j,line in enumerate(rows[1:3]):
                label = finite_record(line.split(), (step,j))
                if r:
                    require(labels[step,j] == label, "mismatched replicated labels")
                labels[step,j] = label
            for line in rows[3:]:
                cell,c,i,v = line.split()
                key = (step, cell, int(c), int(i))
                require(key not in coefficients, "duplicate accepted DG coefficient")
                coefficients[key] = finite_record([v], key)
    require(len(coefficients) == 3*2*3*9, "missing DG coefficients")
    return labels, coefficients


def compare(a, b, tolerance=2.e-10):
    require(a.keys() == b.keys(), "mismatched global identities")
    maxima = []
    for key in sorted(a):
        x, y = a[key], b[key]
        require(len(x) == len(y), f"record lengths differ at {key}")
        if not maxima:
            maxima = [0.]*len(x)
        for i, (left, right) in enumerate(zip(x, y)):
            error = abs(left-right)
            require(error <= tolerance, f"record {key} component {i} differs by {error}")
            maxima[i] = max(maxima[i], error)
    return maxima


def check_corrupt_jump(output, ranks):
    """A single NaN at the receiver center must fail even on matching MPI records."""
    data = read_spatial(output, ranks)
    key = None
    with TemporaryDirectory() as tmp:
        for rank in range(ranks):
            name = f"spatial-{rank}"
            lines = []
            for line in (output/name).read_text().splitlines():
                test, t, cell, node, c, x, y, value = line.split()
                if (int(test), float(t), int(node), int(c), float(x), float(y)) == (3, .024, 4, 1, .75, .5):
                    require(key is None, "duplicate receiver yy")
                    key = (int(test), float(t), cell, int(node), int(c))
                    line = " ".join((test, t, cell, node, c, x, y, "nan"))
                lines.append(line)
            (Path(tmp)/name).write_text("\n".join(lines)+"\n")
        require(key is not None, "missing receiver yy")
        try:
            read_spatial(Path(tmp), ranks)
        except ValueError as error:
            require("nonfinite record" in str(error), f"wrong corruption diagnostic: {error}")
        else:
            raise ValueError("accepted a nonfinite receiving-cell jump record")

    # Exercise the reductions independently of the file reader. Both sides
    # of the partition comparison carry the identical corrupted coefficient.
    x, y, _ = data[key]
    data[key] = (x, y, math.nan)
    for check, diagnostic in ((lambda: spatial_errors(data), "spatial error nan"),
                              (lambda: compare(data, data), "differs by nan")):
        try:
            check()
        except ValueError as error:
            require(diagnostic in str(error), f"wrong corruption diagnostic: {error}")
        else:
            raise ValueError("a comparison accepted the nonfinite receiver yy")


def check_invalid_stage(args, ranks, overflow=False):
    # Only the owner of the left cell sees the invalid input. The finite
    # overflow case first injects a large flux at the last RK evaluation.
    if overflow:
        yy = "if(x<0.1,if(t>0.002,if(t<0.0021,1e308,0),0),0)"
        stage, time = 3, .0125/6
        origin = "owned DG stage coefficient"
        name = "stage-overflow"
    else:
        yy = "if(x<0.1,if(t>0,sqrt(x-1),0),0)"
        stage, time = 2, .0125/3
        origin = "exterior parent stress"
        name = "nonfinite-inflow"
    expected = f"Nonfinite {origin} yy on 1 of {ranks} ranks at SSPRK stage {stage}, time"
    output, log = run(args, f"{name}-{ranks}", ranks, extra=f"""
set Nonlinear solver failure strategy = cut timestep size
subsection Material model
  subsection Phase field fault
    subsection Exterior parent stress
      set Function expression = 0.4+0.3*(x-8*t); {yy}; 0.05
    end
  end
end
""", expected_failure=expected)
    match = re.search(re.escape(expected) + r" ([0-9.eE+-]+) s", " ".join(log.split()))
    require(match and math.isclose(float(match[1]), time, rel_tol=1e-14), "incorrect failure clock")
    require("Repeating the current time step" not in log, "invalid transport data triggered a cutback")
    for rank in range(ranks):
        require((output / f"published-{rank}").read_text().split() == ["0","0"],
                "invalid stage reached publication")
    print(expected, match[1], "s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aspect","plugins","directory"):
        parser.add_argument(name, type=lambda s: Path(s).resolve())
    parser.add_argument("--mpiexec", default="mpiexec")
    args = parser.parse_args()
    args.source = Path(__file__).resolve().parents[2]
    # Each invocation uses fresh files and starts every simulation at t=0.
    clean,_ = run(args,"clean")
    parallel,_ = run(args,"mpi",2)
    retry,log = run(args,"retry",extra="""
subsection Time stepping
  set List of model names = function, reject long first fault step
  subsection Function
    set Function expression = if(time==0,0.025,0.0125)
  end
end
""")
    require(log.count("Repeating the current time step") == 1, "expected one rejected long attempt")
    require((retry/"load-0.025000").exists(), "long attempt did not reach Stokes assembly")
    stale,_ = run(args,"old-parent",extra="""
subsection Material model
  subsection Phase field fault
    set Advect physical stress = false
  end
end
""",expected_failure="Actual Stokes stress load missed the transported receiving-cell parent")
    old, expected, old_prediction = finite_record((stale/"load-0.012500").read_text().split(), "load")
    require(abs(old-old_prediction) < 2.e-11 and abs(old-expected) > .014,
            "old-parent control failed to reproduce the receiving-cell bias")
    for ranks in (1,2):
        check_invalid_stage(args, ranks)
        check_invalid_stage(args, ranks, overflow=True)
        spin, _ = run(args, f"spin-order-{ranks}", ranks, fixture="dg_stress_spin_order")
        print("receiver xy, expected, reversed (Pa), accepted rate (m/s):",
              finite_record((spin/"spin-values").read_text().split(), "spin"))
    serial_data, mpi_data = read_spatial(clean,1), read_spatial(parallel,2)
    check_corrupt_jump(clean,1)
    check_corrupt_jump(parallel,2)
    spatial_errors(serial_data)
    print("max |serial - MPI| by node (x,y,stress)",compare(serial_data, mpi_data))
    labels,stress = accepted(clean,1)
    for path,ranks in ((parallel,2),(retry,1)):
        their_labels,their_stress = accepted(path,ranks)
        tolerance = 0. if path == retry else 2.e-10
        print(path.parent.name, "max differences (x,y,age,slip,rate)", compare(labels,their_labels,tolerance),
              "stress", compare(stress,their_stress,tolerance))
    print("Stokes action, analytic, old-cell prediction (Pa m²)", finite_record((clean/"load-0.012500").read_text().split(), "load"))
    error, = finite_record((clean/"endpoint-error").read_text().split(), "endpoint error")
    print("accepted max stress error (Pa)", error)


if __name__ == "__main__":
    main()
