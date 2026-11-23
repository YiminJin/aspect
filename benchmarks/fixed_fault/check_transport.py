"""Check the two-cell DG stress fixture, its partition face and a discarded attempt."""
import argparse
from pathlib import Path
import os
import subprocess

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(args, name, ranks=1, extra="", expected_failure=None):
    root = args.directory / name
    root.mkdir(parents=True)
    output = root / "output"
    text = (args.source / "tests/dg_stress_transport.prm").read_text()
    mode = args.aspect.name.split("-")[-1]
    libraries = [args.plugins / f"libdg_stress_transport.{mode}.so"]
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
        require(status and expected_failure in message, f"{name} missed its diagnostic")
    else:
        require(status == 0, f"{name} failed; see {root / 'run.log'}")
    return output, message


def read_spatial(output, ranks):
    data = {}
    partition = {}
    for rank in range(ranks):
        for line in (output / f"spatial-{rank}").read_text().splitlines():
            test, t, cell, node, component, x, y, value = line.split()
            key = (int(test), float(t), cell, int(node), int(component))
            require(key not in data, "duplicate owned DG node")
            data[key] = tuple(map(float, (x, y, value)))
            partition[cell] = rank
    require(len(data) == 8 * 2 * 9 * 3, "incomplete spatial output")
    require(len(partition) == 2 and len(set(partition.values())) == ranks,
            "the two cells must straddle the MPI partition")
    return data


def jump_polynomial(duration):
    # Strong form on the unit interval. The lifting of the left trace in P2
    # is 1 - 3 P1(2s-1) + 5 P2(2s-1). Advance its polynomial coefficients
    # by the cubic Taylor polynomial of the affine semidiscrete operator.
    lift = np.array([9., -36., 30.])

    def rhs(p, inflow):
        return 16 * ((inflow - p[0])*lift - np.array([p[1], 2*p[2], 0.]))

    p = np.zeros(3)
    steps = max(1, int(np.ceil(duration * 16 / .1)))
    h = duration / steps
    for _ in range(steps):
        first = rhs(p, 1)
        second = rhs(first, 0)
        third = rhs(second, 0)
        p += h*first + h**2/2*second + h**3/6*third
    return p


def spatial_errors(data):
    errors = np.zeros((5, 3))
    mass = {}
    for (test, t, cell, node, c), (x, y, actual) in data.items():
        if test == 3:
            # Right and left DG traces have distinct identities at x=1/2.
            right = max(row[0] for key, row in data.items() if key[2] == cell) > .5
            reference = (c+1)*(1+y+y*y)
            if right:
                reference *= np.polynomial.polynomial.polyval(2*x-1, jump_polynomial(t))
                # Tensor Simpson integrates the receiver DGQ2 polynomial exactly.
                weight = (.5/36)*(4 if x == .75 else 1)*(4 if y == .5 else 1)
                mass[t,c] = mass.get((t,c), 0.) + weight*actual
        elif test == 4:
            reference = c+1
        else:
            speed = (8., -8., 0.)[test]
            reference = (.3+c)*(x-speed*t)+(.7-.2*c)*y-.4*c
        errors[test,c] = max(errors[test,c], abs(actual-reference))
    np.testing.assert_array_less(errors, 2.e-12)
    # Eliminate the quadratic and cubic temporal terms of a single SSPRK
    # step to extract the signed initial mass transfer. The downstream
    # outflow trace begins at zero, so only the shared face contributes.
    delta = .001
    transfer = np.array([(18*mass[delta,c]-9*mass[2*delta,c]+2*mass[3*delta,c])/(6*delta)
                         for c in range(3)])
    p,w = np.polynomial.legendre.leggauss(3)
    y = (p+1)/2
    expected = np.arange(1,4)*np.dot(w/2, 8*(1+y+y*y))
    np.testing.assert_allclose(transfer, expected, rtol=0, atol=2.e-11)
    print("xx yy xy max nodal errors (Pa), by +u, -u, zero, jump, div(u):\n", errors)
    print("signed interior face transfer (Pa m²/s)", transfer, "errors", transfer-expected)


def accepted(output, ranks):
    labels, coefficients = {}, {}
    clocks = [(0.,0.),(.0125,.0125),(.025,.0125)]
    for r in range(ranks):
        np.testing.assert_array_equal(np.loadtxt(output / f"published-{r}"), clocks)
        require(len(list(output.glob(f"accepted-*-{r}"))) == len(clocks), "unexpected publication")
        for step, clock in enumerate(clocks):
            rows = (output / f"accepted-{step}-{r}").read_text().splitlines()
            np.testing.assert_array_equal(list(map(float,rows[0].split())), clock)
            for j,line in enumerate(rows[1:3]):
                label = list(map(float,line.split()))
                if r:
                    np.testing.assert_array_equal(labels[step,j], label)
                labels[step,j] = label
            for line in rows[3:]:
                cell,c,i,v = line.split()
                key = (step, cell, int(c), int(i))
                require(key not in coefficients, "duplicate accepted DG coefficient")
                coefficients[key] = float(v)
    require(len(coefficients) == 3*2*3*9, "missing DG coefficients")
    return labels, coefficients


def compare(a, b, tolerance=2.e-10):
    require(a.keys() == b.keys(), "mismatched global identities")
    x, y = (np.array([v[k] for k in sorted(a)]) for v in (a,b))
    np.testing.assert_allclose(x,y,rtol=0,atol=tolerance)
    return abs(x-y).max(axis=0)


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
    old, expected, old_prediction = np.loadtxt(stale/"load-0.012500")
    require(abs(old-old_prediction) < 2.e-11 and abs(old-expected) > .014,
            "old-parent control failed to reproduce the receiving-cell bias")
    serial_data, mpi_data = read_spatial(clean,1), read_spatial(parallel,2)
    spatial_errors(serial_data)
    print("max |serial - MPI| by node (x,y,stress)",compare(serial_data, mpi_data))
    labels,stress = accepted(clean,1)
    for path,ranks in ((parallel,2),(retry,1)):
        their_labels,their_stress = accepted(path,ranks)
        tolerance = 0. if path == retry else 2.e-10
        print(path.parent.name, "max differences (x,y,age,slip,rate)", compare(labels,their_labels,tolerance),
              "stress", compare(stress,their_stress,tolerance))
    print("Stokes action, analytic, old-cell prediction (Pa m²)", np.loadtxt(clean/"load-0.012500"))
    print("accepted max stress error (Pa)", float((clean/"endpoint-error").read_text()))


if __name__ == "__main__":
    main()
