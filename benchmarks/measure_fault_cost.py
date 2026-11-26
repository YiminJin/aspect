"""Pair two Linux Release builds on the same installed mature and growing inputs."""
import argparse
import csv
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


def measure_rank(aspect):
    # wait4 reports the solver child's lifetime high-water mark on this rank.
    child = os.fork()
    if child == 0:
        os.execl(aspect, aspect, 'input.prm')
    _, status, usage = os.wait4(child, 0)
    if status:
        raise SystemExit(os.waitstatus_to_exitcode(status) or 1)
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    Path(f'rank-{rank}.rss').write_text(str(usage.ru_maxrss))


def velocity_statistics(table):
    lines = table.decode().splitlines()
    # ASPECT numbers the columns from one in the statistics header.
    indices = []
    for name in ('RMS velocity (m/s)', 'Max. velocity (m/s)'):
        column = next(line.split(':')[0][2:] for line in lines if line.endswith(name))
        indices.append(int(column) - 1)
    return [float(line.split()[i])
            for line in lines if line and not line.startswith('#')
            for i in indices]


def run(binary, directory, settings, ranks, launcher, midpoint, donor=False):
    directory.mkdir()
    (directory/'input.prm').write_text(settings)
    command = [launcher, '--map-by', 'core', '--bind-to', 'core', '-np', str(ranks),
               sys.executable, str(Path(__file__).resolve()), '--rank', str(binary)]
    start = time.monotonic()
    with (directory/'log').open('w') as log:
        subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env=os.environ | dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1'))
    seconds = time.monotonic() - start
    log = (directory/'log').read_text()
    frames = re.split(r'^\*\*\* Timestep \d+:', log, flags=re.MULTILINE)[1:]
    clocks = [float(re.match(r'\s*t=([\d.eE+-]+) seconds', frame)[1]) for frame in frames]
    expected = [0, .005, .01, .015] if midpoint else [0, .0125, .025]
    if clocks != expected:
        raise ValueError(f'{directory}: observed clocks {clocks}')
    marker = 'Midpoint iteration ' if midpoint else 'Solving Stokes system'
    if donor:
        marker = 'Bulk tip coupled Newton '
    counts = tuple(frame.count(marker) for frame in frames)
    if any(count == 0 for count in counts[1:]):
        raise ValueError(f'{directory}: missing solver iterations')
    wall = float(re.findall(r'Total wallclock time elapsed since start\s*\|\s*([\d.eE+-]+)s', log)[-1])
    peaks = [int((directory/f'rank-{i}.rss').read_text()) for i in range(ranks)]
    return counts, seconds, wall, peaks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('baseline', 'current', 'destination'):
        parser.add_argument(arg, type=lambda p: Path(p).resolve())
    parser.add_argument('--mpiexec', default='mpirun')
    parser.add_argument('--donor-input', type=Path,
                        help='compare the frozen nine-strip baseline executable and its indexed input with the installed case')
    args = parser.parse_args()
    if args.baseline.samefile(args.current):
        parser.error('supply distinct executables')
    args.destination.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    cases = [('mature', 'fixed_fault/fixed_fault.prm', 1, 'fixed fault statistics, velocity statistics'),
             ('growing', 'growing_fault/onecut.prm', 2, 'composition statistics, velocity statistics')]
    if args.donor_input:
        cases = [('growing', 'growing_fault/onecut.prm', 2, 'velocity statistics')]
        # Strip recorded paths before ASPECT's shared-library prescan. Both
        # timed runs start fresh and write only the stock velocity statistics.
        donor_input = re.sub(r'^\s*set (?:Additional shared libraries|Output directory|List of postprocessors) =.*$',
                             '', args.donor_input.read_text(), flags=re.MULTILINE)
    with (args.destination/'cost.csv').open('w') as result:
        writer = csv.writer(result)
        writer.writerow(['case', 'pair', 'build', 'elapsed_s', 'aspect_s', 'rss_KiB_by_rank', 'assemblies_by_clock'])
        for name, fixture, ranks, reports in cases:
            reporting = f'''
set Output directory = output
subsection Postprocess
  set List of postprocessors = {reports}
end
'''
            inputs = {label: (here/fixture).read_text() + reporting for label in ('baseline', 'current')}
            if args.donor_input:
                inputs['baseline'] = donor_input + reporting + '''
subsection Checkpointing
  set Steps between checkpoint = 0
end
subsection Termination criteria
  set Checkpoint on termination = false
end
'''
            times = {'baseline': [], 'current': []}
            reference = {}
            for pair in range(5):  # Discard one warm-up pair. Alternate first launch.
                order = list(times) if pair % 2 == 0 else list(reversed(times))
                for label in order:
                    run_dir = args.destination/f'{name}-{pair}-{label}'
                    counts, elapsed, wall, peaks = run(getattr(args, label), run_dir, inputs[label], ranks,
                                                       args.mpiexec, name == 'growing',
                                                       donor=args.donor_input is not None and label == 'baseline')
                    observed = ((run_dir/'output/statistics').read_bytes(), counts)
                    # Across ASPECT versions the statistics schema and Newton
                    # reports differ. Check repeatability within each version.
                    key = label if args.donor_input else 'matched'
                    if observed != reference.setdefault(key, observed):
                        raise ValueError(f'{run_dir}: statistics or nonlinear work differ from the matched control')
                    writer.writerow([name, pair, label, f'{elapsed:.6f}', wall, ' '.join(map(str, peaks)),
                                     ' '.join(map(str, counts))])
                    result.flush()
                    if pair:
                        times[label].append(elapsed)
            for label, samples in times.items():
                print(f'{name} {label}: median end-to-end {statistics.median(samples):.4f} s', flush=True)
            if args.donor_input:
                a, b = (velocity_statistics(reference[key][0]) for key in ('baseline', 'current'))
                if len(a) != 8 or len(a) != len(b) or not all(
                        math.isclose(x, y, rel_tol=1e-7, abs_tol=1e-10) for x, y in zip(a, b)):
                    raise ValueError(f'Donor and installed RMS/maximum velocities differ: {a}, {b}')
                print(f'Largest difference in reported RMS or maximum velocity: {max(abs(x-y) for x,y in zip(a,b)):.3g} m/s')


if __name__ == '__main__':
    if sys.argv[1:2] == ['--rank']:
        measure_rank(sys.argv[2])
    else:
        main()
