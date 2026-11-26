"""Reproduce the strict C1 affine control on one/two ranks and verify the captured state.

Supply the two executables, the matching growing benchmark plugin, a fresh output
directory and the local MPI launcher. Each failing strict check must fail only for
the two historically out-of-budget columns. A different failure stops the replay.
"""
import argparse
import csv
from fractions import Fraction
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def records(files, columns):
    table = {}
    for path in files:
        with path.open() as source:
            for row in csv.DictReader(source):
                key = tuple(row[c] for c in columns)
                if key in table:
                    raise ValueError(f'Duplicate {key} in {path}')
                table[key] = row
    if not table:
        raise ValueError('Missing records')
    return table


def run(checker, args, name, executable, ranks, candidate):
    group = 'candidate' if candidate else name
    dest = args.out / f'{group}-{ranks}'
    cmd = [sys.executable, checker, executable, args.plugin, dest, 'profile']
    if ranks == 2:
        cmd += ['--ranks', str(ranks), '--mpiexec', args.mpiexec, '--serial', args.out/f'{group}-1']
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               DEAL_II_NUM_THREADS='1', UCX_LOG_LEVEL='error')
    env.pop('DIR_C1_PUBLISH_CANDIDATE', None)
    if candidate:
        env['DIR_C1_PUBLISH_CANDIDATE'] = '1'
    log = dest.with_suffix('.log')
    with log.open('w') as stream:
        result = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if ranks == 2 and not candidate:
        lines = log.read_text().splitlines()
        if result.returncode != 1 or not lines or not lines[-1].startswith(
                'ValueError: Cross-rank strict comparisons: beta at (8,)') or (
                '; ds at (8,)' not in lines[-1] or lines[-1].count(';') != 1):
            raise ValueError(f'{group}-2: unexpected result, inspect {log}')
    elif result.returncode != 0:
        raise ValueError(f'{group}-{ranks}: unexpected failure, inspect {log}')
    print(f'{group}-{ranks}: strict checker exit {result.returncode}')


def check_archived_fraction(original, fresh):
    """The archive permits one rounding of the product-plus-sum in the denominator.
    The shared degradation object in this checkout materializes the intact factor.
    Reconstruct both roundings on the same parsed FE phase, without a tolerance.
    """
    modulus = 1e6
    cohesion = 1000.
    magnitude = 26.6666666667 / ((8./3.)*.25) * 2. * modulus / (cohesion*cohesion)
    differences = []
    for old, new in zip(original.splitlines(), fresh.splitlines(), strict=True):
        old_fields, new_fields = old.split(), new.split()
        if old_fields[:4] != new_fields[:4] or old_fields[5:] != new_fields[5:]:
            raise ValueError('Cell/quadrature identity, weights, stress or phase changed')
        phi = float(new_fields[-1])
        c, z = magnitude * phi * (1.+phi), 1.-phi
        fused_denominator = float(Fraction(z)*Fraction(z) + Fraction(c))
        stored_denominator = c + z*z
        previous, current = float(old_fields[4]), float(new_fields[4])
        if previous != c/fused_denominator or current != c/stored_denominator:
            raise ValueError('Fraction not explained by rounding the intact factor')
        if current != previous:
            differences.append(abs(current-previous))
    return differences


def verify(out):
    archive = Path(__file__).resolve().parent.parent/'interface-c2/c1-captures.tar.gz'
    with tarfile.open(archive) as sources:
        for ranks in [1, 2]:
            captures = {g: out/f'{g}-{ranks}'/'affine/output' for g in ('ordinary','production','candidate')}
            # The observer's cell IDs are release CellIds. Compare each layout
            # against its own archived fields and compare policies before the turn.
            prefix = f'build-interface-c2/cap{ranks}/output/'
            tables = ['sections-0000.csv'] + [f'{s}-0000-r{r}.csv' for r in range(ranks) for s in ('points','fields')]
            for name in tables:
                original = sources.extractfile(prefix+name).read()
                for group, run_dir in captures.items():
                    if (run_dir/name).read_bytes() != original:
                        raise ValueError(f'{name} changed on {group}-{ranks}')
            differences = []
            for r in range(ranks):
                for category in ('points','fields'):
                    name = f'turn-{category}-{r}'
                    original = sources.extractfile(prefix+name).read()
                    fresh = (captures['production']/name).read_bytes()
                    if fresh != (captures['candidate']/name).read_bytes():
                        raise ValueError(f'{name} changed between policies on {ranks} ranks')
                    if category == 'points':
                        differences.extend(check_archived_fraction(original, fresh))
                    elif fresh != original:
                        raise ValueError(f'{name} changed since the archive on {ranks} ranks')
                name = f'turn-sums-{r}'
                archived = sources.extractfile(prefix+name).read().splitlines()
                for group in ('production','candidate'):
                    live = (captures[group]/name).read_bytes().splitlines()
                    if live[:2] != archived[:2]:
                        raise ValueError(f'{name} raw or centered moments changed on {group}-{ranks}')
            if len(differences)!=4 or max(differences)!=2.**-64:
                raise ValueError(f'Unexpected number or size of historical fraction differences: {differences}')
            name = 'sections-0001.csv'
            original = sources.extractfile(prefix+name).read()
            for group in ('ordinary','production'):
                if (captures[group]/name).read_bytes() != original:
                    raise ValueError(f'{name} changed on {group}-{ranks}')
            decisions = []
            for group, policy in (('production',0),('candidate',1)):
                run_dir = captures[group]
                with (run_dir/'turn-decisions.csv').open() as stream:
                    first = [row for row in csv.DictReader(stream) if row['k']=='8' and row['incoming']=='0']
                if len(first)!=2 or [row['policy'] for row in first]!=['0','1']:
                    raise ValueError(f'Expected one earliest turn per policy on {group}-{ranks}')
                published = records([run_dir/name], ['section'])['8',]
                if (float(first[policy]['angle']),float(first[policy]['ds'])) != (float(published['beta']),float(published['ds'])):
                    raise ValueError(f'Selected geometry differs from publication on {group}-{ranks}')
                decisions.append(first)
            if decisions[0] != decisions[1]:
                raise ValueError(f'Earliest common-input decisions differ between policies on {ranks} ranks')
            print(f'{ranks} rank(s): archived parent, stage, reductions and production sections identical; four explained 1-ULP F differences; '
                  f'bin/beta/ds = {[(r["bin"],r["angle"],r["ds"]) for r in decisions[0]]}')


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('ordinary','capture','plugin','out','mpiexec'):
        p.add_argument(key, type=lambda x:Path(x).resolve())
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    checker = Path(__file__).resolve().parents[4]/'benchmarks/growing_fault/check_adiabatic_pressure.py'
    if not args.verify_only:
        args.out.mkdir(parents=True, exist_ok=False)
        for group, binary in [('ordinary',args.ordinary),('production',args.capture),('candidate',args.capture)]:
            for ranks in (1,2):
                run(checker,args,group,binary,ranks,group=='candidate')
    verify(args.out)
