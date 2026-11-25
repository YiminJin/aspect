"""Run the Increment 6A review controls with Release executables and fresh outputs."""
import argparse
from copy import copy
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmarks/explicit_dg_transport'))
from check import BASE, compare, fields, require, run, section, setting
from check_3d import simulate


# Release artifacts retained before the 6A build. The probe is loaded in the
# writer process; matching just its filename would also admit the rebuilt DSO.
WRITER_SHA256 = '8769c98b64923906964379ef78917567eb93edc40da93b9d38765b187184022d'
BASELINE_PLUGINS = {
    'libexplicit_dg_compositions.release.so': 'd3f170879cc20b33a1b84f99172ade28a255e1ac5328209b82edf6456ddae12b',
    'libpostprocess_iterations_iterated_Stokes.release.so': '4e483bf9c9bdcd8434139ebdbbeb3041e5f332c33c6f61b7927316b789b1731e',
}
HISTORY = 'libnonlinear_solver_history.release.so'


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def check_baseline(args, plugin):
    require(digest(args.writer) == WRITER_SHA256, f'pre-6A writer SHA-256 mismatch: {args.writer}')
    require(digest(args.writer_plugins/plugin) == BASELINE_PLUGINS[plugin],
            f'pre-6A plugin SHA-256 mismatch: {args.writer_plugins/plugin}')
    require(digest(args.aspect) != WRITER_SHA256, 'the reader must differ from the pre-6A writer')
    for parent in (args.writer.parent, args.writer_plugins):
        require(args.directory != parent and parent not in args.directory.parents,
                f'replay destination is inside the baseline: {parent}')


def restart(args):
    """Write the mixed-basis slab with the retained pre-6A pair and check its continuation."""
    check_baseline(args, 'libexplicit_dg_compositions.release.so')
    args.directory.mkdir(parents=True)
    writer = copy(args)
    writer.aspect, writer.plugins = args.writer, args.writer_plugins
    write = section('Checkpointing', setting('Steps between checkpoint', 1))
    seed = simulate(writer, 'writer', setting('End time', .02) + write)
    clocks = [(0., 0.), (.01, .01), (.02, .01), (.03, .01)]
    require([seed[i][0] for i in sorted(seed)] == clocks[:3], 'wrong pre-6A writer clocks')

    methods = fields(SSPRK3_fields='', Compositional_field_methods='field, explicit DG SSPRK3, explicit DG SSPRK3')
    finish = methods + setting('End time', .03)
    resumed = simulate(args, 'resumed', finish + setting('Resume computation', 'true'),
                       checkpoint=args.directory/'writer/output')
    log = (args.directory/'resumed/run.log').read_text()
    require('Resuming from snapshot' in log and re.findall(r'^\*\*\* Timestep (\d+):', log, re.M) == ['3'],
            'reader did not continue the saved 0.02 s endpoint for one step')
    direct = simulate(args, 'uninterrupted', finish)
    require([direct[i][0] for i in sorted(direct)] == clocks, 'wrong uninterrupted clocks')
    bases = {'companion': 'FE_DGQ<3>(2)', 'damage': 'FE_DGQBernstein<3>(2)', 'tracer': 'FE_DGQ<3>(2)'}
    for _, rows in direct.values():
        cells = {cell for cell, field in rows}
        require(len(cells) == 2 and rows.keys() == {(cell, field) for cell in cells for field in bases},
                'expected three fields on both slabs')
        require(all(row[1] == bases[field] for (_, field), row in rows.items()), 'unexpected slab basis')
    error = compare(resumed, direct, 0.)
    print(f'Accepted clocks {[t for t, dt in clocks]} s, {bases}, 162 coefficients per clock, max difference {error:g}')


def run_case(directory, binary, parameters, plugins):
    directory.mkdir()
    prm = directory/'input.prm'
    prm.write_text(parameters + '\n' + setting('Output directory', 'output')
                   + setting('Additional shared libraries', ', '.join(str(path) for path in plugins)))
    env = dict(os.environ, ASPECT_SOURCE_DIR=str(ROOT), DEAL_II_NUM_THREADS='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (directory/'run.log').open('w') as log:
        subprocess.run([binary, prm], cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return (directory/'run.log').read_text()


def stock_picard(args):
    """Compare stock iteration reports and statistics with the retained Release binary."""
    material = 'libpostprocess_iterations_iterated_Stokes.release.so'
    check_baseline(args, material)
    args.directory.mkdir(parents=True)
    parameters = (ROOT/'tests/postprocess_iterations_iterated_Stokes.prm').read_text()
    for name, binary, plugins in [('baseline', args.writer, args.writer_plugins), ('current', args.aspect, args.plugins)]:
        directory = args.directory/name
        log = run_case(directory, binary, parameters, [plugins/material, args.plugins/HISTORY])
        iterations = [line for line in log.splitlines()
                      if re.search(r'number of nonlinear iterations:|Relative nonlinear residual|Solving Stokes system', line)]
        completions = re.findall(r'number of nonlinear iterations: (\d+)\. State: 1\.', log)
        require(len(iterations) == 23 and completions == ['5', '1', '1', '1', '1'], f'{name}: incomplete Picard history')
        (directory/'iterations').write_text('\n'.join(iterations) + '\n')
    for filename in ('iterations', 'output/statistics'):
        require((args.directory/'baseline'/filename).read_bytes() == (args.directory/'current'/filename).read_bytes(),
                f'pre-6A and current {filename} differ')
    print('Matched stock statistics and 23 iteration records, including five successful completions (5, 1, 1, 1, 1).')


def moving_retry(args):
    """Compare clean and retried physical tables and count phase and nonlinear completions."""
    args.directory.mkdir(parents=True)
    base = (ROOT/'benchmarks/layered_moving_fault/layered.prm').read_text()
    run_case(args.directory/'clean', args.aspect, base, [args.layered_plugin])
    methods = fields(SSPRK3_fields='', Compositional_field_methods='static, static, static, explicit DG SSPRK3')
    retry = section('Time stepping', setting('List of model names', 'function, reject long first fault step') +
                    section('Function', setting('Function expression', 'if(time==0,.025,.0125)')))
    observers = section('Postprocess', setting('List of postprocessors',
                        'layered background, moving fault samples, composition statistics, DG method setup'))
    log = run_case(args.directory/'retry', args.aspect, base + methods + retry + observers,
                   [args.layered_plugin, args.plugins/'libfixed_fault.release.so',
                    args.plugins/'libexplicit_dg_compositions.release.so', args.plugins/HISTORY])
    require(log.count('Repeating the current time step') == 1 and
            re.findall(r'DG completion (\d+) phase_field', log) == ['0', '1', '1', '2'], 'wrong phase attempt sequence')
    iterations = re.findall(r'number of nonlinear iterations: (\d+)\. State: 1\.', log)
    require(len(iterations) == 4, 'missing nonlinear completion')
    expected = {pattern.format(step) for step in range(3)
                for pattern in ('load-{}.txt', 'material-{}-rank0.csv', 'material-{}-surface-0.csv')}
    clean, repeated = [args.directory/name/'output' for name in ('clean', 'retry')]
    for directory in (clean, repeated):
        require({p.name for p in directory.iterdir() if p.match('material-*.csv') or p.match('load-*.txt')} == expected,
                f'incomplete or unexpected endpoint tables in {directory}')
    for name in expected:
        require((clean/name).read_bytes() == (repeated/name).read_bytes(), f'retried accepted state differs in {name}')
    print(f'Phase completions 0, 1, 1, 2. Picard iterations {", ".join(iterations)}. Nine accepted physical tables identical.')


def guide_example(args):
    """Run the settings printed in the user guide on the two-cell box."""
    guide = (ROOT/'doc/sphinx/user/methods/compositional-fields.md').read_text()
    subsection = guide.split('## Selecting an explicit advection method\n', 1)[1]
    excerpt = re.search(r'^```text\n(.*?)^```$', subsection, re.M | re.S)
    require(excerpt is not None, 'missing guide parameter excerpt')
    # Remove the fixture's own selection, refinement and DG settings so that
    # only the documented excerpt can enable the method on a fixed mesh.
    template, removed = re.subn(r'^subsection (?:Compositional fields|Discretization|Mesh refinement)\n.*?^end$',
                                '', BASE, flags=re.M | re.S)
    require(removed == 3, 'unexpected mixed box parameter layout')
    args.directory.mkdir(parents=True)
    finish = setting('End time', .0125)
    guide_data = run(args, 'guide', excerpt[1] + section('Mesh refinement', setting('Initial global refinement', 0))
                     + finish, template=template)
    original = run(args, 'control', finish)
    require(sorted(guide_data) == [0, 1] and guide_data[1][0] == (.0125, .0125), 'missing first physical step')
    print('Guide example at 0 and 0.0125 s, both fields, maximum coefficient difference', compare(guide_data, original, 0.))


def main():
    def path(value):
        return Path(value).resolve()

    parser = argparse.ArgumentParser(description=__doc__)
    cases = parser.add_subparsers(dest='case', required=True)
    for name, action in [('restart', restart), ('stock-picard', stock_picard), ('moving-retry', moving_retry),
                         ('guide', guide_example)]:
        case = cases.add_parser(name, help=action.__doc__)
        for key in ('aspect', 'plugins', 'directory'):
            case.add_argument(key, type=path)
        if name == 'moving-retry':
            case.add_argument('--layered-plugin', type=path, required=True)
        if name in ('restart', 'stock-picard'):
            case.add_argument('--writer', type=path, required=True)
            case.add_argument('--writer-plugins', type=path, required=True)
        case.set_defaults(action=action, mode='release')
    args = parser.parse_args()
    try:
        args.action(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
