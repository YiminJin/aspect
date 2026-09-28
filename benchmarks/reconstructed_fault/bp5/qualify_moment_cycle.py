"""Four-step moment gates and optional matched-geometry two-rank qualification."""
import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np
from analyze_moment_cycle import MODES, samples, independent_load


def read_rows(path):
    with path.open() as stream:
        return [{k: v if k == 'mode' else float(v) for k, v in r.items()}
                for r in csv.DictReader(stream)]


def main(root):
    result = dict(branches={}, comparisons={}, gates=[], mpi='not run', mpi_field_differences={})
    cases = list(MODES)
    if (root/'horizontal_moment_update-mpi2').exists():
        cases.append('horizontal_moment_update-mpi2')
    for case in cases:
        mode = case.removesuffix('-mpi2')
        directory = root/case
        out = directory/f'output-{mode}'
        execution = json.loads((directory/'execution.json').read_text())
        assert execution['returncode'] == 0, (case, execution)
        rows = read_rows(out/'summary.csv')
        assert [r['step'] for r in rows] == list(range(5))
        assert all(r['dt'] == .1 and r['relative'] < 1e-8 for r in rows)
        state_hash = (out/'initial_hash.txt').read_text().strip()
        assert int(state_hash.split()[0]) == 9216
        audit = []
        fresh_checks = []
        for line in (directory/'run.log').read_text(errors='replace').splitlines():
            if 'Fault compatibility audit:' in line:
                fields = line.split('Fault compatibility audit:')[1].split(', ')
                audit.append({k.strip(): float(v) for k, v in (f.split('=') for f in fields)})
            if 'Fault linear solve:' in line:
                match = re.search(r'iterations=(\d+), fresh=([^, ]+), target=([^, ]+)', line)
                assert match, line
                fresh_checks.append(dict(iterations=int(match[1]), fresh=float(match[2]), target=float(match[3])))
        assert audit
        assert fresh_checks and all(r['fresh'] <= r['target'] for r in fresh_checks)
        for row in audit:
            def gamma(n):
                ne = n*np.finfo(float).eps
                return ne/(1-ne)
            row['assembly allowance'] = gamma(row['assembly operations'])*row['assembly scale']
            row['reduction allowance'] = gamma(row['reduction operations'])*row['reduction scale']
            expected = min(row['assembly allowance']+row['reduction allowance'], row['mixed target'])
            assert abs(expected-row['compatibility bound']) <= np.finfo(float).eps*expected
            assert abs(row['rhs null']) <= row['compatibility bound']
            assert row['q norm'] > .99
            assert row['right null'] <= 100*np.finfo(float).eps*row['right null scale']
            assert row['left null'] <= 100*np.finfo(float).eps*row['left null scale']
        flux = read_rows(out/'boundary_flux.csv')
        # Top/bottom normal velocities are prescribed identically zero; periodic
        # side-flux cancellation is measured rather than inferred from p alone.
        assert all(r['bottom'] == 0. and r['top'] == 0. for r in flux)
        assert max(abs(r['total']) for r in flux) < 1e-15
        info = dict(execution=execution, initial_particle_hash=state_hash, states=rows,
                    compatibility=audit, fresh_linear_checks=fresh_checks, boundary_flux=flux, independent_loads={})
        result['branches'][case] = info
        for step in range(1, 5):
            data = samples(out, 'fields', step, 15)
            history = samples(out, 'histories', step, 9)
            np.testing.assert_array_equal(data[:, :3], history[:, :3])
            jump = independent_load(data, history[:, 3:6]-data[:, 9:12])
            projection = independent_load(data, history[:, 6:9]-data[:, 9:12])
            mapping = independent_load(data, history[:, 3:6]-history[:, 6:9])
            error = abs(np.linalg.norm(jump)-rows[step]['Jtotal'])
            assert error < 2e-12, (case, step, error)
            info['independent_loads'][step] = dict(Jtotal=float(np.linalg.norm(jump)),
                difference=float(error), vector_closure=float(np.linalg.norm(jump-projection-mapping)),
                signed_dot=float(projection@mapping))

    assert len({b['execution']['binary_sha256'] for b in result['branches'].values()}) == 1
    assert len({b['execution']['plugin_sha256'] for b in result['branches'].values()}) == 1
    assert len({b['initial_particle_hash'] for b in result['branches'].values()}) == 1
    # Compare the unchanged production branch with the earlier accepted run.
    old = root.parent/'moment-consistency/production/output-production'
    old_rows = read_rows(old/'summary.csv')
    a, b, c = (result['branches'][m]['states'] for m in MODES)
    for step in range(5):
        da = samples(root/'production/output-production', 'fields', step, 15)
        prior = samples(old, 'fields', step, 15)
        np.testing.assert_allclose(da, prior, rtol=0, atol=1e-10)
        ref = samples(root/'native_history_reference/output-native_history_reference', 'fields', step, 15)
        differences = {}
        for case in ('production', 'horizontal_moment_update', *cases[3:]):
            mode = case.removesuffix('-mpi2')
            data = samples(root/case/f'output-{mode}', 'fields', step, 15)
            np.testing.assert_array_equal(data[:, :3], ref[:, :3])
            delta = data[:, 3:]-ref[:, 3:]
            differences[case] = dict(max_abs=np.max(abs(delta), axis=0).tolist(),
                weighted_RMS=np.sqrt(np.average(delta**2, axis=0, weights=ref[:, 2])).tolist())
        result['comparisons'][step] = differences
        if not step:
            continue
        assert abs(a[step]['Jtotal']-old_rows[step]['Jtotal']) < 1e-12
        bound = max(1e-9, 1e-10*c[step]['Fabs'])
        previous_b = b[step-1]['next_mean'] if step > 1 else 0.
        previous_c = c[step-1]['next_mean'] if step > 1 else 0.
        increments = abs((c[step]['next_mean']-previous_c)-(b[step]['next_mean']-previous_b))/abs(b[step]['next_mean']-previous_b)
        reactions = max(abs(c[step][k]-b[step][k])/abs(b[step][k]) for k in ('next_top', 'next_bottom'))
        gate = dict(step=step, B_jump=b[step]['Jtotal'], B_floor=b[step]['assembly_floor'],
                    C_jump=c[step]['Jtotal'], C_allowance=bound, C_reduction=a[step]['Jtotal']/c[step]['Jtotal'],
                    mean_increment_relative_difference=increments, reaction_relative_difference=reactions)
        assert b[step]['Jtotal'] <= b[step]['assembly_floor'], gate
        assert c[step]['Jtotal'] <= bound and gate['C_reduction'] >= 1e4, gate
        assert increments <= 1e-6 and reactions <= 1e-6, gate
        result['gates'].append(gate)
    if len(cases) == 4:
        parallel = result['branches'][cases[-1]]['states']
        for step in range(5):
            serial_data = samples(root/'horizontal_moment_update/output-horizontal_moment_update', 'fields', step, 15)
            parallel_data = samples(root/cases[-1]/'output-horizontal_moment_update', 'fields', step, 15)
            np.testing.assert_array_equal(serial_data[:, :3], parallel_data[:, :3])
            result['mpi_field_differences'][step] = dict(
                velocity_max=float(np.max(abs(parallel_data[:, 3:5]-serial_data[:, 3:5]))),
                current_tensor_max=float(np.max(abs(parallel_data[:, 9:12]-serial_data[:, 9:12]))))
            # Same physical-state tolerances as the original nonlinear fixture.
            np.testing.assert_allclose(parallel_data[:, 3:5], serial_data[:, 3:5], rtol=0, atol=1e-8*.005)
            if step:
                np.testing.assert_allclose(parallel_data[:, 9:12], serial_data[:, 9:12], rtol=0, atol=1e-8*2000)
                assert parallel[step]['Jtotal'] <= max(1e-9, 1e-10*parallel[step]['Fabs'])
                assert a[step]['Jtotal']/parallel[step]['Jtotal'] >= 1e4
            else:
                # T_0 is an evaluated initialization response, not retained
                # history. Verify the actual retained zero tensor on both ranks;
                # retain the larger T_0 response difference explicitly above.
                for case in ('horizontal_moment_update', cases[-1]):
                    history = samples(root/case/'output-horizontal_moment_update', 'histories', 0, 9)
                    np.testing.assert_array_equal(history[:, 3:6], 0.)
        result['mpi'] = 'passed: coordinates, initial hash, physical fields and publication gates'
    result['status'] = 'four-step gates passed; '+result['mpi']
    with (root/'compatibility_audit.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['branch', *result['branches']['production']['compatibility'][0]])
        writer.writeheader()
        for case, branch in result['branches'].items():
            for row in branch['compatibility']:
                writer.writerow(dict(branch=case, **row))
    (root/'qualification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], gates=result['gates']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
