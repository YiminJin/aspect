"""Compare saved moment-cycle branches; no simulations or tolerance changes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_clean_stress_cycle import weak_load

MODES = ('production', 'native_history_reference', 'horizontal_moment_update')


def samples(directory, stem, step, columns):
    data = np.concatenate([np.fromfile(p, dtype=np.float64).reshape(-1, columns)
                           for p in sorted(directory.glob(f'{stem}_{step}_rank*.bin'))])
    return data[np.lexsort((data[:, 1], data[:, 0]))]


def independent_load(data, tensor):
    qps = {i: dict(x=x, y=y, JxW=w,
                   cell=(int(x*64), int((y+.5)*64)))
           for i, (x, y, w) in enumerate(data[:, :3])}
    load = weak_load(qps, dict(enumerate(tensor)))
    return np.array([load[k] for k in sorted(load)])


def main(root):
    result = {'qualification': 'partial; B/C stopped by step-2 pressure compatibility guard',
              'two_rank_replay': 'not run; four-step qualification incomplete',
              'units': {'loads': 'Pa m', 'stress': 'Pa', 'velocity': 'm/s'},
              'branches': {}, 'matched_fields': {}}
    outputs = {}
    for mode in MODES:
        case = root/mode
        out = case/f'output-{mode}'
        outputs[mode] = out
        with (out/'summary.csv').open() as stream:
            rows = [{k: (v if k == 'mode' else float(v)) for k, v in r.items()}
                    for r in csv.DictReader(stream)]
        info = dict(execution=json.loads((case/'execution.json').read_text()),
                    initial_hash=(out/'initial_hash.txt').read_text().strip(),
                    accepted_real_steps=len(rows)-1, states=rows)
        info['initial_particle_hash_valid'] = int(info['initial_hash'].split()[0]) > 0
        initial = samples(out, 'fields', 0, 15)
        info['initial_native_fields_sha256'] = hashlib.sha256(initial.tobytes()).hexdigest()
        result['branches'][mode] = info
        info['independent_load_checks'] = {}
        for r in rows[1:]:
            step = int(r['step'])
            data = samples(out, 'fields', step, 15)
            if list(out.glob(f'histories_{step}_rank*.bin')):
                history = samples(out, 'histories', step, 9)
                np.testing.assert_array_equal(data[:, :3], history[:, :3])
                next_tensor, fit = history[:, 3:6], history[:, 6:9]
            elif list(out.glob(f'fields_{step+1}_rank*.bin')):
                following = samples(out, 'fields', step+1, 15)
                np.testing.assert_array_equal(data[:, :3], following[:, :3])
                next_tensor, fit = following[:, 12:15], None
            else:
                continue
            jump = independent_load(data, next_tensor-data[:, 9:12])
            measured = float(np.linalg.norm(jump))
            # Independent basis evaluation/assembly has a different summation order.
            assert abs(measured-r['Jtotal']) < 2e-12
            check = dict(Jtotal=measured, difference_from_native=measured-r['Jtotal'])
            if fit is not None:
                projection = independent_load(data, fit-data[:, 9:12])
                mapping = independent_load(data, next_tensor-fit)
                check.update(Jproj=float(np.linalg.norm(projection)),
                             Jmap=float(np.linalg.norm(mapping)),
                             proj_map_dot=float(projection@mapping),
                             vector_closure=float(np.linalg.norm(jump-projection-mapping)))
                check['native_squared_norm_identity_error'] = (
                    r['Jtotal']**2-r['Jproj']**2-r['Jmap']**2-2*r['proj_map_dot'])
            info['independent_load_checks'][str(step)] = check

    assert len({v['initial_native_fields_sha256'] for v in result['branches'].values()}) == 1
    baseline_jumps = [0.067293874632279677, 0.11688245550014403, 0.16848560632304388]
    baseline_roughness = [1.5445318143393172, 3.0890509854691146,
                          4.6339349340095302, 6.1799889669050732]
    actual = result['branches']['production']['states'][1:]
    np.testing.assert_allclose([r['Jtotal'] for r in actual[:3]], baseline_jumps,
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose([r['parent_rough'] for r in actual], baseline_roughness,
                               rtol=0, atol=1e-10)
    result['baseline_reproduction'] = 'passed: prior clean-cycle jumps and parent roughness'
    result['initial_hash_limitation'] = (
        'Original particle hash was taken before particle creation (0 0), so is invalid. '
        'Saved initial native fields instead agree byte-for-byte. The plugin hash timing '
        'has been corrected for future runs; no additional simulation was run.')
    # Initial evaluated stress is deliberately not committed; compare real steps
    # only for publication gates, and keep time zero explicitly in field checks.
    for step in (0, 1):
        ref = samples(outputs[MODES[1]], 'fields', step, 15)
        weights = ref[:, 2]
        matched = {}
        for mode in (MODES[0], MODES[2]):
            data = samples(outputs[mode], 'fields', step, 15)
            np.testing.assert_array_equal(data[:, :3], ref[:, :3])
            values = {}
            for name, interval in (('velocity', slice(3, 5)), ('gradient', slice(5, 9)),
                                   ('current_tensor_components', slice(9, 12)),
                                   ('incoming_tensor_components', slice(12, 15))):
                difference = data[:, interval]-ref[:, interval]
                np.testing.assert_array_equal(difference, 0)
                values[name] = dict(max_abs=float(np.max(abs(difference))),
                    component_RMS=np.sqrt(np.average(difference**2, axis=0, weights=weights)).tolist())
            matched[mode] = values
        result['matched_fields'][str(step)] = matched

    a, b, c = (result['branches'][mode]['states'][1] for mode in MODES)
    result['first_real_step_gates'] = dict(
        B_jump_over_independent_assembly_floor=b['Jtotal']/b['assembly_floor'],
        C_jump_reduction=a['Jtotal']/c['Jtotal'],
        C_absolute_allowance=max(1e-9, 1e-10*c['Fabs']),
        C_mean_relative_difference=abs(c['next_mean']-b['next_mean'])/abs(b['next_mean']),
        C_top_reaction_relative_difference=abs(c['next_top']-b['next_top'])/abs(b['next_top']),
        C_bottom_reaction_relative_difference=abs(c['next_bottom']-b['next_bottom'])/abs(b['next_bottom']),
        pass_first_step_only=(b['Jtotal'] <= b['assembly_floor'] and
                             c['Jtotal'] <= min(a['Jtotal']/1e4, max(1e-9, 1e-10*c['Fabs'])) and
                             all(abs(c[k]-b[k])/abs(b[k]) <= 1e-6
                                 for k in ('next_mean', 'next_top', 'next_bottom'))))
    target = root/'comparison.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['first_real_step_gates'], indent=2))
    print(f'Written {target}; no further simulation launched.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
