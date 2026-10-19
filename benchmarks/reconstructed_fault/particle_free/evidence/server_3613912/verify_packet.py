"""Independent signed-load, restart and parameter checks; no ASPECT launch."""
from decimal import Decimal, localcontext
from pathlib import Path
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent


def parameters(path):
    stack, result = [], {}
    for raw in path.read_text().splitlines():
        line = raw.split('#', 1)[0].strip()
        if line.startswith('subsection '):
            stack.append(line[11:].strip())
        elif line == 'end':
            stack.pop()
        elif line.startswith('set '):
            key, value = line[4:].split('=', 1)
            result['/'.join(stack + [key.strip()])] = value.strip()
    assert not stack
    return result


def main():
    data = np.genfromtxt(HERE / 'work_consistency_failure_816.csv', delimiter=',', names=True)
    assert len(data) == 5776 and np.array_equal(data['node'], np.arange(5776))
    assert all(np.all(np.isfinite(data[k])) for k in data.dtype.names)
    assert np.all(data['mass_row_m'] > 0) and np.all(data['step'] == 816)
    result = {'strict_reference_match_passed': False, 'rows': len(data), 'errors': {}}
    for component in ('shear', 'raw_normal'):
        for frame in ('original', 'matching'):
            loads = data[f'{frame}_{component}_Pa_m']
            weak = data[f'mechanical_{component}_Pa_m']
            errors = data[f'{frame}_{component}_error_Pa']
            assert np.array_equal((loads - weak) / data['mass_row_m'], errors)
            # Exact binary operands, high-precision quotient: the difference
            # satisfies Sterbenz (positive loads within a factor two), so only
            # the division rounds. Verify the actual half-ULP rounding bound.
            with localcontext() as context:
                context.prec = 100
                for left, right, mass, actual in zip(loads, weak, data['mass_row_m'], errors):
                    assert 0.5 * right <= left <= 2 * right
                    exact = (Decimal.from_float(float(left)) - Decimal.from_float(float(right))) / Decimal.from_float(float(mass))
                    allowance = Decimal.from_float(math.ulp(float(actual))) / 2
                    assert abs(Decimal.from_float(float(actual)) - exact) <= allowance
            j = int(np.argmax(abs(errors)))
            result['errors'][f'{frame}_{component}'] = {'max_abs_Pa': float(abs(errors[j])),
                'signed_Pa': float(errors[j]), 'node': j, 'down_dip_m': float(data['xd_m'][j])}
        assert np.max(abs(data[f'matching_{component}_error_Pa'])) < 1e-5
        frame_term = data[f'direct_frame_{component}_Pa_m'] / data['mass_row_m']
        # Separate rounded weak-load integrals and direct QP frame contractions
        # need not cancel exactly. Report their closure rather than tune a bound.
        closure = data[f'original_{component}_error_Pa'] - data[f'matching_{component}_error_Pa'] - frame_term
        result[f'{component}_frame_closure_max_Pa'] = float(max(abs(closure)))
    assert result['errors']['original_raw_normal']['max_abs_Pa'] > 1e-5
    result['offending_node'] = {key: float(data[key][4918]) for key in data.dtype.names}
    result['normal_direct_frame_at_offending_node_Pa'] = float(data['direct_frame_raw_normal_Pa_m'][4918] / data['mass_row_m'][4918])
    # The same guard must still reject an inconsistent corrected observation.
    perturbed = data['matching_raw_normal_Pa_m'].copy()
    perturbed[4918] += 2e-5 * data['mass_row_m'][4918]
    result['deliberately_inconsistent_matching_error_Pa'] = float(max(abs((perturbed - data['mechanical_raw_normal_Pa_m']) / data['mass_row_m'])))
    assert result['deliberately_inconsistent_matching_error_Pa'] > 1e-5

    actual = np.genfromtxt(HERE / 'accepted_steps.csv', delimiter=',', names=True)
    old = np.genfromtxt(HERE.parent / 'server_3610007/accepted_steps.csv', delimiter=',', names=True)
    assert np.array_equal(actual['step'], np.arange(810, 816))
    assert actual.dtype.names == old.dtype.names
    assert all(np.array_equal(actual[k], old[k][actual['step'].astype(int)]) for k in actual.dtype.names)
    result['six_replayed_accepted_records_all_columns_bitwise_equal'] = True
    tag = (HERE / 'last_good_checkpoint.txt').read_text().strip()
    label = (HERE / 'bp3_accepted_state.txt').read_text().split()
    assert tag == '3' and label == ['815', '5172072493.8982706']
    result['verified_checkpoint'] = {'slot': '03', 'step': int(label[0]), 'time_s': label[1]}

    expected = parameters(HERE.parent / 'server_3610007_analysis/replay-3610007.prm')
    actual_prm = parameters(HERE / 'parameters.prm')
    source_root = '/work2/11463/yiminjin/stampede3/build/aspect/pf-rsf-dg/source'
    canonical = {k: v.replace(source_root, '$ASPECT_SOURCE_DIR') for k, v in actual_prm.items()}
    assert canonical == expected
    result['effective_parameters_match_after_source_directory_expansion'] = True
    names = ['accepted_steps.csv', 'parameters.prm', 'work_consistency_failure_816.csv',
             'last_good_checkpoint.txt', 'bp3_accepted_state.txt']
    result['supplied_sha256'] = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names}
    (HERE / 'packet-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
