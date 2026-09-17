"""Offline equal-trace recovery and within-free-interval BP3 discriminants.

Use saved production QPs and responses. No mechanics or history is advanced.
The explicit deep endpoint exists only in this algebraic reconstruction;
production eliminates its prescribed constant value through node 794.
"""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-trace-mpl')
import numpy as np
from analyze_uniform_sliding import cat, read, records, select, write
from analyze_free_trace import weak

HERE = Path(__file__).resolve().parent
A = HERE/'within-step-50-local4/A'
B = HERE/'free-trace/independent-local4-mpi'
OUT = B/'qualification'
VP = 1e-9


def main():
    OUT.mkdir(exist_ok=True)
    ea, eb = [json.loads((p/'execution.json').read_text()) for p in (A, B)]
    assert ea['verified'] and eb['converged'] and eb['rollback']
    assert ea['fresh_linear_passed'] and eb['fresh_linear_passed']
    assert not ea['accepted_output_written'] and not eb['accepted_output_written']
    assert ea['checkpoint_sha256'] == eb['checkpoint_sha256']
    for rank in range(4):
        assert (A/f'incoming_rank{rank}.txt').read_bytes() == (B/f'incoming_rank{rank}.txt').read_bytes()
    for name, digest in eb['checkpoint_sha256'].items():
        for p in (HERE/'work-replay-50-local4/restart/01', B/'restart/01'):
            assert hashlib.sha256((p/name).read_bytes()).hexdigest() == digest
    va, vb = [read(p/'noncommitting_surface.csv') for p in (A, B)]
    np.testing.assert_array_equal(va['time'], vb['time'])
    old = read(HERE/'work-replay-50-local4/fault_9.csv')
    raw = cat(B.glob('state_qp_rank*.csv'), ('cell',))
    j, xi = raw['segment'].astype(int), raw['xi']
    n = len(vb['V'])
    deep = j == 794
    np.testing.assert_array_equal(raw['shape1'], np.where(deep, 0., xi))
    np.testing.assert_array_equal(raw['shape0'], 1-raw['shape1'])
    assert np.all(raw['weight'] > 0)

    # Each real bulk QP occurs on exactly one MPI owner and one segment.
    keys = set()
    for path in sorted(B.glob('state_qp_rank*.csv')):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                key = row['cell'], int(row['qp'])
                assert key not in keys, key
                keys.add(key)
    assert len(keys) == len(j)

    # At the saved A rates both traces equal Vp. Reconstruct the scalar
    # source coefficient at every exported QP; S and geometry are unchanged.
    assert va['V'][794] == va['V'][795] == VP
    original_v = (1-xi)*va['V'][j]+xi*va['V'][j+1]
    eliminated_v = raw['shape0']*va['V'][j]+raw['shape1']*va['V'][j+1]
    source_error = float(max(abs(raw['chi']*(original_v-eliminated_v))))
    source_scale = float(max(abs(raw['chi']*original_v)))
    assert source_error/source_scale < 3e-15
    theta = (1-xi)*old['Theta'][j]+xi*old['Theta'][j+1]
    np.testing.assert_allclose(raw['Theta'], theta, rtol=3e-14)

    # Expand the eliminated deep trace as virtual index n. P ties n to 795
    # to recover continuous Q1; E ties n to 794 for the implemented constant
    # deep segment. Test before prescribed-row replacement, for every load.
    recovery = []
    columns = dict(mass=np.ones(len(j)), **{key: raw[key] for key in
                   ('q', 'friction', 'damping', 'sigma', 'R')})
    direction = np.random.default_rng(794795).normal(size=n)
    columns['K_direction'] = raw['Kfixed']*((1-xi)*direction[j]+xi*direction[j+1])
    for name, density in columns.items():
        weighted = raw['weight']*density
        split = (np.bincount(j, weights=(1-xi)*weighted, minlength=n+1)+
                 np.bincount(np.where(deep, n, j+1), weights=xi*weighted, minlength=n+1))
        continuous = split[:n].copy()
        continuous[795] += split[n]
        original = weak(raw, density, n, native=False)
        eliminated = split[:n].copy()
        eliminated[794] += split[n]
        native = weak(raw, density, n)
        scale = max(abs(original))
        error = float(max(abs(continuous-original))/scale)
        elimination_error = float(max(abs(eliminated-native))/max(abs(native)))
        assert error < 3e-13 and elimination_error < 3e-13
        recovery.append(dict(load=name, continuous_recovery_relative_error=error,
                             native_elimination_relative_error=elimination_error,
                             virtual_deep_endpoint_row=float(split[n])))
    records(OUT/'equal_trace_recovery.csv', recovery)

    # Distinguish a free-interval local minimum from the physical-side jump.
    metrics = []
    for name, data in [('A', va), ('B', vb)]:
        vleft, vmin, vright = data['V'][797:794:-1]/VP
        metrics.append(dict(case=name, V39900=vleft, V39950=vmin, V40000_free=vright,
                            V40000_deep=1., left_drop=vleft-vmin, right_rise=vright-vmin,
                            below_both_free_neighbors=min(vleft,vright)-vmin,
                            below_free_chord=.5*(vleft+vright)-vmin,
                            deep_minus_free_jump=1-vright))
    records(OUT/'free_interval_undershoot.csv', metrics)

    # Compare tensors at matching physical points, retaining signed normal
    # components. Existing raw exports are current constitutive stresses.
    baseline = cat((HERE/'work-replay-50-local4').glob('work_qp_10_rank*.csv'), ('cell',))
    baseline = select(baseline, (baseline['source_active'] == 1)&(baseline['chi'] > 0)&
                      (baseline['xd'] >= 37000)&(baseline['xd'] <= 43000))
    xd = (1-xi)*old['xd'][j]+xi*old['xd'][j+1]
    trial = select(raw, (xd >= 37000)&(xd <= 43000))
    baseline = select(baseline, np.lexsort((baseline['x'], baseline['y'])))
    trial = select(trial, np.lexsort((trial['x'], trial['y'])))
    for key in ('x', 'y', 'phi', 'chi'):
        np.testing.assert_array_equal(baseline[key], trial[key])
    np.testing.assert_allclose(baseline['JxW']*baseline['chi'], trial['weight'], rtol=2e-12)
    paired = dict(xd=baseline['xd'], x=baseline['x'], y=baseline['y'], weight=trial['weight'])
    baseline['sigma'] = baseline['sigma_n']
    for key in ('p', 'tau_xx', 'tau_yy', 'tau_xy', 'sigma', 'q'):
        paired[key+'_A'] = baseline[key]
        paired[key+'_B'] = trial[key]
    write(OUT/'matched_constitutive_tensors.csv', paired)
    result = dict(reused_A=True, reused_B=True, new_mechanical_solves=0,
                  checkpoint_and_incoming_match=True, unique_production_QPs=len(keys),
                  deep_element_QPs=int(sum(deep)),
                  equal_trace_source_absolute_error=source_error,
                  equal_trace_source_relative_error=source_error/source_scale,
                  worst_continuous_row_recovery=max(r['continuous_recovery_relative_error'] for r in recovery),
                  worst_native_elimination=max(r['native_elimination_relative_error'] for r in recovery),
                  unchanged_Theta_relative_error=float(max(abs(raw['Theta']/theta-1))),
                  matched_stress_QPs=len(trial['x']),
                  undershoot=metrics,
                  below_both_reduction=1-metrics[1]['below_both_free_neighbors']/metrics[0]['below_both_free_neighbors'],
                  chord_defect_reduction=1-metrics[1]['below_free_chord']/metrics[0]['below_free_chord'])
    (OUT/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
