"""Common-time contraction, actual adaptive controller, and restored-history checks."""
import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np
from run_startup_followup import OUT, STUDY
from check_startup_30km import table, check, raw_qps, load_parts, difference


def dump(name, value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')


def lifecycle(case):
    path = OUT/case
    assert json.loads((path/'execution.json').read_text())['passed']
    if case == 'resume':
        return check('startup-followup/resume', accepted_prefix=True,
                     preceding_state=table(OUT/'adaptive/state_work_2.csv'), initialized_path=OUT/'adaptive')
    result = check('startup-followup/'+case, accepted_prefix=True)
    a = table(path/'state_work_0.csv')
    b = table(STUDY/'small-startup/300/state_work_0.csv')
    for key in ['xd','V','Theta_in','Theta_out','slip','weak_q','weak_sigma']:
        np.testing.assert_array_equal(a[key],b[key])
    return result


def temporal():
    life = lifecycle('75')
    np.testing.assert_array_equal(table(OUT/'75/accepted_steps.csv')['time'],np.arange(0,301,75))
    states = {300:table(STUDY/'small-startup/300/state_work_1.csv'),
              150:table(STUDY/'small-startup/150/state_work_2.csv'),
              75:table(OUT/'75/state_work_4.csv')}
    weak = table(OUT/'75/work_weak_4.csv')
    regions = {}
    for name, lo, hi in [('weakening',0,30000),('transition',28000,35000),('deep',35000,115471)]:
        mask = (states[75]['xd']>=lo)&(states[75]['xd']<=hi)
        w = weak['weight'][mask]
        region = {}
        for key in ['V','slip','Theta_out']:
            changes = []
            for coarse,fine in [(300,150),(150,75)]:
                d = (states[coarse][key]-states[fine][key])[mask]
                changes.append(dict(max_abs=float(max(abs(d))),
                    weighted_RMS=float(np.sqrt(np.dot(w,d*d)/sum(w))),
                    relative_max=float(max(abs(d))/max(abs(states[fine][key][mask])))))
            region[key] = dict(d300_d150=changes[0],d150_d75=changes[1],
                max_contraction=changes[0]['max_abs']/changes[1]['max_abs'],
                rms_contraction=changes[0]['weighted_RMS']/changes[1]['weighted_RMS'])
        regions[name] = region
    result = dict(passed=True,time=300,lifecycle=life,regions=regions)
    dump('temporal_300s.json',result)
    print(json.dumps(regions,indent=2))


def predictor(path):
    p = path/'state_startup_predictor.csv'
    # The already-tested plugin opens an append-only stream on resume; a new
    # branch has no header. Interpret its fixed schema, without altering output.
    header = 'accepted_step,time,maximum_dt,proposed_dt,measure,limit'
    if p.open().readline().strip() == header:
        return table(p)
    return np.atleast_1d(np.genfromtxt(p,delimiter=',',names=header.split(',')))


def adaptive():
    life = lifecycle('adaptive')
    path = OUT/'adaptive'
    clock = table(path/'accepted_steps.csv')
    np.testing.assert_array_equal(clock['step'],np.arange(5))
    arrays = {a.attrib['Name']:np.fromstring(a.text,sep=' ') for a in ET.parse(
        path/'reconstructed_faults/reconstructed_faults-00000.vtu').findall('.//PointData/DataArray')}
    f = np.clip(arrays['composition_strengthening'],0,1)
    ratio = .03/(.004*(1-f)+.04*f)
    rows = predictor(path)
    records = []
    for row in rows:
        k = int(row['accepted_step'])
        s = table(path/f'state_work_{k}.csv')
        dt = row['proposed_dt']
        x = s['V']*dt/.1
        theta = s['Theta_out']*np.exp(-x)-.1/s['V']*np.expm1(-x)
        measured = float(max(ratio*abs(np.log(theta/s['Theta_out']))))
        assert row['maximum_dt'] == 4e6 and row['limit'] == .02
        assert 0 < dt < 4e6 and abs(measured-.02)<2e-13
        assert abs(measured-row['measure'])<2e-13
        entry = dict(accepted_step=k,time=float(row['time']),next_dt=float(dt),measure=measured)
        if k < 4:
            np.testing.assert_allclose(clock['dt'][k+1],dt,rtol=1e-13,atol=0.)
            nxt = table(path/f'state_work_{k+1}.csv')
            entry['realized_weighted_state_change'] = float(max(ratio*abs(np.log(nxt['Theta_out']/nxt['Theta_in']))))
        records.append(entry)
    assert len(records)==5
    dump('adaptive_checks.json',dict(passed=True,lifecycle=life,predictor=records))
    print(json.dumps(records,indent=2))


def restart():
    life = lifecycle('resume')
    a, b = OUT/'adaptive', OUT/'resume'
    assert set(map(int,life['convergence'])) == {3,4}
    assert not (b/'weak_initialization.csv').exists()
    assert not list(b.glob('initial_mesh_*.csv'))
    clocks = [table(p/'accepted_steps.csv') for p in [a,b]]
    for key in ['step','free','lower_active']:
        np.testing.assert_array_equal(clocks[0][key],clocks[1][key])
    failures = []
    def compare(x,y,label):
        # Preserve the established 1e-8 per-component comparison. Collect all
        # discrepancies instead of hiding later lifecycle evidence at the first.
        try:
            return dict(passed=True,**difference(x,y))
        except AssertionError:
            absolute = float(np.max(abs(x-y)))
            scale = float(np.max(abs(x)))
            failures.append(label)
            return dict(passed=False,absolute=absolute,scale=scale,
                        relative=absolute/scale if scale else None)
    result = dict(lifecycle=life,weak_initialization_repeated=False,steps={},controller={})
    for key in ['time','dt']:
        result['controller'][key] = compare(clocks[0][key],clocks[1][key],key)
    pa,pb = predictor(a),predictor(b)
    np.testing.assert_array_equal(pb['accepted_step'],[3,4])
    for key in ['time','maximum_dt','proposed_dt','measure','limit']:
        result['controller'][key+'_after_resume'] = compare(pa[key][3:],pb[key],key)
    # The next dt at the checkpoint is serialized; subsequent proposals must
    # instead be rebuilt from the restored and newly evolved histories.
    result['checkpoint_next_dt'] = compare(np.array([pa['proposed_dt'][2]]),
        np.array([clocks[1]['dt'][3]]),'checkpoint next dt')
    for k in [3,4]:
        entry = {}
        for pattern in [f'audit_bulk_{k}_rank*.csv',f'audit_particles_{k}_rank*.csv']:
            x,y = load_parts(a,pattern),load_parts(b,pattern)
            np.testing.assert_array_equal(x[:,0],y[:,0])
            if 'bulk' in pattern:
                np.testing.assert_array_equal(x[:,1],y[:,1])
                entry[pattern] = {str(int(c)):compare(x[x[:,1]==c,2],y[y[:,1]==c,2],f'{k} bulk {c}') for c in np.unique(x[:,1])}
            else:
                entry[pattern] = {str(c):compare(x[:,c],y[:,c],f'{k} particles {c}') for c in range(1,x.shape[1])}
        x,y = table(a/f'state_work_{k}.csv'),table(b/f'state_work_{k}.csv')
        entry['fault'] = {key:compare(x[key],y[key],f'{k} {key}') for key in
            ['V','Theta_in','Theta_out','slip','weak_q','weak_sigma','weak_friction']}
        entry['weak_residual_absolute_difference'] = float(max(abs(x['weak_residual']-y['weak_residual'])))
        x,y = [table(p/'profiles'/f'fault_{k}.csv') for p in [a,b]]
        for key in ['fault','node','xd_m','x_m','y_m']:
            np.testing.assert_array_equal(x[key],y[key])
        entry['profile'] = {key:compare(x[key],y[key],f'{k} profile {key}') for key in
            ['V_m_per_s','Theta_s','slip_m','q_weak_Pa','sigma_n_weak_Pa']}
        x,y = table(a/f'work_weak_{k}.csv'),table(b/f'work_weak_{k}.csv')
        entry['work_measure'] = {key:compare(x[key],y[key],f'{k} weak {key}') for key in ['weight','bg']}
        def raw(path):
            d = np.concatenate([raw_qps(p) for p in sorted(path.glob(f'work_qp_{k}_rank*.csv'))])
            return np.sort(d,order=['cell','qp'])
        x,y = raw(a),raw(b)
        for key in ['cell','qp','x','y','source_active','segment','xi']:
            np.testing.assert_array_equal(x[key],y[key])
        entry['constitutive'] = {key:compare(x[key],y[key],f'{k} QP {key}') for key in
            ['phi','Ih','chi','V','p','tau_xx','tau_yy','tau_xy','sigma_n','q']}
        result['steps'][k] = entry
    result.update(passed=not failures,failed_comparisons=failures)
    dump('restart_checks.json',result)
    print(json.dumps(dict(passed=result['passed'],failures=failures,controller=result['controller']),indent=2))
    assert not failures, 'Preserve and explain discrepancy; do not change comparison tolerance'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['temporal','adaptive','restart'])
    args = parser.parse_args()
    {'temporal':temporal,'adaptive':adaptive,'restart':restart}[args.action]()
