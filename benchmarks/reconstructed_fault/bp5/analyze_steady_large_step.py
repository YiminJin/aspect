"""Native weak-force and history-timed large-step qualification, not background-scaled errors."""
import argparse
import json
import numpy as np
from run_steady_large_step import OUT
from check_startup_30km import table
from check_first_cycle_restart import convergence


def check(case):
    path=OUT/case
    assert json.loads((path/'execution.json').read_text())['passed']
    conv=convergence(path/'run.log')
    clock=table(path/'accepted_steps.csv')
    init=table(OUT/'startup/steady_initialization.csv')
    assert np.all(clock['fresh_linear_checks_passed']==1)
    assert np.all(clock['free']==1156) and np.all(clock['lower_active']==0)
    assert np.max(clock['Theta_relative_error'])<1e-12
    previous=table(OUT/'startup/state_work_1.csv') if case!='startup' else None
    results={}
    for k in conv:
        s=table(path/f'state_work_{k}.csv')
        w=table(path/f'work_weak_{k}.csv')
        assert np.max(abs(w['bg']-init['background_load'])/w['weight'])<1e-5
        assert not (path/'weak_initialization.csv').exists()
        # Recover b/a from the initial native mixture is unnecessary: both
        # plateaus have known a, while the independently exported profile
        # supplies the actual projected transition mixture.
        import xml.etree.ElementTree as ET
        arrays={a.attrib['Name']:np.fromstring(a.text,sep=' ') for a in ET.parse(
            OUT/'startup/reconstructed_faults/reconstructed_faults-00000.vtu').findall('.//PointData/DataArray')}
        f=np.clip(arrays['composition_strengthening'],0,1)
        ratio=.03/(.004*(1-f)+.04*f)
        realized=0.
        predicted=0.
        if k==0:
            np.testing.assert_array_equal(s['Theta_in'],np.full(len(s),1e8))
            np.testing.assert_array_equal(s['Theta_out'],s['Theta_in'])
        else:
            np.testing.assert_array_equal(s['Theta_in'],previous['Theta_out'])
            dt=s['dt'][0]
            theta=s['Theta_in']*np.exp(-s['V']*dt/.1)-.1/s['V']*np.expm1(-s['V']*dt/.1)
            assert np.max(abs(theta/s['Theta_out']-1))<1e-12
            assert np.max(abs(s['slip']-(previous['slip']+dt*s['V'])))<1e-15
            pred=s['Theta_in']*np.exp(-previous['V']*dt/.1)-.1/previous['V']*np.expm1(-previous['V']*dt/.1)
            predicted=float(np.max(ratio*abs(np.log(pred/s['Theta_in']))))
            realized=float(np.max(ratio*abs(np.log(s['Theta_out']/s['Theta_in']))))
            assert predicted<=.02+1e-12
        results[k]=dict(time=float(s['time'][0]),dt=float(s['dt'][0]),
            V_range=[float(min(s['V'])),float(max(s['V']))],
            predicted_state_change=predicted,realized_state_change=realized)
        previous=s
    result=dict(passed=True,convergence=conv,states=results)
    (path/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def compare(one,half):
    a,b=OUT/one,OUT/half
    assert json.loads((a/'checks.json').read_text())['passed']
    assert json.loads((b/'checks.json').read_text())['passed']
    k=2  # First comparison step always starts after accepted step 1.
    x=table(a/f'state_work_{k}.csv')
    y=table(b/'state_work_3.csv')
    old=table(OUT/'startup/state_work_1.csv')
    first_half=table(b/'state_work_2.csv')
    np.testing.assert_array_equal(x['Theta_in'],first_half['Theta_in'])
    assert x['time'][0]==y['time'][0]
    weights=table(a/f'work_weak_{k}.csv')['weight']
    rms=lambda z:float(np.sqrt(np.dot(weights,z*z)/sum(weights)))
    results={}
    for name in ('V','slip','Theta_in','Theta_out','weak_q','weak_sigma','weak_friction'):
        u,v,initial=x[name],y[name],old[name]
        if name.startswith('weak_'):
            u=u/weights;v=v/weights;initial=initial/table(OUT/'startup/work_weak_1.csv')['weight']
        if name=='Theta_in':
            initial=old['Theta_out']
        error=u-v;change=v-initial
        results[name]=dict(max_absolute=float(max(abs(error))),weighted_RMS=rms(error),
            fine_evolving_change_max=float(max(abs(change))),fine_evolving_change_RMS=rms(change),
            error_over_change_RMS=rms(error)/rms(change) if rms(change) else None,
            error_over_change_max=float(max(abs(error))/max(abs(change))) if max(abs(change)) else None)
    result=dict(one=one,half=half,begin_time=float(old['time'][0]),end_time=float(x['time'][0]),
        identical_incoming_state=True,metrics=results,
        incoming_state_note='The final half-step consumes the first half-step update; its Theta_in is deliberately not the full-step Theta_in.')
    (OUT/f'comparison-{half}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case')
    p.add_argument('--one',default='startup')
    args=p.parse_args()
    check(args.case)
    if args.case.startswith('half'):
        compare(args.one,args.case)
