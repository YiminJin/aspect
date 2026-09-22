"""Check fresh-start timing/aging and preserve the observed contact regression."""
import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import numpy as np
from run_small_startup import OUT, STUDY
from check_startup_30km import table, check


def state_change(v,theta,dt):
    x=v*dt/.1
    return theta*np.exp(-x)-.1/v*np.expm1(-x)


def startup(case):
    path=OUT/case
    assert json.loads((path/'execution.json').read_text())['passed']
    clock=table(path/'accepted_steps.csv');dt=int(case)
    np.testing.assert_array_equal(clock['time'],np.arange(0,901,dt))
    checked=check('small-startup/'+case,accepted_prefix=True)
    initial=table(path/'state_work_0.csv');baseline_initial=table(STUDY/'startup/state_work_0.csv')
    for field in ['xd','V','Theta_in','Theta_out','slip','weak_q','weak_sigma']:
        np.testing.assert_array_equal(initial[field],baseline_initial[field])
    arrays={a.attrib['Name']:np.fromstring(a.text,sep=' ') for a in ET.parse(
        path/'reconstructed_faults/reconstructed_faults-00000.vtu').findall('.//PointData/DataArray')}
    f=np.clip(arrays['composition_strengthening'],0,1);ratio=.03/(.004*(1-f)+.04*f)
    predictor=table(path/'state_startup_predictor.csv')
    report=[];previous=None
    for k in clock['step'].astype(int):
        s=table(path/f'state_work_{k}.csv')
        p=predictor[predictor['accepted_step']==k];assert len(p)==1
        predicted=state_change(s['V'],s['Theta_out'],p['proposed_dt'][0])
        measure=max(ratio*abs(np.log(predicted/s['Theta_out'])))
        assert abs(measure-p['measure'][0])<2e-13
        assert measure<=.1+2e-13
        r=dict(step=int(k),predictor_measure=float(measure),proposed_dt=float(p['proposed_dt'][0]),
               min_V_over_Vp=float(min(s['V']/1e-9)),max_V_over_Vp=float(max(s['V']/1e-9)))
        if previous is not None:
            predicted_in=state_change(previous['V'],s['Theta_in'],dt)
            r.update(incoming_predictor=float(max(ratio*abs(np.log(predicted_in/s['Theta_in'])))),
                     realized_state_measure=float(max(ratio*abs(np.log(s['Theta_out']/s['Theta_in'])))),
                     max_Theta_ratio=float(max(s['Theta_out']/s['Theta_in'])))
            assert r['incoming_predictor']<=.1+2e-13
        previous=s;report.append(r)
    result=dict(passed=True,scope='complete fresh small-step startup',states=report,lifecycle=checked)
    (path/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(report,indent=2))


def temporal():
    startup('300');startup('150');rows=[]
    for time in [0,300,600,900]:
        a=table(OUT/'300'/f'state_work_{time//300}.csv');b=table(OUT/'150'/f'state_work_{time//150}.csv')
        wa=table(OUT/'300'/f'work_weak_{time//300}.csv');wb=table(OUT/'150'/f'work_weak_{time//150}.csv')
        for name,lo,hi in [('weakening',0,30000),('transition',28000,35000),('deep',35000,115471)]:
            mask=(a['xd']>=lo)&(a['xd']<=hi);w=wa['weight'][mask]
            values={key:(a[key],b[key]) for key in ['V','Theta_out','slip']}
            values.update(delta_q=((wa['q']-wa['bg'])/wa['weight'],(wb['q']-wb['bg'])/wb['weight']),
                          delta_sigma=(wa['sigma']/wa['weight']-50e6,wb['sigma']/wb['weight']-50e6))
            entry=dict(time=time,region=name)
            for key,(u,v) in values.items():
                d=(u-v)[mask];scale=max(abs(v[mask]));entry[key]=dict(max_abs=float(max(abs(d))),
                    relative_max=float(max(abs(d))/scale) if scale else 0.,
                    weighted_RMS=float(np.sqrt(np.dot(w,d*d)/sum(w))))
            rows.append(entry)
    (OUT/'temporal.json').write_text(json.dumps(rows,indent=2)+'\n')


def bounds():
    path=OUT/'bounds';log=(path/'run.log').read_text();case=table(path/'nonlinear_bounds_2.csv')
    states=table(path/'state_work_1.csv');minimum=1e-20;results=[];node_rows=[]
    active_previous=set();contacts_previous=set();contacts_seen=set();released=set()
    chunks=re.split(r'\*\*\* Timestep (\d+):',log)
    body=next(chunks[i+1] for i in range(1,len(chunks),2) if chunks[i]=='2')
    searches=re.findall(r'accepted after (\d+) rejected candidates; alpha=([0-9.e+-]+)\.\n',body)
    original=(STUDY/'startup/run.log').read_text()
    oldchunks=re.split(r'\*\*\* Timestep (\d+):',original)
    oldbody=next(oldchunks[i+1] for i in range(1,len(oldchunks),2) if oldchunks[i]=='2')
    residual_pattern=r'after nonlinear iteration\s+(\d+): ([^,\n]+), ([^\n]+)'
    current=np.array([list(map(float,m)) for m in re.findall(residual_pattern,body)])
    baseline=np.array([list(map(float,m)) for m in re.findall(residual_pattern,oldbody)])
    np.testing.assert_allclose(current,baseline[:len(current)],rtol=2e-5,atol=1e-12)
    baseline_state=table(STUDY/'startup/state_work_1.csv')
    for field in ['V','Theta_in','Theta_out','slip']:
        np.testing.assert_allclose(states[field],baseline_state[field],rtol=1e-12,atol=0.)
    # Byte identity is a stronger, cheap check for the retained incoming bulk
    # and particle fields. Report it separately from rounded log agreement.
    incoming={p.name:hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256(
        (STUDY/'startup'/p.name).read_bytes()).hexdigest()
        for pattern in ['audit_bulk_1_rank*.csv','audit_particles_1_rank*.csv']
        for p in path.glob(pattern)}
    for iteration in np.unique(case['iteration']).astype(int):
        if iteration>=len(searches):break # A wall-capped pending direction is not an accepted update.
        r=case[case['iteration']==iteration];ids=r['vertex'].astype(int)
        fraction=np.full(len(r),np.inf);down=(r['dV']<0)&(r['lower_active']==0)&(r['prescribed']==0)
        fraction[down]=(r['V'][down]-minimum)/-r['dV'][down]
        alpha_max=min(1.,min(fraction));np.testing.assert_allclose(r['alpha_max'],alpha_max,rtol=2e-15,atol=0.)
        rejected,printed=searches[iteration]
        alpha=alpha_max*2.**(-int(rejected))
        assert abs(float(printed)/alpha-1)<5e-6 # Ordinary log has six significant digits.
        active=set(ids[r['lower_active']==1]);at_bound=set(ids[r['V']-minimum<=100*np.finfo(float).eps*np.maximum(minimum,abs(r['V']))])
        limiting=ids[fraction==min(fraction)] if min(fraction)<=1 else np.array([],dtype=int)
        new=at_bound-contacts_seen;returning=(at_bound-contacts_previous)&contacts_seen
        releases=contacts_previous-at_bound;released|=releases;contacts_seen|=at_bound
        for i in np.flatnonzero(down|(r['lower_active']==1)|(r['V']==minimum)):
            node_rows.append([iteration,ids[i],states['xd'][ids[i]],r['V'][i],r['dV'][i],fraction[i],
                int(r['lower_active'][i]),int(ids[i] in at_bound),int(ids[i] in limiting),alpha,alpha_max])
        following=case[case['iteration']==iteration+1]
        if len(following):
            np.testing.assert_array_equal(ids,following['vertex'].astype(int))
            trial=r['V']+alpha*r['dV'];contact=down&(fraction==alpha);trial[contact]=minimum
            allowance=16*np.finfo(float).eps*(abs(r['V'])+abs(alpha*r['dV']))
            assert np.all(abs(trial-following['V'])<=allowance)
            np.testing.assert_array_equal(following['V'][contact],np.full(sum(contact),minimum))
        results.append(dict(iteration=int(iteration),active_count=len(active),contact_count=len(at_bound),
            newly_contacted=sorted(map(int,new)),returned_contact=sorted(map(int,returning)),
            left_contact=sorted(map(int,releases)),newly_active=sorted(map(int,active-active_previous)),
            released_active=sorted(map(int,active_previous-active)),
            limiting_nodes=list(map(int,limiting)),alpha_max=float(alpha_max),accepted_alpha=float(alpha),
            rejected_candidates=int(rejected)))
        active_previous=active;contacts_previous=at_bound
    with (path/'contact_nodes.csv').open('w') as out:
        writer=csv.writer(out);writer.writerow(['iteration','node','xd','V','dV','contact_fraction','active','at_bound','limiting','accepted_alpha','alpha_max']);writer.writerows(node_rows)
    report=dict(original_failure_reproduced=bool(json.loads((path/'execution.json').read_text())['passed']),
        incoming_file_identity=incoming,
        captured_linearizations=len(current),
        comparison_residual_max_abs=float(np.max(abs(current-baseline[:len(current)]))),
        distinct_contacts=len(contacts_seen),distinct_left_contact=len(released),iterations=results)
    (path/'contact_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='iterations'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('case',choices=['300','150','temporal','bounds']);args=parser.parse_args()
    if args.case in ['300','150']:startup(args.case)
    elif args.case=='temporal':temporal()
    else:bounds()
