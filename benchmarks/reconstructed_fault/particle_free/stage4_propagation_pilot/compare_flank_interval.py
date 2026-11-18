"""Compare two trajectories at the same physical endpoint from a common parent."""
import os
for k in ['MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import argparse,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[4]
from scipy.ndimage import label as label_components
from phase import coefficients,read,fronts,basis,sample,H
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('parent',type=Path);p.add_argument('parent_step',type=int)
p.add_argument('coarse',type=Path);p.add_argument('other',type=Path)
p.add_argument('destination',type=Path)
args=p.parse_args()
def frame(folder,step):
    s=read(folder/f'pilot-sections-{step:04d}.csv')
    g=read(folder/f'tip-planes-{step:04d}.csv')
    e=read(folder/'bulk-accepted.csv');e=e[e['step']==step][0]
    b=coefficients(folder,step)
    return s,g,e,b,fronts(b,16)
def norm(b):
    q,w=np.polynomial.legendre.leggauss(4);u=basis((q+1)/2)
    vals=np.einsum('ai,yxij,bj->yxab',u,b,u,optimize=True)
    return H*np.sqrt(np.einsum('yxab,a,b->',vals**2,w/2,w/2))
par=args.parent_step
s0,g0,e0,b0,f0=frame(args.parent,par)
sc,gc,ec,bc,fc=frame(args.coarse,par+1)
other=read(args.other/'bulk-accepted.csv')
matching=other[np.isclose(other['time'],ec['time'],rtol=0,atol=1e-13)]
assert len(matching)==1
sf,gf,ef,bf,ff=frame(args.other,int(matching[0]['step']))
assert np.array_equal(s0['x'],sf['x']) and np.array_equal(sc['x'],s0['x'])
assert np.max(np.abs(gc['angle'][g0['formed']>0]-g0['angle'][g0['formed']>0]))==0
assert np.max(np.abs(gf['angle'][g0['formed']>0]-g0['angle'][g0['formed']>0]))==0
assert abs(ec['dt']-(ef['time']-e0['time']))<1e-13
other=other[(other['time']>e0['time']+1e-13)&(other['time']<=ef['time']+1e-13)]
r=dict(time=float(ec['time']),parent_time=float(e0['time']),compared_dt=[ec['dt'],ef['dt']],
       phase_inf_difference=float(abs(bf-bc).max()),relative_phase_increment_l2=norm(bf-bc)/norm(bc-b0),
       section_slip_inf_difference=float(abs(sf['slip_endpoint']-sc['slip_endpoint']).max()),
       new_formed=[int(np.sum((g['formed']>0)&(g0['formed']==0))) for g in (gc,gf)],
       formed_mismatches=int((gf['formed']!=gc['formed']).sum()),
       contour_advance_m=[f['right_x']-f0['right_x'] for f in (fc,ff)],
       integrated_slip_increment_m2=[s['ds']@s['slip_endpoint']-s0['ds']@s0['slip_endpoint'] for s in (sc,sf)],
       fracture_increment_J_m=[e['fracture']-e0['fracture'] for e in (ec,ef)],
       signed_work_defect_J_m=[float((e['Wext']-e['energy_change']-e['Db']-e['Dc']-e['Df']).sum()) for e in (np.atleast_1d(ec),other)],
       mechanical=[ec['mechanical_residual'],ef['mechanical_residual']],phase_kkt=[ec['phase_KKT'],ef['phase_KKT']])
# Both native branches use midpoint section rates. Integrate the finer stages
# over the same physical interval before comparing the rate histories.
assert len(other)==2 and abs(ec['dt']/2-other['dt']).max()<1e-14
coarse_rate=sc['rate_stage']
mean_rate=sum(read(args.other/f"pilot-sections-{int(e['step']):04d}.csv")['rate_stage']*e['dt'] for e in other)/ec['dt']
weight=g0['reference_dx']
r['relative_rms_mean_rate']=np.sqrt(weight@(mean_rate-coarse_rate)**2/(weight@coarse_rate**2))
r['physical_work_absolute_J_m']=[float(abs(e['Wext']-e['energy_change']-e['Db']-e['Dc']-e['Df']).sum()) for e in (np.atleast_1d(ec),other)]
initial=coefficients(root/'build-particle-free/stage4-propagation-pilot/analysis_views/through70',0)
intact=abs(initial).max(axis=(2,3))==0
r['intact_integral_m2']=[float(H**2*b[intact].mean(axis=(1,2)).sum()) for b in (b0,bc,bf)]
cells=(np.maximum(bc.max(axis=(2,3)),bf.max(axis=(2,3)))>1e-12)&intact
table=[]
bhalf=coefficients(args.other,int(other[0]['step']))
for y,x in np.argwhere(cells):
    row={'x':float((x+.5)*H),'y':float((y+.5)*H)}
    for label,b in [('parent',b0),('40us_mid',bhalf),('80us_end',bc),('40us_end',bf)]:
        row[label+'_mean']=float(b[y,x].mean());row[label+'_max']=float(b[y,x].max())
    table.append(row)
r['initially_intact_cells']=table
# Connectivity lives on both traces of the DG cells. Report sample counts
# separately from the exact Bernstein cell-mean increments above.
n=8
mask=np.repeat(np.repeat(intact,n,axis=0),n,axis=1)
r['intact_connectivity']={}
for tag,b in [('80us',bc),('40us',bf)]:
    values=sample(b,n)
    support={}
    for level in (1e-5,1e-4,1e-3,.01,.05):
        labels,_=label_components(values>=level)
        seed=labels[labels.shape[0]//2,labels.shape[1]//2]
        assert seed>0
        support[str(level)]=dict(connected=int(np.sum(mask&(labels==seed))),
                                 detached=int(np.sum(mask&(labels>0)&(labels!=seed))))
    r['intact_connectivity'][tag]=support
r['minimum_coefficient_increment']=[float((bc-b0).min()),float((bhalf-b0).min()),float((bf-bhalf).min())]
r['maximum_geometry_sampling_difference_m']=max(abs(fronts(b,8)['right_x']-f['right_x']) for b,f in [(bc,fc),(bf,ff)])
args.destination.write_text(json.dumps(r,indent=2)+'\n')
