"""Audit the retained 30-degree midpoint specimen using native point and coefficient exports."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare import fields, read


ANGLE = np.pi/6
T = np.array([np.cos(ANGLE), np.sin(ANGLE)])
N = np.array([-T[1], T[0]])
G, ETA, DC = 1e6, 1e8, .01
KF = 26.6666666667/((8/3)*.25)
M = KF/(1000**2/(2*G))
CONTRACTION = np.array([1., 1., 1., 2.])


def shards(folder, stem):
    paths = sorted(folder.glob(stem+'-r*.csv'))
    assert len(paths) == 2
    return np.concatenate([read(p) for p in paths])


def tensor(points, label):
    return np.column_stack([points[label+i] for i in ('xx','yy','zz','xy')])


def inverse_compliance(phi):
    return np.divide((1-phi)**2, M*phi*(1+phi), out=np.zeros_like(phi), where=phi>0)


def age(theta, rate, duration):
    x = duration*np.abs(rate)/DC
    return theta*np.exp(-x)+duration*np.divide(-np.expm1(-x), x, out=np.ones_like(x), where=x>0)


def point_audit(old, end, folder, step, row, old_energy, maps):
    dt = row['dt']
    h = dt/2
    for key in ('cell','q','section','x','y','weight'):
        np.testing.assert_array_equal(old[key],end[key])
    w = end['weight']
    assert abs(sum(w)-2.) < 1e-12
    planes = read(folder/f'tip-planes-{step:04}.csv')
    rates = read(folder/f'tip-sections-{step:04}.csv')
    k = end['section']
    angle = ANGLE + planes['beta'][k]
    st = np.column_stack((-.5*np.sin(2*angle), .5*np.sin(2*angle),
                          np.zeros_like(angle), .5*np.cos(2*angle)))
    sig0, sig1, a0, a1 = (tensor(p,l) for l in ('tau','A') for p in (old,end))
    sig, a = (sig0+sig1)/2, (a0+a1)/2
    phi = (old['phi']+end['phi'])/2
    dphi = end['phi']-old['phi']
    np.testing.assert_allclose(phi,end['stage_phi'],atol=3e-15,rtol=0)
    np.testing.assert_allclose(a[:,3],end['stage_Axy'],atol=1e-15)
    np.testing.assert_allclose(sig[:,3],end['stage_tauxy'],atol=1e-13)
    normal_error=max(np.max(abs(sig[:,j]-(end['stage_normal_'+axis]+end['stage_pressure']))) for j,axis in enumerate('xy'))
    resolved = np.sum(sig*st*CONTRACTION,axis=1)
    q = resolved - rates['force'][k]
    np.testing.assert_allclose(q,end['qcoh'],atol=1e-11,rtol=1e-12)
    def coh_energy(p, A):
        assert np.all(A[p==0]==0)
        return G*inverse_compliance(p)*np.sum(A*A*CONTRACTION,axis=1)
    U0, U1 = coh_energy(old['phi'],a0), coh_energy(end['phi'],a1)
    assert max(abs(w@U0-old_energy['cohesive']),abs(w@U1-row['cohesive'])) < 1e-14
    assert np.all(phi<1)
    Y = M*(1+3*phi)/(1-phi)**3*q*q/(2*G)
    np.testing.assert_allclose(Y,end['Y'],atol=3e-13,rtol=1e-11)
    de = 2*np.sum(st*(a1-a0)*CONTRACTION,axis=1)
    e_mid = 2*np.sum(st*a*CONTRACTION,axis=1)
    point_rate = de/dt+G/ETA*e_mid
    np.testing.assert_allclose(point_rate,end['w'],atol=1e-16,rtol=2e-12)
    work = q*de
    terms = (w@work,w@(U1-U0), w@(Y*dphi))
    contribution = w*(work-(U1-U0)-Y*dphi)
    # r(phi) = (1 + 1/phi - 4/(1+phi))/M. The quadratic term alone
    # is midpoint-exact. The two rational terms admit a cancellation-free
    # endpoint-only expression for the midpoint chain-rule defect.
    def ratio_defect(c):
        lo, hi = old['phi']+c, end['phi']+c
        ok = (lo>0)&(hi>0)
        mismatch = a1[ok]/hi[ok,None]-a0[ok]/lo[ok,None]
        return (G/M)*np.sum(w[ok]*dphi[ok]*lo[ok]*hi[ok]/(lo[ok]+hi[ok])**2
                            *np.sum(mismatch**2*CONTRACTION,axis=1))
    inverse_phi, regular = ratio_defect(0.), -4*ratio_defect(1.)
    signed = contribution.sum()
    assert abs(inverse_phi+regular-signed)<1e-14
    assert abs(signed-row['phase_remainder'])<2e-14
    # Invert the stage Maxwell law. The area mean of strain rate equals that
    # of the affine, divergence-free all-wall velocity by the divergence theorem.
    strain_rate = (sig1-sig0)/(2*G*dt)+sig/(2*ETA)+st*point_rate[:,None]
    L = .02*np.outer(T,N)+.003*(np.outer(T,T)-np.outer(N,N))
    mean = w@strain_rate/sum(w)
    target = [L[0,0],L[1,1],0.,(L[0,1]+L[1,0])/2]
    reactions = shards(folder,f'tip-reactions-{step:04}')
    xy = np.column_stack((reactions['x']-1.,reactions['y']-.5))
    speed = (xy@L.T)[np.arange(len(xy)),reactions['component']]
    wall = dt*(speed@reactions['force'])
    elastic = w @ (np.sum((sig1-sig0)*(sig1+sig0)*CONTRACTION,axis=1)/(4*G))
    db = dt*(w@np.sum(sig*sig*CONTRACTION,axis=1))/(2*ETA)
    dc = dt*2*G/ETA*(w@coh_energy(phi,a))
    df = dt*(w@(point_rate*rates['force'][k]))
    physical = wall-elastic-terms[1]-(row['fracture']-old_energy['fracture'])-db-dc-df
    ref = row['Wext']-row['energy_change']-row['Db']-row['Dc']-row['Df']
    checks = dict(mean_strain_rate=max(abs(mean-target)),wall_velocity=max(abs(speed-reactions['stage_velocity'])),
                  wall_work=abs(wall-row['Wext']), age=max(abs(age(planes['parent_age'],rates['V'],dt)-planes['age'])),
                  slip=max(abs(planes['slip']-planes['parent_slip']-dt*rates['V'])),
                  signed_shear=max(abs(end['signed_shear']-old['signed_shear']-dt*point_rate)),
                  stress_mid=max(abs(sig[:,3]-end['stage_tauxy'])),viscous_bulk=abs(db-row['Db']),
                  viscous_cohesive=abs(dc-row['Dc']),friction_radiation=abs(df-row['Df']))
    V = rates['resolved_V']
    f = 3*np.arcsinh(V/.002*np.exp((.6+.02*np.log(.001*age(planes['parent_age'],V,h)/DC))/.015))+1000*V
    checks['stage_force']=max(abs(f-rates['force']))
    checks['stress_mid_normal']=normal_error
    assert normal_error<1e-10 and checks['signed_shear']<1e-14 and checks['slip']<1e-14
    assert max(checks[n] for n in ('wall_work','viscous_bulk','viscous_cohesive','friction_radiation'))<1e-12
    # Check the domain mean against the independent prescribed boundary gradient.
    assert checks['mean_strain_rate']<1e-12 and checks['wall_velocity']<1e-12
    assert checks['age']<1e-12 and checks['stage_force']<1e-8 and abs(physical-ref)<1e-12
    xy = np.column_stack((end['x']-1.,end['y']-.5))
    xi,z = xy@T,xy@N
    masks = {'behind_seed_center':xi<0,'old_taper':(xi>=0)&(xi<.5),'ahead':xi>=.5,
             'negative_normal':z<0,'positive_normal':z>=0,'previously_zero_phase':old['phi']==0,
             'zero_increment':dphi==0,'increment_0_to_0.001':(dphi>0)&(dphi<=.001),
             'increment_0.001_to_0.01':(dphi>.001)&(dphi<=.01),'increment_above_0.01':dphi>.01}
    parts={name:dict(signed=contribution[mask].sum(),absolute=abs(contribution[mask]).sum())
           for name,mask in masks.items()}
    cells,cell_index = np.unique(end['cell'],return_inverse=True)
    totals = np.bincount(cell_index, np.abs(contribution))
    grid=np.zeros((8,16))
    np.add.at(grid, ((end['y']*8).astype(int),(end['x']*8).astype(int)), abs(contribution))
    maps.append(grid)
    hot=[]
    for i in np.argsort(-totals)[:4]:
        mask = cell_index==i
        # |contribution|-weighted physical centroid of each of the largest cells.
        weight = abs(contribution[mask])/totals[i]
        hot.append(dict(cell=str(cells[i]),x=weight@end['x'][mask],y=weight@end['y'][mask],
                        signed=sum(contribution[mask]),abs_fraction=totals[i]/sum(abs(contribution)),
                        phi_old=weight@old['phi'][mask],delta_phi=weight@dphi[mask]))
    return dict(checks=checks,traction_work=terms[0],cohesive_energy_change=terms[1],stage_driving_work=terms[2],
                reconstructed_remainder=signed,recorded_remainder=row['phase_remainder'],
                inverse_phi_term=inverse_phi,regular_term=regular,positive=sum(contribution[contribution>0]),negative=sum(contribution[contribution<0]),
                physical_residual=physical,recorded_physical_residual=ref,regions=parts,hot_cells=hot)


def phase_audit(folder):
    b=np.sort(shards(folder,'tip-balance-0002'),order=['cell','i'])
    assert len(b)==128*9
    lower,stage=b['lower'],b['stage_B']
    upper=.5*(1+lower)
    np.testing.assert_allclose(b['B'],2*stage-lower,rtol=0,atol=5e-16)
    def prediction(gradient):
        return stage-gradient/(KF*b['mass'])
    pred=prediction(b['residual'])
    status=np.where(pred<=lower,-1,np.where(pred>=upper,1,0))
    kkt=max(abs(stage-np.clip(pred,lower,upper)))
    # The AT1/SIPG gradient is affine, so g_n = 2*g_m - g_end.
    # Remove mechanical driving from the phase rows while holding the accepted state.
    gn=2*b['stage_fracture_gradient']-b['fracture_gradient']
    d=b['B']-lower
    original=read(folder/'bulk-accepted.csv')[0]
    assert abs(kkt-original['phase_KKT'])<1e-15
    return b,status,dict(lower=int(sum(status<0)),free=int(sum(status==0)),upper=int(sum(status>0)),KKT=kkt,
                         exact_lower=int(sum(stage==lower)),min_endpoint_increment=min(d),
                         max_stage=max(stage),max_endpoint=max(b['B']),
                         fracture_only_parent_downhill_rows=int(sum((gn<0)&(lower<1))),
                         endpoint_fracture_only_downhill_rows=int(sum((b['fracture_gradient']<0)&(b['B']<1))),
                         fracture_change_to_stage=.25*(gn+b['stage_fracture_gradient'])@d,
                         endpoint_fracture_gradient_dot_delta=b['fracture_gradient']@d,
                         fracture_only_minimum_fraction_along_increment=-(gn@d)/((b['fracture_gradient']-gn)@d),
                         parent_fracture_gradient_dot_delta=gn@d,mid_fracture_gradient_dot_delta=b['stage_fracture_gradient']@d,
                         stage_driving_dot_delta=b['driving_load']@d,
                         fracture_only_stage_KKT=max(abs(stage-np.clip(prediction(b['stage_fracture_gradient']),lower,upper))))


def second_phase_increment(first, end, rows):
    """Apply the saved first-half endpoint fracture gradient to the actual next increment."""
    increments=np.empty(len(rows))
    for cell in np.unique(rows['cell']):
        mask=first['cell']==cell
        group=rows['cell']==cell
        np.testing.assert_array_equal(rows['i'][group],np.arange(9))
        x=np.mod(8*first['x'][mask],1)
        y=np.mod(8*first['y'][mask],1)
        def bern(s):return np.array([(1-s)**2,2*s*(1-s),s*s])
        basis=(bern(y)[:,None,:]*bern(x)[None,:,:]).reshape(9,-1).T
        np.testing.assert_allclose(basis@rows['B'][group],first['phi'][mask],rtol=0,atol=2e-15)
        delta=end['phi'][mask]-first['phi'][mask]
        inc,_,rank,_=np.linalg.lstsq(basis,delta,rcond=None)
        assert rank==9
        np.testing.assert_allclose(basis@inc,delta,rtol=0,atol=2e-15)
        increments[group]=inc
    neg=rows['fracture_gradient']<0
    return dict(parent_fracture_gradient_dot_second_increment=rows['fracture_gradient']@increments,
                released_rows_with_increment_above_1e_8=int(sum(increments>1e-8)),
                downhill_rows_with_increment_above_1e_8=int(sum(neg&(increments>1e-8))),
                fraction_of_phase_volume_on_downhill_rows=(rows['mass'][neg]@increments[neg])/(rows['mass']@increments))


def plot_cells(maps, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, axes=plt.subplots(1,3,figsize=(10,3.4),sharey=True,layout='constrained')
    titles=['5–10 ms, one step','5–7.5 ms, first half','7.5–10 ms, second half']
    for grid,title,ax in zip(maps,titles,axes):
        values=grid*1e6
        im=ax.imshow(np.ma.masked_less(values,1e-5), origin='lower', extent=(0,2,0,1),
                     interpolation='nearest',norm=LogNorm(1e-5,1),cmap='viridis')
        # Straight incoming centerline until it intersects the specimen's lower boundary.
        center=np.array([1.,.5])
        tip=center+.5*T
        start=center-.5/T[1]*T
        ax.plot([start[0],tip[0]],[start[1],tip[1]],color='black',lw=.8,ls='--')
        ax.plot(*tip, marker='o',mfc='white',mec='black',ms=4)
        ax.set(title=title,xlabel='x (m)',xlim=(0,2),ylim=(0,1))
    axes[0].set_ylabel('y (m)')
    fig.colorbar(im,ax=axes,orientation='horizontal',shrink=.72,pad=.15,
                 label='Sum of absolute pointwise midpoint remainder per cell (μJ/m)')
    fig.savefig(path,dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path)
    parser.add_argument('--figure',type=Path)
    args=parser.parse_args()
    r=args.runs
    def folder(n):return r/n/'output'
    base=folder('localized-oneo1')
    old=fields(base,1)
    ledger0=read(base/'bulk-accepted.csv')[0]
    out={}
    maps=[]
    for case in ('localized-oneo2','strip18-midpoint-half'):
        p0,row0=old,ledger0
        for row in read(folder(case)/'bulk-accepted.csv'):
            step=int(row['step'])
            p1=fields(folder(case),step)
            out[f'{case}-{step}']=point_audit(p0,p1,folder(case),step,row,row0,maps)
            p0,row0=p1,row
    a,sa,out['coarse_rows']=phase_audit(folder('strip18-audit-full'))
    b,sb,out['half_rows']=phase_audit(folder('strip18-audit-half1'))
    for key in ('cell','i','lower'):
        np.testing.assert_array_equal(a[key],b[key])
    out['contact_changes']=[dict(cell=str(a['cell'][i]),local=int(a['i'][i]),coarse=int(sa[i]),half=int(sb[i]),
                                 coarse_increment=a['B'][i]-a['lower'][i],half_increment=b['B'][i]-b['lower'][i])
                            for i in np.flatnonzero(sa!=sb)]
    half=folder('strip18-midpoint-half')
    out['second_increment']=second_phase_increment(fields(half,2),fields(half,3),b)
    out['max_stage_B_difference']=max(abs(a['stage_B']-b['stage_B']))
    if args.figure:
        plot_cells(maps,args.figure)
    print(json.dumps(out,indent=2))
