"""Offline audit of row averages, production consistent projection and friction input.

prepare: reconstruct the full tridiagonal mass from saved production QPs.
analyze: consume solutions from test_traction_projection's production solver.
No trajectories, constitutive updates or history writes are performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_inclined_moment import table, rms

MODES=('production','native_history_reference')
NAMES=('p','d','sigma','one','friction_input')


def assemble(root, mode, step):
    out=root/mode/f'output-{mode}'
    q=table(out/f'traction_{step}_rank0.csv');g=table(out/'geometry.csv')
    s=.5*g['x']+np.sqrt(3)/2*g['y'];n=len(s)
    # These are all associated, uniquely owned native QPs, not row averages.
    # manager.cc sets shape_1=xi and shape_0=1-xi for this open straight fault.
    w=q['w']*q['chi'];j=q['segment'].astype(int);N=(1-q['xi'],q['xi'])
    diagonal=np.zeros(n);off=np.zeros(n-1);rhs=np.zeros((n,len(NAMES)))
    values=np.column_stack([q['p'],q['d'],q['sigma'],np.ones(len(q)),np.full(len(q),1000.)])
    for i in (0,1):
        np.add.at(diagonal,j+i,w*N[i]*N[i])
        for c in range(len(NAMES)):np.add.at(rhs[:,c],j+i,w*N[i]*values[:,c])
    np.add.at(off,j,w*N[0]*N[1])
    M=np.diag(diagonal)+np.diag(off,1)+np.diag(off,-1)
    mass=M@np.ones(n)
    np.testing.assert_allclose(mass,rhs[:,3],rtol=5e-15,atol=1e-16)
    np.testing.assert_allclose(rhs[:,0]+rhs[:,1],rhs[:,2],rtol=1e-13,atol=1e-11)
    saved=table(out/f'weak_profile_{step}.csv')
    for c,name in enumerate(NAMES[:3]):
        np.testing.assert_allclose(rhs[:,c]/mass,saved[name],rtol=1e-13,atol=1e-10)
    return s,M,rhs,q


def prepare(root, target):
    target.mkdir(parents=True,exist_ok=False)
    provenance={};manifest=[];reference=None
    for mode in MODES:
        out=root/mode/f'output-{mode}'
        prm=(out/'parameters.prm').read_text()
        assert 'set Use adiabatic pressure in fault friction = true' in prm
        for step in (1,4):
            name=f'{mode}_{step}';manifest.append(name)
            s,M,rhs,q=assemble(root,mode,step)
            if reference is not None:np.testing.assert_array_equal(M,reference)
            reference=M
            # All rows including endpoints enter the inverse. Restricting M to
            # the plotted window would impose a different projection problem.
            with (target/f'{name}.input').open('w') as f:
                f.write(f'{len(s)} {len(NAMES)}\n')
                np.savetxt(f,np.diag(M)[None,:],fmt='%.17g')
                np.savetxt(f,np.diag(M,1)[None,:],fmt='%.17g')
                np.savetxt(f,rhs.T,fmt='%.17g')
            np.savetxt(target/f'{name}_assembly.csv',np.column_stack([s,np.diag(M),np.r_[0,np.diag(M,1)],
                np.r_[np.diag(M,1),0],rhs]),delimiter=',',fmt='%.17g',
                header='s,mass_diagonal,mass_left,mass_right,'+','.join('rhs_'+n for n in NAMES),comments='')
            paths=[out/f'traction_{step}_rank0.csv',out/'geometry.csv',out/'parameters.prm']
            provenance[name]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (target/'manifest.txt').write_text('\n'.join(manifest)+'\n')
    (target/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    base=Path(__file__).resolve().parent
    prm=(root/'production/run.prm').read_text()
    libs=f'{(root/"production/libbp5_moment_cycle.release.so").resolve()}, {base}/build/libtest_traction_projection.release.so'
    prm=prm.replace('set Additional shared libraries = ./libbp5_moment_cycle.release.so',
                    f'set Additional shared libraries = {libs}')
    (target/'validate.prm').write_text(prm)


def stats(v,s,mass,mask):
    mean=float(np.average(v[mask],weights=mass[mask]))
    a=(s[1:-1]-s[:-2])/(s[2:]-s[:-2])
    chord=v[1:-1]-(1-a)*v[:-2]-a*v[2:]
    return dict(mean=mean,mean_removed_rms=rms(v[mask]-mean,mass[mask]),
                chord_rms=rms(chord[mask[1:-1]],mass[1:-1][mask[1:-1]]),
                maximum_absolute=float(np.max(abs(v[mask]))))


def analyze(root,target):
    result=dict(interior='|s|<=0.2 m, same 12 nodes as previous comparison',
        definitions={'row':'D^-1 b, D=diag(M 1)','consistent':'M^-1 b, full open-fault matrix',
            'friction':'sigma_bg+p_ad=0+1000 Pa at every production point; not p+d'},
        checks={},comparisons={})
    fields={}
    for mode in MODES:
        for step in (1,4):
            name=f'{mode}_{step}';s,M,rhs,q=assemble(root,mode,step);mass=M@np.ones(len(s));mask=abs(s)<=.2
            projected=np.loadtxt(target/f'{name}.projected').T
            row=rhs/mass[:,None]
            independent=np.linalg.solve(M,rhs)
            np.testing.assert_allclose(projected,independent,rtol=2e-14,atol=2e-11)
            check=dict(matrix_condition=float(np.linalg.cond(M)),
                solve_residual_over_mass=float(np.max(abs(M@projected-rhs)/mass[:,None])),
                row_constant_error=float(np.max(abs(row[:,3]-1))),
                constant_error=float(np.max(abs(projected[:,3]-1))),
                constant_friction_error=float(np.max(abs(projected[:,4]-1000))),
                row_closure=float(np.max(abs(row[:,0]+row[:,1]-row[:,2]))),
                projection_closure=float(np.max(abs(projected[:,0]+projected[:,1]-projected[:,2]))),
                raw_closure=float(np.max(abs(q['p']+q['d']-q['sigma']))))
            assert max(check[k] for k in check if k!='matrix_condition')<1e-9
            result['checks'][name]=check
            fields[mode,step]={'row':row,'consistent':projected}
            np.savetxt(target/f'{name}_comparison.csv',np.column_stack([s,mass,row,projected]),delimiter=',',
                header='s,mass,'+','.join(f'{p}_{n}' for p in ('row','consistent') for n in NAMES),comments='')
    for step in (1,4):
        result['comparisons'][step]={}
        for method in ('row','consistent'):
            a=fields['production',step][method];b=fields['native_history_reference',step][method]
            result['comparisons'][step][method]={}
            for label,v in (('A',a),('B',b),('A-B',a-b)):
                result['comparisons'][step][method][label]={name:stats(v[:,c],s,mass,mask) for c,name in enumerate(NAMES) if name!='one'}
        # The constant pointwise friction normal input is exact and independent
        # of both pressure gauge and the observational mass inverse.
        result['comparisons'][step]['actual_friction_input']={'A_Pa':1000,'B_Pa':1000,'A_minus_B_Pa':0,'spatial_roughness_Pa':0}
    (target/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axs=plt.subplots(3,3,figsize=(12,9),sharex=True)
    labels=(r'$p$',r'$d=-n^T\tau n$',r'$p+d$')
    for col in range(3):
        for row,step in enumerate((1,4)):
            for mode,label,color in (('production','A','tab:blue'),('native_history_reference','B','tab:orange')):
                for method,style in (('row','-'),('consistent','--')):
                    axs[row,col].plot(s[mask],fields[mode,step][method][mask,col],style,
                        color=color,marker='.',label=f'{label}: {method}')
            axs[row,col].set_title(f'{labels[col]}, step {step} ({step*.1:g} s)')
        for method,color in (('row','black'),('consistent','purple')):
            difference=fields['production',4][method]-fields['native_history_reference',4][method]
            axs[2,col].plot(s[mask],difference[mask,col],'.-',color=color,label=method)
        axs[2,col].set_title(f'A − B, final: {labels[col]}')
        axs[2,col].set_xlabel('s from box centre (m)')
    axs[0,0].legend(fontsize=8);axs[2,0].legend(fontsize=8)
    axs[2,2].axhline(0,color='green',ls=':',label='actual friction input A−B');axs[2,2].legend(fontsize=7)
    for ax in axs.ravel():ax.grid(alpha=.25);ax.set_ylabel('Pa')
    fig.suptitle('Diagnostic mechanical traction: row averages vs production consistent-Q1 projection\n'
        'Actual friction normal input = 1000 Pa for A and B at both times; first-step A−B = 0',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(target/'traction_definition.png',dpi=170);plt.close(fig)
    print(json.dumps(result['comparisons'][4],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','analyze'))
    p.add_argument('root',type=Path);p.add_argument('target',type=Path);a=p.parse_args()
    (prepare if a.action=='prepare' else analyze)(a.root,a.target)
