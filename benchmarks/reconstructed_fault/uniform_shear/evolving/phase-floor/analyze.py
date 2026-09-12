"""Frozen phase audit: constrained actions, cancellation and extended precision."""
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from reference import read_parameters

here=Path(__file__).resolve().parent
data=here/'output'
parameters=read_parameters(data/'parameters.prm')
ell=float(parameters['Phase field model/Length scale'])
Ec=float(parameters['Material model/Phase field fault/Critical energy release rates'])/((8/3)*ell)
Hc=float(parameters['Material model/Phase field fault/Cohesions'])**2/(2*float(parameters['Material model/Phase field fault/Elastic shear moduli']))
m=Ec/Hc


def norm(x): return float(np.sqrt(np.sum(x*x)))


report=[]
for step in (1,2):
    prefix=data/f'audit{step}'
    raw=np.loadtxt(str(prefix)+'_weights.csv',delimiter=',',skiprows=1)
    ids,dofs=raw[:,0].astype(int),raw[:,1].astype(int)
    starts=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1]
    group=np.repeat(np.arange(len(starts)),np.diff(np.r_[starts,len(ids)]))
    size=max(dofs)+1
    master=np.arange(size)
    constraints=np.atleast_2d(np.loadtxt(str(prefix)+'_constraints.csv',delimiter=',',skiprows=1))
    assert np.all(constraints[:,2]==1), 'This fixed-box diagnostic has only periodic identity phase constraints.'
    for slave,parent,_ in constraints: master[int(slave)]=int(parent)
    free=np.unique(master[dofs])
    def reduce(values):
        result=np.zeros(size,dtype=values.dtype)
        np.add.at(result,master[dofs],values)
        return result

    def evaluate(nodal,direction=None,dtype=np.longdouble):
        weights=raw[:,2].astype(dtype); grad=raw[:,3:5].astype(dtype)
        volume=raw[:,5].astype(dtype); H=raw[:,6].astype(dtype)
        values=nodal[dofs].astype(dtype)
        phi=np.add.reduceat(values*weights,starts)[group]
        dphi=np.add.reduceat(values[:,None]*grad,starts,axis=0)[group]
        A=(1-phi)**2+m*phi*(1+phi)
        gp=-m*(1-phi)*(1+3*phi)/(A*A)
        B=(1-phi)*(1+3*phi)
        Ap=2*(phi-1)+m*(1+2*phi)
        Bp=2*(1-3*phi)
        gpp=m*(2*B*Ap-Bp*A)/(A*A*A)
        F=2*Ec*ell*ell
        reaction=-weights*(H*gp+Ec)*volume
        gradient=-F*np.sum(grad*dphi,axis=1)*volume
        rhs=reduce(reaction+gradient)
        # A positive pre-cancellation scale, including the sensitivity of
        # phase/gradient interpolation to representable nodal values.
        absphi=np.add.reduceat(abs(values*weights),starts)[group]
        absgrad=np.add.reduceat(abs(values[:,None]*grad),starts,axis=0)[group]
        scale=reduce(volume*(abs(weights)*(abs(H*gp)+Ec+abs(H*gpp)*absphi)
                     +F*np.sum(abs(grad)*absgrad,axis=1)))
        action=None
        if direction is not None:
            p=np.add.reduceat(direction[dofs].astype(dtype)*weights,starts)[group]
            g=np.add.reduceat(direction[dofs,None].astype(dtype)*grad,starts,axis=0)[group]
            action=reduce(volume*(weights*H*gpp*p+F*np.sum(grad*g,axis=1)))
        return rhs,reduce(reaction),reduce(gradient),scale,action

    histories=np.genfromtxt(str(prefix)+'_iterations.csv',delimiter=',',names=True)
    iterations=sorted(set([0,int(histories[-1]['iteration'])]))
    for iteration in iterations:
        files=sorted(data.glob(f'audit{step}_i{iteration}_a*.csv'))
        for path in files:
            rows=np.loadtxt(path,delimiter=',',skiprows=1)
            indices=rows[:,0].astype(int)
            fields=[]
            for col in (4,5,6,7,8,9,10):
                field=np.zeros(size);field[indices]=rows[:,col];fields.append(field)
            state,rhs,update,represented,Jrep,trial,trial_rhs=fields
            delta=trial.astype(np.longdouble)-state.astype(np.longdouble)
            fresh,rxn,grad,scale,action=evaluate(state,delta)
            fresh_trial,_,_,_,_=evaluate(trial)
            alpha=float(path.stem.split('_a')[-1])
            ideal_trial=state.astype(np.longdouble)+alpha*update.astype(np.longdouble)
            ideal,_,_,_,ideal_action=evaluate(ideal_trial,alpha*update.astype(np.longdouble))
            low,_,_,_,_=evaluate(state,dtype=np.float64)
            error=trial_rhs-rhs+Jrep
            row=dict(step=step,iteration=iteration,alpha=float(path.stem.split('_a')[-1]),
                production_residual=norm(rhs[free]),extended_residual=norm(fresh[free]),
                production_vs_extended=norm((rhs-fresh)[free]),
                offline_double_vs_extended=norm((low-fresh)[free]),
                reaction=norm(rxn[free]),gradient=norm(grad[free]),
                reaction_gradient_cancellation=(norm(rxn[free])+norm(grad[free]))/max(norm(fresh[free]),1e-300),
                term_scale=norm(scale[free]),epsilon_term_scale=np.finfo(float).eps*norm(scale[free]),
                update=norm(update[free]),represented=norm(delta[free]),Jrepresented=norm(Jrep[free]),
                J_action_vs_extended=norm((Jrep-action)[free]),
                production_trial=norm(trial_rhs[free]),extended_trial=norm(fresh_trial[free]),
                extended_unrounded_trial=norm(ideal[free]),
                rounding_residual_effect=norm((fresh_trial-ideal)[free]),
                affine_error=norm(error[free]),extended_affine_error=norm((fresh_trial-fresh+action)[free]),
                unchanged_free_dofs=int(np.count_nonzero(delta[free]==0)),free_dofs=len(free))
            report.append(row)
            np.savez_compressed(here/f'extended_step{step}_i{iteration}_a{row["alpha"]}.npz',
                dofs=free,production=rhs[free],extended=np.asarray(fresh[free],float),
                reaction=np.asarray(rxn[free],float),gradient=np.asarray(grad[free],float),
                positive_scale=np.asarray(scale[free],float))
(here/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
for row in report:
    if row['alpha'] in (0,1,1000000): print(json.dumps(row))
