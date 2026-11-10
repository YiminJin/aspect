#!/usr/bin/env python3
"""Integration-based diagnostic comparison; never changes the simulated fields."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np

class CellPolynomials:
    """Native Q2 phase or diagnostic Q5 point-history interpolant on Cartesian cells.

    Q2 uses all 36 Gauss observations and checks its reconstruction. Point
    histories have no Q2 FE representation: Q5 is explicitly an observation
    interpolant through their six-by-six native samples, not a solver projection.
    """
    def __init__(self,points,h,fields,degree):
        self.h=h;self.nx=round(8/h);self.fields=fields;self.degree=degree
        ordered=points[np.lexsort((points.q,points.cell))]
        groups=np.split(ordered,np.flatnonzero(ordered.cell[1:]!=ordered.cell[:-1])+1)
        self.coefficients=np.empty((len(groups),(degree+1)**2,len(fields)))
        self.reconstruction_error=0.
        for p in groups:
            cx,cy=np.mean(p.x),np.mean(p.y)
            x,y=2*(p.x-cx)/h,2*(p.y-cy)/h
            v=np.polynomial.legendre.legvander2d(x,y,[degree,degree]).reshape(-1,(degree+1)**2)
            values=np.stack([p[f] for f in fields],axis=1)
            coefficients=np.linalg.lstsq(v,values,rcond=None)[0]
            index=int(round(cx/h-.5))+self.nx*int(round(cy/h-.5))
            self.coefficients[index]=coefficients
            self.reconstruction_error=max(self.reconstruction_error,float(np.max(abs(v@coefficients-values))))
        if degree==2:assert self.reconstruction_error<1e-11

    def evaluate(self,x,y):
        ix=np.clip(np.floor(x/self.h).astype(int),0,self.nx-1)
        iy=np.clip(np.floor(y/self.h).astype(int),0,round(4/self.h)-1)
        X=2*(x-(ix+.5)*self.h)/self.h;Y=2*(y-(iy+.5)*self.h)/self.h
        v=np.polynomial.legendre.legvander2d(X,Y,[self.degree,self.degree]).reshape(-1,(self.degree+1)**2)
        return np.einsum('ij,ijk->ik',v,self.coefficients[ix+self.nx*iy])

    def gradient(self,x,y):
        """Physical cell-interior derivatives, excluding distributional DG jumps."""
        ix=np.clip(np.floor(x/self.h).astype(int),0,self.nx-1)
        iy=np.clip(np.floor(y/self.h).astype(int),0,round(4/self.h)-1)
        X=2*(x-(ix+.5)*self.h)/self.h;Y=2*(y-(iy+.5)*self.h)/self.h
        vx=np.polynomial.legendre.legvander(X,self.degree)
        vy=np.polynomial.legendre.legvander(Y,self.degree)
        dx=np.zeros_like(vx);dy=np.zeros_like(vy)
        for i in range(1,self.degree+1):
            derivative=np.polynomial.legendre.legder(np.eye(self.degree+1)[i])
            dx[:,i]=np.polynomial.legendre.legval(X,derivative)
            dy[:,i]=np.polynomial.legendre.legval(Y,derivative)
        basis=np.stack([(dx[:,:,None]*vy[:,None,:]).reshape(len(x),-1),
                        (vx[:,:,None]*dy[:,None,:]).reshape(len(x),-1)],axis=1)
        return 2/self.h*np.einsum('ndj,njf->ndf',basis,self.coefficients[ix+self.nx*iy])

def weighted_error(a,b,w):
    diff=a-b;error=np.sqrt(np.sum(w*diff**2));norm=np.sqrt(np.sum(w*b**2))
    return dict(max_abs=float(np.max(abs(diff))),L2=float(error),relative_L2=float(error/norm) if norm else None)

def spatial_comparison(coarse,coarse_initial,fine,fine_initial,hc=.125,hf=.0625):
    cp=CellPolynomials(coarse,hc,['phi'],2);ci=CellPolynomials(coarse_initial,hc,['phi'],2)
    mapped=cp.evaluate(fine.x,fine.y)[:,0];mapped_initial=ci.evaluate(fine.x,fine.y)[:,0]
    result=dict(measure='Fine native Gauss JxW at common physical coordinates; Q2 phase exact within each coarse cell.',
                phi=weighted_error(mapped,fine.phi,fine.weight),
                initial_phi=weighted_error(mapped_initial,fine_initial.phi,fine.weight),
                increment=weighted_error(mapped-mapped_initial,fine.phi-fine_initial.phi,fine.weight),
                phase_reconstruction_defect=cp.reconstruction_error)
    fp=CellPolynomials(fine,hf,['phi'],2);fi=CellPolynomials(fine_initial,hf,['phi'],2)
    grad_initial=ci.gradient(fine.x,fine.y)[:,:,0]
    grad_final=cp.gradient(fine.x,fine.y)[:,:,0]
    fine_grad_initial=fi.gradient(fine.x,fine.y)[:,:,0]
    fine_grad_final=fp.gradient(fine.x,fine.y)[:,:,0]
    for name,a,b in [('initial_gradient',grad_initial,fine_grad_initial),
                     ('phase_gradient',grad_final,fine_grad_final),
                     ('increment_gradient',grad_final-grad_initial,fine_grad_final-fine_grad_initial)]:
        error=np.sqrt(np.sum(fine.weight*np.sum((a-b)**2,axis=1)))
        reference=np.sqrt(np.sum(fine.weight*np.sum(b*b,axis=1)))
        result[name]=dict(max_abs_component=float(np.max(abs(a-b))),L2=float(error),
                          relative_L2=float(error/reference) if reference else None)
    result['gradient_measure']='Exact Q2 cell-interior gradients at fine Gauss points; DG face jumps remain included separately in full SIPG energy.'
    fields=['tauxx','tauyy','tauzz','tauxy','Axx','Ayy','Azz','Axy']
    histories=CellPolynomials(coarse,hc,fields,5);values=histories.evaluate(fine.x,fine.y)
    result['history_interpolation']='Q5 diagnostic polynomial through Q6 native point histories; no Q2 stress-space assumption or state transfer.'
    for j,name in enumerate(fields):result[name]=weighted_error(values[:,j],fine[name],fine.weight)
    for offset,name in [(0,'stress_tensor'),(4,'cohesive_tensor')]:
        diff=np.stack([values[:,j]-fine[fields[j]] for j in range(offset,offset+4)],axis=1)
        truth=np.stack([fine[fields[j]] for j in range(offset,offset+4)],axis=1)
        metric=np.array([1,1,1,2]);error=np.sqrt(sum(fine.weight*np.sum(diff**2*metric,axis=1)))
        norm=np.sqrt(sum(fine.weight*np.sum(truth**2*metric,axis=1)))
        result[name]=dict(L2=float(error),relative_L2=float(error/norm))
    return result,mapped-mapped_initial-(fine.phi-fine_initial.phi)

def trace_plot(cases,out,stem='native-traces'):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for case_index,(result,rows,saved) in enumerate(cases):
        p=saved[max(saved)];seed=saved[0];h=.125/2**result['command']['arguments']['refinement']
        end=CellPolynomials(p,h,['phi'],2);initial=CellPolynomials(seed,h,['phi'],2)
        tag=result['method']+f" h={h:g}, dt={rows.dt[0]:g}"
        for side in [-1,1]:
            # Sample cell interiors, leaving NaN gaps so lines do not smooth DG jumps.
            for axis,extent,center in [(0,8.,2.),(1,4.,4.)]:
                coordinate=np.concatenate([(cell+np.r_[np.linspace(.001,.999,30),np.nan])*h for cell in range(round(extent/h))])
                valid=np.isfinite(coordinate);x=np.full(coordinate.size,center+side*1e-12);y=x.copy()
                if axis==0:x=coordinate;y=np.full(coordinate.size,center+side*1e-12)
                else:y=coordinate;x=np.full(coordinate.size,center+side*1e-12)
                value=np.full(coordinate.size,np.nan);increment=value.copy()
                value[valid]=end.evaluate(x[valid],y[valid])[:,0]
                increment[valid]=value[valid]-initial.evaluate(x[valid],y[valid])[:,0]
                label=tag if side==-1 else None
                axes[0,axis].plot(coordinate,value,'-' if side==-1 else '--',lw=.9,label=label,color=f'C{case_index}')
                axes[1,axis].plot(coordinate,increment,'-' if side==-1 else '--',lw=.9,label=label,color=f'C{case_index}')
    for j in [0,1]:
        for i in [0,1]:
            axes[i,j].set(xlabel='x (m), y=2 traces' if j==0 else 'y (m), x=4 traces',ylabel='phase' if i==0 else 'phase increment',xlim=(1.75,2.5) if j==0 else (1.65,2.35))
            axes[i,j].grid(alpha=.2);axes[i,j].legend(fontsize=6)
    ends=sorted({round(float(rows.time[-1]),9) for _,rows,_ in cases})
    fig.suptitle('Native Q2 traces at '+', '.join(f'{t:g}' for t in ends)+' s: both DG sides; axial panel shows left end')
    for suffix in ['png','pdf']:fig.savefig(out/f'{stem}.{suffix}',dpi=170)
    plt.close(fig)
