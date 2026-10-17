#!/usr/bin/env python3
"""Independent physical Q1 prolongation and saved all-mode DG reconstruction checks."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_pf6_mature import read
from prepare_pf6_mature_resolution import mesh,index,locate,ORIGIN,BASE
from analyze_pf6_mature_resolution import ordered


def q1(points,owners,nodal):
    out=np.empty(len(points))
    for n,(x,owner) in enumerate(zip(points,owners)):
        cell,lo,h=owner;a,b=(x-lo)/h;p=nodal[cell]
        out[n]=(1-a)*(1-b)*p[0]+a*(1-b)*p[1]+(1-a)*b*p[2]+a*b*p[3]
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--initial-only',action='store_true');a=p.parse_args()
    cells=read(a.directory/'inputs-lf/coarse-phase.csv')
    # Parse reference with normal CSV to avoid the analyzer's CellId-only text convention.
    nodal={};idx={}
    for i,h in enumerate(cells['h']):
        lo=np.array([cells['x'][i],cells['y'][i]]);level=int(round(np.log2(BASE/h)))
        cell=str(i);idx[(level,*np.rint((lo-ORIGIN)/h).astype(int))]=(cell,lo,h)
        nodal[cell]=np.array([cells[k][i] for k in ['phi00','phi10','phi01','phi11']])
    # Verify continuity at all reference vertices and edge midpoints, retaining
    # each neighboring cell's physical trace, including hanging interfaces.
    trace={};jump=0.
    for cell,lo,h in idx.values():
        for xi,eta in [(0,0),(1,0),(0,1),(1,1),(.5,0),(.5,1),(0,.5),(1,.5)]:
            x=lo+h*np.array([xi,eta]);v=q1(np.array([x]),[(cell,lo,h)],nodal)[0]
            other=q1(np.array([x]),locate(np.array([x]),idx),nodal)[0]
            jump=max(jump,abs(v-other))
    assert jump<2e-15
    result={'reference_trace_defect':float(jump),'native':{}}
    if not a.initial_only:
        for label in ['dg-cc','dg-fc','particle-cc','particle-fc']:
            run=a.directory/'runs'/('dg-fc-final' if label=='dg-fc' else label)/'run'
            data=ordered(read(run/'common-stress-0.csv'))
            x=np.column_stack([data['x'],data['y']]);expected=q1(x,locate(x,idx),nodal)
            error=float(max(abs(data['phi']-expected)));assert error<2e-14,(label,error)
            coefficients=read(run/'physical-fe-0.csv');mask=coefficients['component']==9
            x=np.column_stack([coefficients['x'][mask],coefficients['y'][mask]])
            vertex_error=float(max(abs(coefficients['accepted'][mask]-q1(x,locate(x,idx),nodal))))
            assert vertex_error<2e-14,(label,vertex_error)
            result['native'][label]=dict(common_point_phase_Linf=error,all_vertex_phase_Linf=vertex_error)
        for backend in ['dg','particle']:
            first=ordered(read(a.directory/'runs'/(backend+'-cc')/'run/common-stress-0.csv'))
            second=ordered(read(a.directory/'runs'/('dg-fc-final' if backend=='dg' else 'particle-fc')/'run/common-stress-0.csv'))
            chi=float(max(abs(first['chi']-second['chi'])));scale=float(max(abs(first['chi'])))
            p1=read(a.directory/'runs'/(backend+'-cc')/'run/coupled-profile-0.csv');p2=read(a.directory/'runs'/('dg-fc-final' if backend=='dg' else 'particle-fc')/'run/coupled-profile-0.csv')
            I=float(max(abs(p1['I_h_m']-p2['I_h_m'])));Irel=I/float(max(abs(p1['I_h_m'])))
            assert chi/scale<2e-9 and Irel<2e-9,(backend,chi/scale,Irel)
            result[backend+'_unchanged_localization']=dict(chi_Linf_per_m=chi,chi_relative_Linf=chi/scale,I_Linf_m=I,I_relative_Linf=Irel)
        for label in ['dg-cc','dg-fc','dg-ff']:
            run=a.directory/'runs'/('dg-fc-final' if label=='dg-fc' else label)/'run';nodes=read(run/'dg-support.csv');grid=mesh(run);bounds={c:(lo,h) for c,lev,lo,h in grid}
            d=ordered(read(run/'common-stress-3.csv'));coef=read(run/'dg-coefficients-3.csv');coeff={}
            for cell,m,xx,yy,xy in zip(coef['cell'],coef['mode'],coef['xx'],coef['yy'],coef['xy']):
                coeff.setdefault(cell,np.zeros((9,3)))[int(m)]=[xx,yy,xy]
            total=np.zeros((len(d['x']),3))
            def basis(x):return np.column_stack([2*(x-.5)*(x-1),4*x*(1-x),2*x*(x-.5)])
            groups={}
            for j,cell in enumerate(d['cell']):groups.setdefault(cell,[]).append(j)
            for cell,indices in groups.items():
                ids=np.array(indices);lo,h=bounds[cell]
                x=(d['x'][ids]-lo[0])/h;y=(d['y'][ids]-lo[1])/h
                bx,by=basis(x),basis(y)
                B=np.array([bx[:,int(round(2*xi))]*by[:,int(round(2*eta))] for xi,eta in zip(nodes['xi'],nodes['eta'])]).T
                total[ids]=B@coeff[cell]
            given=np.column_stack([d[k] for k in ['T_xx','T_yy','T_xy']])
            defect=float(np.max(np.abs(total-given)));scale=float(np.max(np.abs(given)))
            assert defect/scale<2e-12,(label,defect/scale)
            S=np.array([-np.sqrt(3)/4,np.sqrt(3)/4,.25])
            tau=total-2*(d['chi']*d['Z'])[:,None]*S
            expected=np.column_stack([d[k] for k in ['tau_xx','tau_yy','tau_xy']])
            tau_defect=float(np.max(abs(tau-expected)))
            result['native'].setdefault(label,{}).update(
                saved_DG_coefficient_defect_Pa=defect,
                saved_DG_coefficient_scaled_defect=defect/scale,
                reconstructed_tau_component_defect_Pa=tau_defect,modes=9)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
