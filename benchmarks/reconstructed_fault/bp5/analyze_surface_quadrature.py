"""Native-work comparison and independent RHS check for the single initialization."""
import json
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_normal_control import read
from analyze_loading_startup import table
from analyze_loading_tractions import chord, metric
from run_surface_quadrature import OUT, BASE, STUDY, PANELS, CASE


def main():
    path=OUT/CASE;dest=OUT/'comparison';dest.mkdir(exist_ok=True)
    assert json.loads((path/'execution.json').read_text())['passed']
    a,ra,ia=read(BASE);b,rb,ib=read(path)
    assert len(table(path/'accepted_steps.csv'))==1
    for file in BASE.glob('initial_mesh_*.csv'):
        assert file.read_bytes()==(path/file.name).read_bytes()
    for key in ('cell','qp','x','y','source_active','segment','xi','phi','JxW'):
        np.testing.assert_array_equal(ra[key],rb[key])
    for key in ('xd','Theta'):
        np.testing.assert_array_equal(a[key],b[key])
    A=table(BASE/'surface.csv');B=table(path/'surface.csv')
    for key in ('node','x','y'):np.testing.assert_array_equal(A[key],B[key])
    tree=ET.parse(path/'reconstructed_faults/reconstructed_faults-00000.vtu')
    previous=np.fromstring(tree.find('.//PointData/DataArray[@Name="previous_I_h"]').text,sep=' ')
    np.testing.assert_allclose(previous,B['Ih'],rtol=1e-13,atol=0)
    j=rb['segment'].astype(int);xi=rb['xi'];active=(rb['source_active']==1)&(rb['chi']>0)
    ih=(1-xi[active])*B['Ih'][j[active]]+xi[active]*B['Ih'][j[active]+1]
    np.testing.assert_allclose(rb['Ih'][active],ih,rtol=1e-13,atol=0)
    np.testing.assert_allclose(rb['chi'][active]*rb['Ih'][active],ra['chi'][active]*ra['Ih'][active],rtol=1e-13,atol=0)
    # Apply the exact Q1 mass matrix to the production nodal result, not a
    # lumped substitute. Compare with the saved independent volume integral.
    def rhs(values):
        h=np.hypot(np.diff(B['x']),np.diff(B['y']))
        load=np.zeros(len(B));mass=np.zeros(len(B))
        load[:-1]+=h*(values[:-1]/3+values[1:]/6)
        load[1:]+=h*(values[:-1]/6+values[1:]/3)
        mass[:-1]+=h/2;mass[1:]+=h/2
        return load/mass
    exact=table(STUDY/f'surface-quadrature-offline/projection_subdivisions_{PANELS}.csv')
    mask=(a['xd']>=27000)&(a['xd']<=29800)
    np.testing.assert_allclose(a['xd'][mask],exact['xd'],atol=1e-8,rtol=0)
    exact_rhs=exact['accurate_rhs_per_mass']
    projection=dict(baseline_max_relative_RHS_error=float(max(abs(rhs(A['Ih'])[mask]/exact_rhs-1))),
                    corrected_max_relative_RHS_error=float(max(abs(rhs(B['Ih'])[mask]/exact_rhs-1))),
                    corrected_vs_offline_composite=float(max(abs(rhs(B['Ih'])[mask]/exact['composite_rhs_per_mass']-1))),
                    baseline_rhs_relative_peak_to_peak=float(np.ptp(rhs(A['Ih'])[mask])/np.mean(exact_rhs)),
                    corrected_rhs_relative_peak_to_peak=float(np.ptp(rhs(B['Ih'])[mask])/np.mean(exact_rhs)),
                    exact_rhs_relative_peak_to_peak=float(np.ptp(exact_rhs)/np.mean(exact_rhs)))
    completion=np.concatenate([table(p) for p in sorted(path.glob('ih_bottom_completion_rank*.csv'))])
    completion.sort(order='id')
    np.testing.assert_array_equal(completion['id'],np.arange(3*PANELS*(len(B)-1)))
    np.testing.assert_allclose(completion['completed'],completion['inside']+completion['outside'],rtol=1e-14,atol=0)
    init_a=table(BASE/'steady_initialization.csv');init_b=table(path/'steady_initialization.csv')
    report=dict(reference=ia,corrected=ib,projection=projection,
                matched_mesh_phase_geometry_state=True,profile_count=len(completion),
                background_max_change_Pa=float(max(abs(init_b['tau_bg']-init_a['tau_bg']))),windows={},
                execution=json.loads((path/'execution.json').read_text()))
    moments={}
    for label,data,raw in [('baseline',a,ra),('corrected',b,rb)]:
        r=raw[(raw['source_active']==1)&(raw['chi']>0)]
        j=r['segment'].astype(int);xi=r['xi'];d=np.zeros(len(B));mass=np.zeros(len(B))
        for end,shape in ((0,1-xi),(1,xi)):
            mass+=np.bincount(j+end,weights=r['JxW']*r['chi']*shape,minlength=len(B))
            d+=np.bincount(j+end,weights=r['JxW']*r['chi']**2*shape,minlength=len(B))
        check=(a['xd']>=26900)&(a['xd']<=29900)
        np.testing.assert_allclose(mass[check],data['weight'][check],rtol=1e-12,atol=0)
        moments[label]={name:metric(chord(z)[mask],a['weight'][mask]) for name,z in
                        [('mass',mass),('second_moment',d),('second_over_mass',d/data['weight'])]}
    report['native_moment_chords']=moments
    for name,lo,hi in [('primary',27000,29800),('transition',30200,32800),('broad',1000,110000)]:
        region=(a['xd']>=lo)&(a['xd']<=hi);w=a['weight'][region];result={}
        for field in ('q','q_total','V','P','D','mechanical_normal'):
            result[field]=dict(baseline_range=[float(min(a[field][region])),float(max(a[field][region]))],
                corrected_range=[float(min(b[field][region])),float(max(b[field][region]))],
                baseline_mean=float(np.dot(w,a[field][region])/sum(w)),
                corrected_mean=float(np.dot(w,b[field][region])/sum(w)),
                baseline_chord=metric(a['chord_'+field][region],w),
                corrected_chord=metric(b['chord_'+field][region],w),
                difference=metric((b[field]-a[field])[region],w))
        report['windows'][name]=result
        result['background_max_change_Pa']=float(max(abs(init_b['tau_bg'][region]-init_a['tau_bg'][region])))
    for label,data in [('baseline',a),('corrected',b)]:
        np.savetxt(dest/f'{label}.csv',np.column_stack(list(data.values())),delimiter=',',header=','.join(data),comments='')
    np.savetxt(dest/'projection.csv',np.column_stack([a['xd'][mask],rhs(A['Ih'])[mask],rhs(B['Ih'])[mask],exact_rhs]),
        delimiter=',',comments='',header='xd,baseline_RHS_per_mass,corrected_RHS_per_mass,independent_RHS_per_mass')
    for name,lo,hi in [('primary',27,29.8),('transition',29,34),('whole_fault',0,115.5)]:
        fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
        for ax,field,label in zip(axes.flat,('q','V','P','D','mechanical_normal','q_total'),
              ('Native shear perturbation [Pa]','V [m/s]','Pressure [Pa]','-tau:N [Pa]','p-tau:N [Pa]','Native total shear [Pa]')):
            for data,style,title in ((a,'-','original 3-point RHS'),(b,'--',f'{PANELS}-panel RHS')):
                visible=(data['xd']>=lo*1000)&(data['xd']<=hi*1000)
                ax.plot(data['xd'][visible]/1000,data[field][visible],style,lw=1,label=title)
            ax.set_xlim(lo,hi);ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(fontsize=7)
        for ax in axes[-1]:ax.set_xlabel('Down-dip distance [km]')
        fig.tight_layout();fig.savefig(dest/f'{name}.png',dpi=180);plt.close(fig)
    (dest/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(projection=projection,primary=report['windows']['primary'],execution=report['execution']),indent=2))


if __name__=='__main__':main()
