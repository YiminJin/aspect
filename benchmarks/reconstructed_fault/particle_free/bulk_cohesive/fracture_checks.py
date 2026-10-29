#!/usr/bin/env python3
"""Unit/scales audit and independent checks of accepted native fields/work."""
import argparse,json
import numpy as np
from scipy.integrate import cumulative_trapezoid
from fracture_analysis import ROOT,OUT,BUILD,NEW,OLD,read,save_csv
from fracture_strip import SETS,compliance,seed_table,G,ETA,R

def scales():
    rows=[];results={}
    for name,p in SETS.items():
        ell=p['ell'];Gc=p['Gc'];drop=p['drop'];Kf=Gc/((8/3)*ell);m=2*G*Kf/drop**2
        r,phi=seed_table(ell);tip=.5 if name=='retained' else 1.
        th=np.linspace(0,np.pi/2,262145);p0=.6*np.cos(th)**2
        v=2*ell*(1-p0)/np.sqrt(1.5-p0);rr=np.r_[0.,cumulative_trapezoid(v,th)]
        h=compliance(p0,m)[0];hh=compliance(np.array(.6),m)[0];f=p0-.6*h/hh
        Fband=2*Kf*np.trapezoid((p0+f)*v,th)
        Fcap=np.pi*Kf*np.trapezoid(rr*(p0+f)*v,th)
        I=2*np.trapezoid(h*v,th)
        Lc=G*Gc/drop**2;estimate=np.sqrt(4*G*Gc/(np.pi*tip))
        rows.append([name,G,ETA,ETA/G,Gc,drop,R,R+drop,ell,Kf,m,2*m*(m-3),Lc,Gc/drop,r[-1],2*r[-1],1.5,1.,tip,1.5-tip,I,Fband,Fcap,tip*Fband+Fcap,estimate])
        results[name]=dict(full_band_AT1_Gc_J_m2=Gc,seed_band_fracture_J_m2=float(Fband),seed_cap_fracture_J_m=float(Fcap),
            continuum_seed_total_J_m=float(tip*Fband+Fcap),seed_I_m=float(I),cohesive_length_m=Lc,
            homogeneous_drop_Pa=drop,sharp_crack_F1_estimated_drop_Pa=float(estimate),
            estimate_to_virgin_drop=float(estimate/drop),time_relaxation_s=ETA/G,
            g_second_at_zero=2*m*(m-3),full_band_width_m=4*ell)
    save_csv(OUT/'fracture-scales.csv',['case','G_Pa','eta_Pa_s','Maxwell_time_s','Gc_J_m2','peak_minus_residual_Pa','residual_Pa','peak_Pa','ell_m','Kf_Pa','m','g_second_at_zero','cohesive_length_m','slip_scale_m','seed_radius_m','seed_support_width_m','domain_L_m','domain_H_m','seed_length_m','ligament_m','seed_normalization_I_m','seed_band_fracture_J_m2','seed_cap_fracture_J_m','seed_total_continuum_fracture_J_m','plane_strain_F1_estimated_excess_Pa'],rows)
    (OUT/'scales-audit.json').write_text(json.dumps(results,indent=2)+'\n')

def dense(path,steps,m,ell,Gc):
    ledger=read(path/'bulk-accepted.csv');Kf=Gc/((8/3)*ell);result={}
    for n in steps:
        file=path/f'bulk-{n:04d}.csv'
        if not file.exists():continue
        # Keep authoritative tensor fields and histories at their original QPs.
        a=np.loadtxt(file,delimiter=',',skiprows=1,usecols=range(1,18))
        x,y,w,phi,Txx,Tyy,Txy,txx,tyy,txy,aa,q,k,L,Y,Eb,Ec=a.T
        h,hp,_=compliance(phi,m);norm=txx*txx+tyy*tyy+(txx+tyy)**2+2*txy*txy
        error=dict(reconstructed_stress_scaled=float(max(abs(Txx-txx).max(),abs(Tyy-tyy).max(),abs(Txy-G*aa-txy).max())/1000),
            cohesive_relation_scaled=float(abs(G*aa-h*q).max()/1000),
            driving_scaled=float(abs(Y-hp*q*q/(2*G)).max()/Kf),
            bulk_density_error_Pa=float(abs(Eb-norm/(4*G)).max()),
            cohesive_density_error_Pa=float(abs(Ec-h*q*q/(2*G)).max()),
            friction_bound_scaled=float(np.maximum(abs(txy-q)-R,0).max()/1000),
            minimum_cumulative_absolute_shear=float(L.min()),
            maximum_physical_tau_xy_Pa=float(txy.max()),maximum_driving_Pa=float(Y.max()),
            initially_virgin_max_physical_tau_xy_Pa=float(txy[phi==0].max(initial=0)))
        assert max(error[k] for k in ['reconstructed_stress_scaled','cohesive_relation_scaled','driving_scaled','friction_bound_scaled'])<1e-12,error
        assert max(error['bulk_density_error_Pa'],error['cohesive_density_error_Pa'])<1e-10
        if n:
            row=ledger[ledger['step']==n][0]
            error['bulk_storage_error_J_m']=float(abs(np.sum(w*Eb)-row['bulk']))
            error['cohesive_storage_error_J_m']=float(abs(np.sum(w*Ec)-row['cohesive']))
            error['cumulative_friction_error_J_m']=float(abs(R*np.sum(w*L)-sum(ledger['Df'][ledger['step']<=n])))
            assert max(error[k] for k in ['bulk_storage_error_J_m','cohesive_storage_error_J_m','cumulative_friction_error_J_m'])<1e-9
        assert abs(w.sum()-1.5)<1e-12 and np.all(w>0)
        result[str(n)]=error
    return result

def main(candidate):
    scales()
    if not candidate:return
    summary=json.loads((OUT/'candidate-summary.json').read_text());last=summary['steps']
    steps=sorted(set([0,*[k for k in [96,120,121,124,128,139,140,148] if k<=last]]))
    fieldchecks=dense(NEW,steps,4.,.046875,1.)
    # Native Q1 seed vs independent analytic profile table: actual tabulation,
    # support and nodal sampling are checked independently of later growth.
    table=read(NEW/'seed-prescription.csv');r,p=seed_table(.046875)
    table_radius_error=float(abs(r-table['radius_m']).max());table_phi_error=float(abs(p-table['phi']).max())
    assert table_radius_error<1e-9 and table_phi_error<1e-14
    nodes=np.loadtxt(NEW/'phase-0000.csv',delimiter=',',skiprows=1)
    exact=np.interp(np.hypot(np.maximum(nodes[:,1]-1.,0),nodes[:,2]-.5),r,p,right=0)
    node_error=float(abs(nodes[:,3]-exact).max());assert node_error<1e-9
    entry=json.loads((OUT/'entry.json').read_text());protected=entry['protected_bp3']
    import hashlib
    def digest(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    assert all(digest(ROOT/n)==v['sha256'] for n,v in protected.items())
    unchanged={n:checksum for n,checksum in entry['files'].items() if n.startswith(('source/','build-particle-free/bulk-parallel/'))}
    assert all(digest(ROOT/n)==v for n,v in unchanged.items())
    native_log=(ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence/bulk-fracture-m4.log').read_text(errors='replace')
    dof_lines=[line.strip() for line in native_log.splitlines() if 'degrees of freedom' in line or 'active cells' in line]
    result=dict(dense=fieldchecks,seed_table_radius_error_m=table_radius_error,seed_table_phi_error=table_phi_error,
        initial_nodal_prescription_error=node_error,protected_BP3_files_unchanged=len(protected),
        native_equations_executable_and_baseline_plugin_unchanged=list(unchanged),native_mesh_dof_lines=dof_lines)
    (OUT/'native-checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',action='store_true');a=p.parse_args();main(a.candidate)
