#!/usr/bin/env python3
"""Reduce common-point mature resolution controls. Backend differences are neutral."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_pf6_mature import read,norm3,cut_jumps
ROOT=Path(__file__).resolve().parents[3]
L=1000/np.sin(np.pi/3);UPPER=np.array([-523.1751345948129,1137.25])
TANGENT=np.array([.5,-np.sqrt(3)/2]);NORMAL=np.array([np.sqrt(3)/2,.5])
VP=1e-9;THETA0=.008/VP;SIGMA0=50e6;TAU0=26546122.3650655
LABELS=['dg-cc','dg-fc','dg-ff','particle-cc','particle-fc','particle-ff']


def ordered(data):
    order=np.argsort(data['qp']);assert np.array_equal(data['qp'][order],np.arange(len(order)))
    return {k:v[order] for k,v in data.items()}


def partitions(s):
    end=(s<125)|(s>L-125)
    return {'all':np.ones(len(s),dtype=bool),'interior':~end,'endpoints':end}


def tensors(a,b,h):
    for k in ['qp','x','y','JxW']: assert np.array_equal(a[k],b[k]),k
    s=(a['x']-UPPER[0])*TANGENT[0]+(a['y']-UPPER[1])*TANGENT[1]
    dn=norm3({k:a[k]-b[k] for k in ['tau_xx','tau_yy','tau_zz','tau_xy']})
    signal=(norm3(a)+norm3(b))/2
    result={}
    for region,mask in partitions(s).items():
        metrics={}
        for name,weight in [('whole',a['JxW']),('localized',a['JxW']*h)]:
            weight=weight*mask
            n=float(np.sum(weight*dn));d=float(np.sum(weight*signal))
            metrics[name+'_RMS_Pa']=float(np.sqrt(n/np.sum(weight)))
            metrics[name+'_symmetric_relative_L2']=float(np.sqrt(n/d)) if d else None
            metrics[name+'_first_RMS_Pa']=float(np.sqrt(np.sum(weight*norm3(a))/np.sum(weight)))
            metrics[name+'_second_RMS_Pa']=float(np.sqrt(np.sum(weight*norm3(b))/np.sum(weight)))
            cauchy_difference=dn+3*(a['p']-b['p'])**2
            cauchy_signal=signal+1.5*(a['p']**2+b['p']**2)
            metrics[name+'_Cauchy_RMS_Pa']=float(np.sqrt(np.sum(weight*cauchy_difference)/np.sum(weight)))
            cauchy_denominator=float(np.sum(weight*cauchy_signal))
            metrics[name+'_Cauchy_symmetric_relative_L2']=float(np.sqrt(np.sum(weight*cauchy_difference)/cauchy_denominator)) if cauchy_denominator else None
        metrics['full_tensor_max_Pa']=float(np.sqrt(np.max(dn[mask])))
        metrics['pressure_max_Pa']=float(np.max(np.abs(a['p'][mask]-b['p'][mask])))
        dx,dy,dxy=(a[k]-b[k] for k in ['tau_xx','tau_yy','tau_xy'])
        raw_normal=a['p']-b['p']-.75*dx-.25*dy-np.sqrt(3)/2*dxy
        raw_shear=-np.sqrt(3)/4*dx+np.sqrt(3)/4*dy+.5*dxy
        metrics['unfiltered_point_normal_max_Pa']=float(np.max(abs(raw_normal[mask])))
        metrics['unfiltered_point_shear_max_Pa']=float(np.max(abs(raw_shear[mask])))
        metrics['phi_Linf']=float(np.max(np.abs(a['phi'][mask]-b['phi'][mask])))
        metrics['chi_Linf_per_m']=float(np.max(np.abs(a['chi'][mask]-b['chi'][mask])))
        result[region]=metrics
    return result


def surface(pa,pb,wa,wb):
    assert np.array_equal(pa['s_m'],pb['s_m'])
    for k in ['node','s_m']: assert np.array_equal(wa[k],wb[k]),k
    # Native bulk-work mass depends on the quadrature partition and chi;
    # a fixed surface grid alone does not make this mass identical.
    masses=[v['mass_diagonal']+v['mass_edge']+np.r_[0,v['mass_edge'][:-1]] for v in [wa,wb]]
    backgrounds={'shear':TAU0,'raw_normal':SIGMA0,'filtered_normal':SIGMA0}
    length=np.diff(pa['s_m']);edge=np.r_[length/6,0.]
    diagonal=np.r_[length/3,0.]+np.r_[0.,length/3]
    def geometric_load(v):return diagonal*v+edge*np.r_[v[1:],0.]+np.r_[0.,edge[:-1]*v[:-1]]
    keys={'shear':'shear_Pa','raw_normal':'raw_normal_Pa','filtered_normal':'filtered_normal_Pa'}
    result={}
    for region,mask in partitions(pa['s_m']).items():
        metrics={key:float(np.max(np.abs(pa[key][mask]-pb[key][mask])))
                 for key in ['V_m_s','theta_s','slip_m','shear_Pa','raw_normal_Pa','filtered_normal_Pa','I_h_m']}
        metrics['V_difference_over_Vp']=metrics['V_m_s']/VP
        metrics['theta_difference_over_theta0']=metrics['theta_s']/THETA0
        metrics['slip_difference_over_Dc']=metrics['slip_m']/.008
        metrics['native_mass_diagonal_difference_m']=float(max(abs(wa['mass_diagonal'][mask]-wb['mass_diagonal'][mask])))
        metrics['native_mass_one_relative_difference']=float(max(abs(masses[0][mask]-masses[1][mask]))/max(abs(masses[0][mask])))
        for key,background in backgrounds.items():
            first=wa[key]-background*masses[0];second=wb[key]-background*masses[1]
            difference=first-second
            amplitude=max(float(np.max(np.abs(first[mask]))),float(np.max(np.abs(second[mask]))))
            metrics[key+'_native_total_weak_difference_Pa_m']=float(max(abs((wa[key]-wb[key])[mask])))
            metrics[key+'_background_measure_difference_Pa_m']=float(max(abs(background*(masses[0]-masses[1])[mask])))
            first_common=geometric_load(pa[keys[key]]-background);second_common=geometric_load(pb[keys[key]]-background)
            common_delta=float(max(abs((first_common-second_common)[mask])))
            common_scale=max(float(max(abs(first_common[mask]))),float(max(abs(second_common[mask]))))
            metrics[key+'_common_geometric_weak_difference_Pa_m']=common_delta
            metrics[key+'_common_geometric_relative_perturbation']=common_delta/common_scale if common_scale else None
            metrics[key+'_weak_max_Pa_m']=float(np.max(np.abs(difference[mask])))
            metrics[key+'_weak_perturbation_scale_Pa_m']=amplitude
            metrics[key+'_weak_relative_perturbation']=metrics[key+'_weak_max_Pa_m']/amplitude if amplitude else None
            metrics[key+'_weak_RMS_Pa_m']=float(np.sqrt(np.mean(difference[mask]**2)))
        result[region]=metrics
    return result


def main():
    global TAU0
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    # Exact background from the retained native initializer's csv.
    # Official tau0 is calculated with the regularized law, not its log approximation.
    TAU0=SIGMA0*.025*np.arcsinh(VP/(2*1e-6)*np.exp((.6+.015*np.log(1e-6/VP))/.025))+4624440*VP
    physical=read(a.directory.parent/'inputs-lf/physical-profile.csv')
    result={'definition':'Neutral pairwise differences, not accuracy errors against DG or particles',
            'endpoint_window_m':125.,'tau0_Pa':float(TAU0),'runs':{},'comparisons':{},'initialization':{}}
    data={}
    for label in LABELS:
        folder=a.directory/('dg-fc-final' if label=='dg-fc' else label)/'run';record=json.loads(folder.parent.joinpath('record.json').read_text());assert record['exit_code']==0,label
        checks=read(folder/'coupled-checks.csv');accepted=read(folder/'accepted_steps.csv')
        assert np.all(checks['mechanical_qps']==36) and np.all(checks['publications']==1)
        entry={k:record[k] for k in ['seconds','max_rss_KiB','artifacts','arguments']}
        entry.update(max_native_publication_defect_Pa=float(max(checks['stress_error_Pa'])),
                     max_native_allowance_fraction=float(max(checks['error_fraction_of_allowance'])),
                     max_Z_recurrence_defect_Pa_m=float(max(checks['Z_error_Pa_m'])),
                     max_nonlinear_residual=float(max(accepted['normalized_nonlinear_residual'])),
                     max_surface_RMS_Pa=float(max(accepted['surface_RMS_Pa'])),
                     max_Theta_aging_relative_defect=float(max(accepted['Theta_relative_error'])),
                     particles=int(checks['particles'][0]),all_publications_single=True)
        assert int(checks['step'][-1])==3 and checks['time_s'][-1]==600000
        data[label]={n:ordered(read(folder/f'common-stress-{n}.csv')) for n in range(4)}
        entry['jumps']=cut_jumps(read(folder/'bulk-normal-cuts-3.csv'))
        result['runs'][label]=entry
        initial=data[label][0]
        r=abs((initial['x']-UPPER[0])*NORMAL[0]+(initial['y']-UPPER[1])*NORMAL[1])
        phi=np.interp(r,physical['r_m'],physical['phi'],right=0.)
        weights=initial['JxW']
        result['initialization'][label]=dict(phi_Linf=float(max(abs(initial['phi']-phi))),
            phi_relative_L2=float(np.sqrt(sum(weights*(initial['phi']-phi)**2)/sum(weights*phi**2))),
            committed_tau_max_Pa=float(np.sqrt(max(norm3(initial)))))
    base=data['dg-cc'][0];r=abs((base['x']-UPPER[0])*NORMAL[0]+(base['y']-UPPER[1])*NORMAL[1])
    h=np.interp(r,physical['r_m'],physical['h'],right=0.)
    s=(base['x']-UPPER[0])*TANGENT[0]+(base['y']-UPPER[1])*TANGENT[1]
    result['common_points']=len(s)
    result['bulk_region_points']={k:int(sum(v)) for k,v in partitions(s).items()}
    result['surface_region_nodes']={k:int(sum(v)) for k,v in partitions(read(a.directory/'dg-cc/run/coupled-profile-0.csv')['s_m']).items()}
    for label in LABELS:
        for step in range(1,4):
            for field in ['phi','chi']:
                assert np.array_equal(data[label][0][field],data[label][step][field]),(label,step,field)
        if label.startswith('dg'):
            d=data[label][3];T={k:d[v] for k,v in zip(['tau_xx','tau_yy','tau_xy'],['T_xx','T_yy','T_xy'])}
            T['tau_zz']=-T['tau_xx']-T['tau_yy']
            correction={k:T[k]-d[k] for k in T}
            weight=d['JxW']*h
            rms=lambda v:float(np.sqrt(sum(weight*norm3(v))/sum(weight)))
            result['runs'][label]['localized_cancellation']=dict(T_RMS_Pa=rms(T),
                localization_RMS_Pa=rms(correction),tau_RMS_Pa=rms(d),T_over_tau=rms(T)/rms(d))
    result['decomposition']={}
    for backend in ['dg','particle']:
        result['decomposition'][backend]={}
        for step in range(1,4):
            cc,fc,ff=(data[backend+'-'+c][step] for c in ['cc','fc','ff'])
            keys=['tau_xx','tau_yy','tau_zz','tau_xy']
            mechanical={k:fc[k]-cc[k] for k in keys};profile={k:ff[k]-fc[k] for k in keys}
            dot=lambda a,b:a['tau_xx']*b['tau_xx']+a['tau_yy']*b['tau_yy']+a['tau_zz']*b['tau_zz']+2*a['tau_xy']*b['tau_xy']
            weight=cc['JxW']*h
            nm=float(sum(weight*norm3(mechanical)));np_=float(sum(weight*norm3(profile)))
            cross=float(sum(weight*dot(mechanical,profile)))
            result['decomposition'][backend][str(step)]=dict(
                localized_mechanics_RMS_Pa=float(np.sqrt(nm/sum(weight))),
                localized_profile_RMS_Pa=float(np.sqrt(np_/sum(weight))),
                effect_cosine=cross/np.sqrt(nm*np_),
                localized_total_RMS_Pa=float(np.sqrt((nm+np_+2*cross)/sum(weight))))
    pairs=[('dg-cc','dg-fc'),('dg-fc','dg-ff'),('dg-cc','dg-ff'),
           ('particle-cc','particle-fc'),('particle-fc','particle-ff'),('particle-cc','particle-ff')]
    pairs += [('dg-'+kind,'particle-'+kind) for kind in ['cc','fc','ff']]
    for first,second in pairs:
        result['comparisons'][first+'__'+second]={}
        for step in range(4):
            def load(label,prefix):return read(a.directory/('dg-fc-final' if label=='dg-fc' else label)/'run'/f'{prefix}-{step}.csv')
            entry={'surface':surface(load(first,'coupled-profile'),load(second,'coupled-profile'),load(first,'surface-weak'),load(second,'surface-weak'))}
            entry['tensor']=tensors(data[first][step],data[second][step],h)
            result['comparisons'][first+'__'+second][str(step)]=entry
    result['native_seconds']=sum(v['seconds'] for v in result['runs'].values())
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['definition','native_seconds']},indent=2))
if __name__=='__main__': main()
