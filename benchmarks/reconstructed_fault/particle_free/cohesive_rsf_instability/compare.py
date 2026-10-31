#!/usr/bin/env python3
"""Compact human-review comparisons of accepted native ASPECT exports (SI).

No evolution, state conversion, or interpolation into a different constitutive
state. Temporal curves use linear interpolation of diagnostic functionals only.
Spatial profiles use identical Gauss points in these fixed-geometry controls.
"""
import argparse, csv, json, os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cohesive-rsf-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import read


def value(h, key):
    return np.array([r.get(key, 0.) for r in h])


def initial(x):
    return np.minimum(1, np.minimum(x-4, np.maximum(0, 9-x))).clip(0, 1)


def field_at(time, hh, ss, key):
    """Common-time diagnostic interpolation, never accepted-history transfer."""
    tx=value(hh,'time_s');j=min(len(tx)-1,int(np.searchsorted(tx,time)))
    if j==0:return ss[0][key]
    lam=(time-tx[j-1])/(tx[j]-tx[j-1]);return (1-lam)*ss[j-1][key]+lam*ss[j][key]


def save_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def bulk_at(time, hh, stems):
    """Interpolate exported diagnostics only; never transfer accepted histories."""
    tx=value(hh,'time_s');j=min(len(tx)-1,int(np.searchsorted(tx,time)))
    def table(i):
        files=sorted(stems[i].parent.glob(stems[i].name+'-bulk-r*.csv'))
        data=np.concatenate([np.genfromtxt(f,delimiter=',',names=True,dtype=None,encoding='utf8') for f in files])
        return np.sort(data,order='cell')
    b=table(j);keys=['Txx_Pa','Tyy_Pa','Tzz_Pa','Txy_Pa','pprime_Pa']
    vals=np.column_stack([b[k] for k in keys])
    if j and time < tx[j]:
        a=table(j-1);assert np.array_equal(a['cell'],b['cell'])
        lam=(time-tx[j-1])/(tx[j]-tx[j-1]);vals=(1-lam)*np.column_stack([a[k] for k in keys])+lam*vals
    return b,vals


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--primary', type=Path, required=True)
    p.add_argument('--neutral', type=Path, required=True)
    p.add_argument('--half', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    runs = {'State weakening (b=.02)': read(args.primary),
            'Matched neutral (b=a=.005)': read(args.neutral)}
    if args.half: runs['Half increment limits'] = read(args.half)
    common_end=min(v[0][-1]['time_s'] for v in runs.values())
    h, s, _ = next(iter(runs.values())); w0 = s[0]['weight_m']; a0 = s[0]['damage']
    t = value(h, 'time_s'); event = int(np.searchsorted(t, 374.56704168613635))
    interior = (a0 > 0) & (a0 < 1)
    bond0 = w0[interior] @ ((1-a0[interior]) *
            (s[0]['jump_m'][interior]-s[0]['reference_m'][interior])**2 /
            (2 * 1.5e-6 * a0[interior]))
    report = {'initial_storage_J_m': dict(bulk=h[0]['bulk_J_m'],
            cohesive=float(bond0), pre_existing_fracture=float(3*w0@a0))}
    histories = []
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.4))
    for label, (hh, ss, _) in runs.items():
        x = ss[-1]['x_m']; w = ss[-1]['weight_m']; a0 = initial(x)
        tt = value(hh, 'time_s'); speed = value(hh, 'maximum_speed_m_s')
        ax[0, 0].semilogy(tt, np.maximum(speed, 1e-12), label=label)
        ax[0, 1].plot(tt, value(hh, 'mean_mature_theta'), label=label)
        ax[0, 2].plot(tt, value(hh, 'mean_mature_slip_m')/6e-5, label=label)
        mask = tt >= t[event]
        ax[1, 0].plot(tt[mask]-t[event], value(hh, 'growth_m')[mask], label=label)
        ax[1, 1].plot(tt[mask]-t[event], speed[mask], label=label)
        ax[1, 2].plot(x, field_at(common_end,hh,ss,'damage')-a0, label=label)
        growth = value(hh, 'new_intact_damage_m')
        active = np.flatnonzero(growth > 1e-8)
        onset = [float(tt[active[0]-1]), float(tt[active[0]])] if len(active) else None
        delta = ss[-1]['damage']-a0
        for row in hh:
            assert int(row['publications']) == row['step'], ('publication', label, row['step'])
            if 'ledger_error_J_m' in row:
                assert abs(row['ledger_error_J_m']) < 1e-10*max(1.,abs(row['endpoint_work_J_m'])), ('ledger',label,row['step'])
                assert row['mechanical_weak_Pa'] <= 1e-8, ('mechanics',label,row['step'])
                assert row['weak_error_Pa'] <= 1e-8, ('surface',label,row['step'])
        report[label] = dict(final_time_s=float(tt[-1]), max_speed_m_s=float(max(speed)),
                final_speed_m_s=float(speed[-1]), mature_slip_Dc=float(value(hh,'mean_mature_slip_m')[-1]/6e-5),
                integrated_added_damage_m=float(w@delta), peak_damage_increment=float(max(delta)),
                new_intact_damage_m=float(growth[-1]),
                new_intact_integral_above_1e_8_m_bracket_s=onset,
                fully_decohered_length_m=float(value(hh,'fully_decohered_length_m')[-1]),
                publication_counts_match_steps=True,
                maximum_mechanical_weak_Pa=float(max(value(hh,'mechanical_weak_Pa'))),
                maximum_surface_weak_Pa=float(max(value(hh,'weak_error_Pa'))),
                maximum_BE_ledger_J_m=float(max(abs(value(hh,'ledger_error_J_m')))),
                maximum_source_scaled=float(max(value(hh,'source_scaled'))),
                outer_rejections=int(value(hh,'clock_outer_rejections')[-1]))
        for r in hh:
            histories.append(dict(case=label, **{k:r.get(k,0.) for k in ['step','time_s','dt_s','maximum_speed_m_s',
                'mean_mature_theta','mean_mature_slip_m','growth_m','new_intact_damage_m','fully_decohered_length_m',
                'bulk_J_m','bond_J_m','bulk_increment_J_m','bond_increment_J_m',
                'trapezoid_work_J_m','friction_J_m','radiation_J_m','maxwell_J_m',
                'BE_bulk_J_m','surface_remainder_J_m',
                'fracture_increment_J_m','physical_remainder_J_m','ledger_error_J_m']}))
    labels = [('Native maximum slip rate','Time (s)','|V|max (m/s)'),
              ('State weakening','Time (s)','Normalized age θ'),
              ('Sliding on initial mature patch','Time (s)','Mean signed slip / Dc'),
              ('Focused replay: added damage',f'Time − {t[event]:.8f} s','∫ (a−a₀) ds (m)'),
              ('Focused replay: slip rate',f'Time − {t[event]:.8f} s','|V|max (m/s)'),
              (f'Common time {common_end:.8f} s','x (m)','a−a₀')]
    for aa, (title, xx, yy) in zip(ax.flat, labels):
        aa.set(title=title, xlabel=xx, ylabel=yy); aa.grid(alpha=.2); aa.legend(fontsize=7)
    ax[1,2].set_xlim(7.5,9.5); ax[1,2].axvline(9,color='k',ls=':',lw=.8)
    fig.suptitle('Constant slow boundary loading; quasi-dynamic mechanics and radiation damping')
    fig.tight_layout(); fig.savefig(args.output/'comparison.png', dpi=160); plt.close(fig)
    save_csv(args.output/'indexed-histories.csv', histories)
    profiles=[]
    for label,(hh,ss,_) in runs.items():
        x = ss[-1]['x_m']; w = ss[-1]['weight_m']; a0 = initial(x)
        for q in range(len(x)):
            profiles.append(dict(case=label,time_s=common_end,x_m=float(x[q]),
                weight_m=float(w[q]),initial_damage=float(a0[q]),
                **{k:float(field_at(common_end,hh,ss,k)[q]) for k in ['damage','jump_m','velocity_m_s','theta_normalized','q_Pa','pn_work_Pa','resistance_Pa']}))
    save_csv(args.output/'final-profiles.csv', profiles)
    if args.half:
        fh, fs, f_stems = list(runs.values())[-1]; ft=value(fh,'time_s')
        end=min(t[-1],ft[-1])
        times=np.unique(np.r_[np.linspace(t[event],end,81),
            t[(t>=t[event])&(t<=end)],ft[(ft>=t[event])&(ft<=end)]])
        rows=[]
        for time in times:
            row=dict(time_s=float(time))
            for key in ['growth_m','maximum_speed_m_s','mean_mature_slip_m','mean_mature_theta','new_intact_damage_m']:
                v=float(np.interp(time,t,value(h,key))); f=float(np.interp(time,ft,value(fh,key)))
                row.update({key+'_primary':v,key+'_half':f,key+'_absolute_difference':abs(f-v)})
            rows.append(row)
        save_csv(args.output/'temporal-common-times.csv',rows)
        summary={}
        for key in ['growth_m','maximum_speed_m_s','mean_mature_slip_m','mean_mature_theta','new_intact_damage_m']:
            v=rows[-1][key+'_primary'];f=rows[-1][key+'_half']
            # Growth uncertainty is normalized by the growth, not initial damage.
            diffs=np.array([r[key+'_absolute_difference'] for r in rows]);ref=np.array([abs(r[key+'_half']) for r in rows])
            summary[key]=dict(primary=v,half=f,absolute_difference=abs(f-v),relative_to_half=abs(f-v)/abs(f) if f else None,
                maximum_absolute_difference=float(max(diffs)),maximum_abs_over_peak_half=float(max(diffs)/max(ref)) if max(ref)>0 else None)
            if key=='mean_mature_theta':
                summary[key]['maximum_relative_difference']=float(max(diffs/ref))
        # Match physical end by interpolation of diagnostic fields, never evolve it.
        # The temporary virgin reserve may differ from the initial point set.
        # The focused replay retains identical physical Gauss points.
        assert np.allclose(s[-1]['x_m'],fs[-1]['x_m'],rtol=0,atol=1e-10)
        w=s[-1]['weight_m'];a0=initial(s[-1]['x_m'])
        dd=field_at(end,h,s,'damage')-a0;ff=field_at(end,fh,fs,'damage')-a0
        summary['damage_growth_L2']=dict(absolute=float(np.sqrt(w@((dd-ff)**2))),
            relative=float(np.sqrt(w@((dd-ff)**2)/(w@(ff**2)))) if w@(ff**2)>0 else None)
        for key in ['jump_m','theta_normalized','pn_work_Pa','resistance_Pa']:
            av=field_at(end,h,s,key);bv=field_at(end,fh,fs,key);dv=av-bv
            summary[key+'_field']=dict(maximum_absolute=float(max(abs(dv))),
                RMS_absolute=float(np.sqrt(w@(dv*dv)/sum(w))),
                L2_relative=float(np.sqrt(w@(dv*dv)/(w@(bv*bv)))) if w@(bv*bv)>0 else None)
            mature=(s[-1]['x_m']>=5)&(s[-1]['x_m']<=8)
            summary[key+'_field']['mature_patch_L2_relative']=float(np.sqrt(w[mature]@(dv[mature]**2)/(w[mature]@(bv[mature]**2)))) if w[mature]@(bv[mature]**2)>0 else None
        # Same cell-centre physical tensor diagnostic, full plane-strain norm.
        # This measures temporal differences, not spatial point-stress accuracy.
        ph,_,p_stems=next(iter(runs.values()))
        cells,bv=bulk_at(end,fh,f_stems);pc,pv=bulk_at(end,ph,p_stems)
        ic,iv=bulk_at(t[event],ph,p_stems)
        assert np.array_equal(cells['cell'],pc['cell']) and np.array_equal(cells['cell'],ic['cell'])
        area=cells['area_m2'];metric=np.array([1,1,1,2])
        norm=lambda v:float(np.sqrt(area@((v[:,:4]**2)@metric)))
        absolute=norm(pv-bv)
        summary['physical_tensor_cell_centres']=dict(L2_absolute_Pa_m=absolute,
            relative_to_final_stress=absolute/norm(bv),relative_to_replay_stress_change=absolute/norm(bv-iv),
            pressure_RMS_difference_Pa=float(np.sqrt(area@((pv[:,4]-bv[:,4])**2)/sum(area))),
            scope='Temporal diagnostic at common cells; no spatial point-stress qualification')
        ep=[]
        for center in np.linspace(4.5,9.5,41):
            test=np.exp(-.5*((s[-1]['x_m']-center)/.375)**2);measure=w@test
            pa=field_at(end,h,s,'resistance_Pa');pb=field_at(end,fh,fs,'resistance_Pa')
            ep.append(dict(kernel_centre_x_m=float(center),primary_Pa=float(w@(test*pa)/measure),half_Pa=float(w@(test*pb)/measure)))
        save_csv(args.output/'temporal-averaged-loads.csv',ep)
        summary['averaged_resistance_max_difference_Pa']=max(abs(r['primary_Pa']-r['half_Pa']) for r in ep)
        summary['temporal_energy_J_m']={}
        for key in ['physical_remainder_J_m','fracture_increment_J_m','friction_J_m','radiation_J_m','bulk_increment_J_m','trapezoid_work_J_m']:
            def cumulative_at(hh,time):return float(np.interp(time,value(hh,'time_s'),np.cumsum(value(hh,key))))
            summary['temporal_energy_J_m'][key]={label:cumulative_at(hh,end)-cumulative_at(hh,t[event]) for label,hh in [('primary',h),('half',fh)]}
        summary['common_end_s']=float(end);summary['requested_end_reached']=bool(abs(ft[-1]-t[-1])<1e-8)
        report['temporal_comparison']=summary
        fig,aa=plt.subplots(2,2,figsize=(10.5,6.8));tt=times-t[event]
        for key,tag,style in [('mean_mature_theta_primary','Primary','-'),('mean_mature_theta_half','Half limits','--')]:
            aa[0,0].semilogy(tt,[r[key] for r in rows],style,label=tag)
        aa[0,0].set(title='Mature-patch frictional age',ylabel='Mean normalized θ',xlabel='Time from replay start (s)')
        for key,tag in [('mean_mature_theta','Mean mature state'),('maximum_speed_m_s','Maximum slip rate')]:
            aa[0,1].plot(tt,[100*r[key+'_absolute_difference']/abs(r[key+'_half']) for r in rows],label=tag)
        aa[0,1].axhline(1,color='k',ls=':',lw=.8,label='Declared 1% comparison target')
        aa[0,1].set(title='Temporal sensitivity',ylabel='Difference relative to half limits (%)',xlabel='Time from replay start (s)')
        xx=s[-1]['x_m'];aa[1,0].plot(xx,dd,label='Primary');aa[1,0].plot(xx,ff,'--',label='Half limits')
        aa[1,0].axvline(9,color='k',ls=':',lw=.8);aa[1,0].set(xlim=(8.8,9.2),title='Added damage at common endpoint',xlabel='x (m)',ylabel='a−a₀')
        aa[1,1].plot([r['kernel_centre_x_m'] for r in ep],[r['primary_Pa']-r['half_Pa'] for r in ep])
        aa[1,1].set(title='Common averaged work-conjugate resistance',xlabel='Gaussian kernel centre x (m), σ=.375 m',ylabel='Primary − half limits (Pa)')
        for axes in aa.flat:
            axes.grid(alpha=.2)
            if axes.get_legend_handles_labels()[0]:axes.legend(fontsize=7)
        fig.tight_layout();fig.savefig(args.output/'uncertainty.png',dpi=160);plt.close(fig)
    # Physical energy: full trajectory and the actual rapid-growth window.
    energies=[]
    for label,(hh,_,_) in runs.items():
        tt=value(hh,'time_s')
        for window,start in [('full',0.),('rapid',374.499),('replay',t[event])]:
            if label.startswith('Matched neutral') and window != 'full':
                continue  # Its large steps do not resolve the rapid-window endpoints.
            ids=np.flatnonzero(tt>start)
            if not len(ids):continue
            total=lambda k:float(sum(value(hh,k)[ids]))
            friction=total('friction_J_m');radiation=total('radiation_J_m');fracture=total('fracture_increment_J_m')
            absolute=float(sum(abs(value(hh,'physical_remainder_J_m')[ids])))
            energies.append(dict(case=label,window=window,window_start_s=float(tt[max(0,ids[0]-1)]),
                first_end_time_s=float(tt[ids[0]]),last_end_time_s=float(tt[ids[-1]]),
                bulk_increment_J_m=total('bulk_increment_J_m'),bond_increment_J_m=total('bond_increment_J_m'),
                trapezoid_work_J_m=total('trapezoid_work_J_m'),friction_J_m=friction,radiation_J_m=radiation,
                Maxwell_J_m=total('maxwell_J_m'),fracture_J_m=fracture,
                signed_physical_remainder_J_m=total('physical_remainder_J_m'),absolute_physical_remainder_J_m=absolute,
                absolute_over_sliding=absolute/(friction+radiation) if friction+radiation else None,
                absolute_over_fracture=absolute/fracture if fracture else None,
                BE_bulk_J_m=total('BE_bulk_J_m'),surface_remainder_J_m=total('surface_remainder_J_m')))
    save_csv(args.output/'energy-windows.csv',energies)
    report['energy_windows']=energies
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
