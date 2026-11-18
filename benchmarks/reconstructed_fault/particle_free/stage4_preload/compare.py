#!/usr/bin/env python3
"""Compare native margins and work after the sequential numerical pair exits."""
from calibrate import *
import csv,shutil

def table(path):return np.atleast_1d(np.genfromtxt(path,delimiter=',',names=True))
def main():
    status=json.loads((BUILD/'comparison-finished.json').read_text());assert status['complete']
    selection=json.loads((BUILD/'selection.json').read_text())
    z=np.load(OUT/'calibration-arrays.npz');cal=z['rows'];a0=cal[0]
    x,y,j,B,mass,fracture=a0[:,:6].T
    coefficients=np.linalg.solve(np.vander(z['gammas']/1e-4,3),cal[:,:,6]-fracture).T
    predicted=np.polyval(coefficients.T,selection['gamma']/1e-4)/mass
    tip=(x==6+H/2)&(abs(y-2)<H)
    left=(x==2-H/2)&(abs(y-2)<H)
    critical=np.flatnonzero(tip)[np.argmax(predicted[tip])]
    intact=np.repeat(np.max(abs(B.reshape(-1,9)),axis=1)==0,9)
    masks=dict(tip=tip,left_tip=left,intact_ahead=intact&(x>=6.3125)&(x<6.5)&(abs(y-2)<.125),
               all_intact=intact,remote=(abs(x-4)>3)|(abs(y-2)>1),
               old_tail=(x>=6)&(x<6.3125)&(abs(y-2)<.25)&~intact)
    coords=np.arange(5.875,7.501,H)
    chi=np.array([direction(a0,c) for c in coords]);area=chi@mass
    histories={};records={};axial={};fields={};tip_rows={};last_raw={}
    for law in ('weakening','neutral'):
        name='near-'+law;run=RUNS/('stage4-preload-'+name);out=run/'output'
        summary=table(out/'pilot-summary.csv');energy=table(out/'bulk-accepted.csv')
        regions=table(out/'release-regions.csv');rows=[];profiles=[];local_tip=[]
        for state in summary:
            step=int(state['step']);a=scan(name,step);net=(a[:,7]-a[:,5])/mass
            assert np.array_equal(a[:,:4],a0[:,:4]),'Fixed phase or mesh changed.'
            assert np.max(abs(a[:,5]-fracture))<1e-12
            assert np.max(abs(regions[regions['step']==step]['added_fracture_row_partition']))==0
            r={key:float(state[key]) for key in summary.dtype.names};r['step']=step
            r['critical_row_net']=float(net[critical])
            for key,mask in masks.items():
                r[key+'_max_net']=float(np.max(net[mask]));r[key+'_positive_rows']=int(np.count_nonzero(net[mask]>1e-8))
            for key,center in [('tip_cap',6.03125),('ahead',6.09375),('intact_probe',6.375)]:
                direction_=direction(a,center);r[key+'_net']=float(direction_@(a[:,7]-a[:,5])/(direction_@mass))
            r.update(added_fracture=0.,physical_work_error=0.,mechanical_residual=0.)
            if step:
                e=energy[energy['step']==step][0]
                assert int(e['publications'])==step
                r['physical_work_error']=float(e['Wext']-e['energy_change']-e['Db']-e['Dc']-e['Df'])
                r['mechanical_residual']=float(e['mechanical_residual'])
            else:
                records[law]=dict(initial_fixed_plane_prediction_error_Pa=float(np.max(abs((a[:,6]-fracture)/mass-predicted))))
            rows.append(r);profiles.append(chi@(a[:,7]-a[:,5])/area);local_tip.append(net[tip|left])
            if step==0 or step==selection['steps']:fields[(law,step)]=net.reshape(128,256,9).max(axis=2)
            if step==selection['steps']:last_raw[law]=a
        assert len(rows)==selection['steps']+1
        histories[law]=rows;axial[law]=np.array(profiles);tip_rows[law]=np.array(local_tip)
        with (OUT/f'{law}-history.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        last=table(out/f'tip-sections-{selection["steps"]:04d}.csv')
        positive=float(np.sum(last['ds']*last['positive_w']));negative=float(np.sum(last['ds']*last['negative_w']))
        command=json.loads((run/'command.json').read_text())
        records[law].update(initial=rows[0],final=rows[-1],exit_code=command['exit_code'],seconds=command['seconds'],
            positive_rate_integral=positive,negative_rate_integral=negative,
            cancellation_fraction=1-abs(positive+negative)/(positive-negative),
            signed_cumulative_physical_work_error=sum(r['physical_work_error'] for r in rows),
            accumulated_absolute_physical_work_error=sum(abs(r['physical_work_error']) for r in rows),
            accumulated_absolute_external_work=float(np.sum(abs(energy['Wext']))),
            maximum_mechanical_residual=max(r['mechanical_residual'] for r in rows))
    weak=histories['weakening'];neutral=histories['neutral']
    difference={key:[w[key]-n[key] for w,n in zip(weak,neutral)] for key in
                ('critical_row_net','tip_max_net','tip_cap_net','ahead_net','intact_probe_net','peak_rate','integrated_rate','integrated_slip')}
    tip_difference=tip_rows['weakening']-tip_rows['neutral']
    result=dict(selection=selection,critical_row=a0[critical,:4].tolist(),laws=records,difference_weakening_minus_neutral=difference,
                largest_signed_difference_over_all_tip_cell_rows=float(np.max(tip_difference[1:])),
                maximum_positive_excess_over_all_tip_cell_rows=max(0.,float(np.max(tip_difference[1:]))),
                maximum_absolute_difference_over_all_tip_cell_rows=float(np.max(abs(tip_difference))),
                weakening_positive_control_negative_tip_rows=int(np.sum((tip_rows['weakening']>0)&(tip_rows['neutral']<0))),
                maximum_absolute_tip_row_history_difference=max(abs(v) for v in difference['critical_row_net']),
                maximum_positive_tip_row_excess=max(0.,max(difference['critical_row_net'][1:])),
                final_peak_rate_ratio=weak[-1]['peak_rate']/neutral[-1]['peak_rate'],
                final_slip_ratio=weak[-1]['integrated_slip']/neutral[-1]['integrated_slip'],
                geometry_note='All native phase coefficients are unchanged at every recorded stage: connected contour and fracture inventory cannot advance in this fixed-phase comparison.')
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    a=last_raw['weakening'];b=last_raw['neutral'];delta=(a[:,7]-b[:,7])/mass
    spatial={}
    for name,mask in [('near_cap',(x>=6)&(x<6.15625)&(abs(y-2)<.125)),
                      ('immediately_ahead',(x>=6.03125)&(x<6.15625)&(abs(y-2)<.125)),
                      ('old_taper',(x>=5.5)&(x<6.3125)&(abs(y-2)<.25))]:
        indices=np.flatnonzero(mask);i=indices[np.argmax(delta[mask])]
        spatial[name]=dict(min_difference=float(np.min(delta[mask])),max_difference=float(np.max(delta[mask])),max_location=a[i,:3].tolist(),
                           weak_net=float((a[i,7]-a[i,5])/mass[i]),control_net=float((b[i,7]-b[i,5])/mass[i]))
    (OUT/'final-spatial-differences.json').write_text(json.dumps(spatial,indent=2)+'\n')
    np.savez_compressed(OUT/'comparison-fields.npz',coordinates=coords,weakening_axial=axial['weakening'],neutral_axial=axial['neutral'],tip_row_locations=a0[tip|left,:3],weakening_tip_rows=tip_rows['weakening'],neutral_tip_rows=tip_rows['neutral'],
                        initial_row_max=fields[('weakening',0)],weakening_final_row_max=fields[('weakening',selection['steps'])],neutral_final_row_max=fields[('neutral',selection['steps'])])
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for law,color in [('weakening','tab:red'),('neutral','tab:blue')]:
        rows=histories[law];t=np.array([r['stage_time'] for r in rows])*1e3;label='Weakening' if law=='weakening' else 'Matched b=a'
        axs[0,0].plot(t,[r['peak_rate']*1e3 for r in rows],'-o',color=color,ms=3,label=label)
        axs[0,1].plot(t,[r['critical_row_net'] for r in rows],'-o',color=color,ms=3,label=label)
        axs[1,0].plot(t,[r['ahead_net'] for r in rows],color=color,label=label+', ahead')
        axs[1,0].plot(t,[r['intact_probe_net'] for r in rows],'--',color=color,label=label+', intact')
    t=np.array([r['stage_time'] for r in weak])*1e3
    axs[1,1].plot(t,np.array(difference['critical_row_net'])*1e6,'-o',label='Limiting tip row')
    axs[1,1].plot(t,np.array(difference['tip_cap_net'])*1e6,'--',label='Smooth tip cap')
    axs[0,0].set(ylabel='Peak stage slip rate (mm/s)',title='Slip-law separation')
    axs[0,1].axhline(0,color='k',lw=.7);axs[0,1].set(ylabel='Net damage driving (Pa)',title='Most susceptible initial axial-tip row')
    axs[1,0].set(ylabel='Mean net damage driving (Pa)',title='Fixed smooth probes')
    axs[1,1].axhline(0,color='k',lw=.7);axs[1,1].set(ylabel='Weakening − b=a (µPa)',title='Tip-driving excess, signed')
    for ax in axs.flat:ax.set_xlabel('Stage time (ms)');ax.legend(fontsize=8)
    for ext in ('png','pdf'):fig.savefig(OUT/f'law-comparison.{ext}',dpi=170)
    fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    im=axs[0].imshow(fields[('weakening',0)],origin='lower',extent=[0,8,0,4],vmin=-2,vmax=2,cmap='coolwarm',aspect='equal')
    axs[0].set(xlabel='x (m)',ylabel='y (m)',title='Initial maximum row driving per cell (Pa)');fig.colorbar(im,ax=axs[0],shrink=.7,extend='both')
    for law,color in [('weakening','tab:red'),('neutral','tab:blue')]:
        axs[1].plot(coords,axial[law][-1],color=color,label=law+' final stage')
    axs[1].plot(coords,axial['weakening'][0],'k--',label='Common initial')
    axs[1].axhline(0,color='k',lw=.7);axs[1].axvline(6.00205438,color='gray',ls=':');axs[1].axvspan(6.3125,7.5,color='gray',alpha=.08)
    axs[1].set(xlabel='Probe center x (m)',ylabel='Mean net driving (Pa)',title='Ahead probes remain unfavorable');axs[1].legend(fontsize=8)
    for ext in ('png','pdf'):fig.savefig(OUT/f'margin-map.{ext}',dpi=170)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
