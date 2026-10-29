#!/usr/bin/env python3
"""Separate common-path evolution errors, reconstruction and fresh causal runs."""
import csv
import json
import numpy as np
from investigate_cohesive_interface_bulk_contact import WORK, OUT, ARCS, build, save, table, PATH_WORK
from evolve_cohesive_interface_bulk_contact import initialize
from cohesive_interface_path import functionals, PathKernel
from fresh_cohesive_interface_bulk_contact import initial as fresh_initial


def main():
    short={};rows=[];profiles=[]
    for p in sorted(OUT.glob('short-*-evolution.json')):
        d=json.loads(p.read_text());label=d['label'];short[label]=d
        folder=WORK/label;meta=json.loads((WORK/d['kernel']/'input.json').read_text())
        k=PathKernel(WORK/d['kernel'],ARCS,'work');_,reconstruction=initialize(k)
        # Diagnostic metadata only: reclassify absent tied reactions correctly.
        # The stored initial/final states and every physical trial stay unchanged.
        d['initial']=reconstruction;save(p,d);save(folder/'reconstruction.json',reconstruction)
        r=np.genfromtxt(OUT/(label+'-history.csv'),delimiter=',',names=True)
        first=json.loads((folder/'initial.json').read_text());last=json.loads((folder/'final.json').read_text())
        f=d['final_functionals'];oldslip=k.w@np.array(first['histories']['jump']);slip=k.w@np.array(last['histories']['jump'])
        rows.append(dict(label=label,dt_s=d['dt_s'],growth_m=d['growth_m'],slip_increment_m2=float(slip-oldslip),
            process_x_m=f['process_front_x_m'],process_y_m=f['process_front_y_m'],process_tangent_deg=f['process_front_tangent_deg'],
            full_x_m=f['full_front_x_m'],full_y_m=f['full_front_y_m'],full_tangent_deg=f['full_front_tangent_deg'],
            physical_error_J_m=d['absolute_physical_remainder_J_m'],relative_sliding=d['relative_sliding'],relative_fracture=d['relative_fracture'],
            initial_free_residual_Pa=reconstruction['initial_unbalanced_free_residual_Pa'],initial_reconstruction_energy_J_m=reconstruction['reconstruction_storage_difference_J_m'],
            maximum_true_weak_Pa=float(max(r['weak_error_Pa'])),minimum_contact_Pa=float(min(r['minimum_normal_Pa'])),
            maximum_gap_m=float(max(r['normal_gap_m'])),maximum_BE_ledger_J_m=d['maximum_BE_ledger_J_m'],
            publications=d['publications'],seconds=d['seconds']))
        c=np.r_[last['wall_memory'],last['memory']];pn=10000+k.N@c
        for q in range(len(k.w)):
            profiles.append(dict(label=label,time_s=last['clock_s'],x_m=float(k.s['x_m'][q]),y_m=float(k.s['y_m'][q]),weight_m=float(k.w[q]),
                damage=last['histories']['damage'][q],slip_m=last['histories']['jump'][q],normal_Pa=float(pn[q])))
        k.close()
    table(OUT/'short-comparison.csv',rows);table(OUT/'short-profiles.csv',profiles)
    lookup={r['label']:r for r in rows};errors=[]
    pairs=[('short-narrow64','short-narrow96','original lifting mesh spread'),('short-wide64','short-wide96','fixed lifting mesh spread'),
        ('short-wide-Q3','short-wide96','fixed lifting similar-DoF alternative'),
        ('short-wide-Q3','short-wide64','fixed lifting degree spread'),('short-fitted64','short-fitted96','fitted resolution'),
        ('short-fitted-Q3','short-fitted96','fitted alternative'),('short-wide96','short-fitted96','fixed lifting vs fitted'),
        ('short-offset96','short-wide96','off-grid support'),('short-fitted96','short-fitted96-half','fitted time'),
        ('short-wide96','short-wide96-half','fixed lifting time')]
    if 'short-narrow-Q3' in lookup:pairs.append(('short-narrow-Q3','short-narrow64','original lifting degree spread'))
    for left,right,meaning in pairs:
        a,b=lookup[left],lookup[right]
        errors.append(dict(comparison=meaning,left=left,right=right,growth_difference_m=a['growth_m']-b['growth_m'],
            growth_relative=abs(a['growth_m']-b['growth_m'])/b['growth_m'],slip_relative=abs(a['slip_increment_m2']-b['slip_increment_m2'])/abs(b['slip_increment_m2']),
            full_front_difference_m=float(np.hypot(a['full_x_m']-b['full_x_m'],a['full_y_m']-b['full_y_m'])),
            process_front_difference_m=float(np.hypot(a['process_x_m']-b['process_x_m'],a['process_y_m']-b['process_y_m']))))
    table(OUT/'short-errors.csv',errors)
    causal=[]
    for label,root in [('work64',PATH_WORK),('work96',PATH_WORK),('causal-wide64',WORK),('causal-wide96',WORK)]:
        path=(OUT if root==WORK else ROOT_OUT)/(label+'-summary.json')
        if not path.exists():continue
        d=json.loads(path.read_text());f=d['final']
        causal.append(dict(label=label,damage_integral_m=f['new_damage_integral_m'],full_x_m=f['full_front_x_m'],full_y_m=f['full_front_y_m'],
            full_tangent_deg=f['full_front_tangent_deg'],process_x_m=f['process_front_x_m'],process_y_m=f['process_front_y_m'],process_tangent_deg=f['process_front_tangent_deg'],
            geometry_publications=d['geometry_publications'],physical_publications=d['physical_publications'],seconds=d['seconds'],
            relative_work_sliding=d['energy']['relative_sliding'],relative_work_fracture=d['energy']['relative_fracture'],
            all_transfers_bitwise=all(e['transfer']['old_stress_pressure_columns_bitwise'] and e['transfer']['old_histories_bitwise'] for e in d['events']),
            maximum_transfer_stress_Pa=max(e['transfer']['physical_tensor_pressure_error_Pa'] for e in d['events']),
            maximum_transfer_energy_J_m=max(abs(e['transfer']['energy_change_J_m']) for e in d['events'])))
    if causal:table(OUT/'causal-comparison.csv',causal)
    fresh=[];fresh_profiles=[]
    for label in ['fresh-wide64','fresh-wide96','fresh-fitted64','fresh-fitted96']:
        path=OUT/(label+'-summary.json')
        if not path.exists():continue
        d=json.loads(path.read_text());s=json.loads((WORK/label/'final.json').read_text())
        k=PathKernel(WORK/s['kernel_folder'],ARCS,'work');f=d['final']
        rebuilt,initial_checks=fresh_initial(k)
        saved_initial=json.loads((WORK/label/'initial.json').read_text())
        assert rebuilt==saved_initial
        save(OUT/(label+'-initial-checks.json'),initial_checks)
        fresh.append(dict(label=label,damage_integral_m=f['new_damage_integral_m'],full_x_m=f['full_front_x_m'],full_y_m=f['full_front_y_m'],
            full_arc_m=f['full_front_arc_m']-3,process_x_m=f['process_front_x_m'],process_y_m=f['process_front_y_m'],
            relative_work_sliding=d['energy']['relative_sliding'],relative_work_fracture=d['energy']['relative_fracture'],
            initial_weak_Pa=json.loads((OUT/(label+'-initial-checks.json')).read_text())['true_weak_Pa'],
            physical_publications=d['physical_publications'],geometry_publications=d['geometry_publications'],seconds=d['seconds']))
        c=np.r_[s['wall_memory'],s['memory']];pn=10000+k.N@c
        for q in range(len(k.w)):
            fresh_profiles.append(dict(label=label,x_m=float(k.s['x_m'][q]),y_m=float(k.s['y_m'][q]),weight_m=float(k.w[q]),damage=s['histories']['damage'][q],
                slip_m=s['histories']['jump'][q],normal_Pa=float(pn[q])))
        k.close()
    if fresh:
        table(OUT/'fresh-comparison.csv',fresh);table(OUT/'fresh-profiles.csv',fresh_profiles)
        f={r['label']:r for r in fresh};differences=[]
        for a,b in [('fresh-wide64','fresh-wide96'),('fresh-fitted64','fresh-fitted96'),('fresh-wide96','fresh-fitted96')]:
            if a not in f or b not in f:continue
            x,y=f[a],f[b]
            differences.append(dict(left=a,right=b,damage_difference_m=x['damage_integral_m']-y['damage_integral_m'],
                damage_relative=abs(x['damage_integral_m']-y['damage_integral_m'])/y['damage_integral_m'],
                full_front_difference_m=float(np.hypot(x['full_x_m']-y['full_x_m'],x['full_y_m']-y['full_y_m'])),
                process_front_difference_m=float(np.hypot(x['process_x_m']-y['process_x_m'],x['process_y_m']-y['process_y_m']))))
        if differences:table(OUT/'fresh-errors.csv',differences)
    own=[]
    for name in ['wide96','fitted96']:
        p=OUT/('own-time-'+name+'-evolution.json')
        if not p.exists():continue
        d=json.loads(p.read_text());baseline=json.loads((OUT/('fresh-'+name+'-summary.json')).read_text())
        a,b=baseline['final'],d['final_functionals'];window_growth=a['new_damage_integral_m']-d['initial_functionals']['new_damage_integral_m']
        own.append(dict(label=name,source_clock_s=d['initial']['clock_s'],same_basis_restart=True,
            growth_difference_m=window_growth-d['growth_m'],growth_relative=abs(window_growth-d['growth_m'])/d['growth_m'],
            full_front_difference_m=float(np.hypot(a['full_front_x_m']-b['full_front_x_m'],a['full_front_y_m']-b['full_front_y_m'])),
            process_front_difference_m=float(np.hypot(a['process_front_x_m']-b['process_front_x_m'],a['process_front_y_m']-b['process_front_y_m'])),
            physical_remainder_J_m=d['absolute_physical_remainder_J_m'],relative_fracture=d['relative_fracture'],
            publication_increment=d['publication_increment'],seconds=d['seconds']))
    if own:table(OUT/'own-time-errors.csv',own)
    save(OUT/'evolution-assessment.json',dict(common_path=errors,
        short_scope='Separately initialized diagnostic responses from common saved surface/memory data; initial reconstruction differences are measured, not charged as physical work.',
        initial_tied_reaction_note='Raw nodal force without newly tied reactions is not a free residual; least-squares elimination of tied reactions reports the irreducible free part.',
        all_native_rejection_retry_restart=all(d['checks']['native_rejection_retry_restart_bitwise'] for d in short.values()),
        all_once_only=all(d['checks']['duplicate_rejected'] and d['steps']==d['publications'] for d in short.values()),
        maximum_FD_relative=max(d['checks']['FD_relative'] for d in short.values())))


ROOT_OUT=OUT.parent/'cohesive_interface_path'
if __name__=='__main__':main()
