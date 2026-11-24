"""Resolve strip 15's centered score on the two captured selector inputs."""
import argparse
import csv
import json
from pathlib import Path

import mpmath as mp

from turn import candidate


def refine(row, left, right):
    """Golden-section maximization within neighboring scan samples."""
    ratio = (mp.sqrt(5)-1)/2
    a, b = right-ratio*(right-left), left+ratio*(right-left)
    def evaluate(x):
        return candidate(row, x)['score']
    fa, fb = evaluate(a), evaluate(b)
    while right-left > mp.mpf('1e-22'):
        if fa < fb:
            left, a, fa = a, b, fb
            b = left+ratio*(right-left)
            fb = evaluate(b)
        else:
            right, b, fb = b, a, fa
            a = right-ratio*(right-left)
            fa = evaluate(a)
    return (left+right)/2


def profile(row, probes):
    angles = mp.linspace(-mp.pi/4, mp.pi/4, 513)
    scores = [candidate(row, x)['score'] for x in angles]
    bins = [i for i in range(1,len(angles)-1) if scores[i-1] < scores[i] >= scores[i+1]]
    peaks = [refine(row, angles[i-1], angles[i+1]) for i in bins]
    candidates = [*peaks, angles[0], angles[-1]]
    peak = max(candidates, key=lambda x: candidate(row,x)['score'])
    own = mp.mpf(float(row['angle_rad']))
    if peak not in peaks or abs(own-peak) > angles[1]-angles[0]:
        raise ValueError('Selected direction and best interior peak occupy different basins')
    maximum = candidate(row,peak)['score']
    tie = 128*mp.mpf(2)**-52*sum(mp.mpf(float(row[k])) for k in ('m3','m4'))
    evidence = {}
    for name, x in [('maximum',peak), *probes.items(), ('incoming',0),
                    ('left_boundary',angles[0]), ('right_boundary',angles[-1])]:
        observed = candidate(row,x)
        evidence[name] = dict(beta=x, loss=maximum-observed['score'], **observed)
    for i, x in enumerate(peaks):
        value=candidate(row,x)['score']
        evidence[f'sector_peak_{i}'] = dict(beta=x,score=value,loss=maximum-value)
    # Test this fixture's score scale against the two recorded choices and
    # a physically distinct direction. Retain the strict angle check separately.
    competitors=[maximum-candidate(row,x)['score'] for x in candidates if x!=peak]
    if not max(evidence[k]['loss'] for k in probes) < tie < min(evidence['incoming']['loss'],*competitors):
        raise ValueError('The production update threshold fails the near/off-peak control')
    with mp.workdps(85):
        radius=mp.mpf('1e-7')
        refined=refine(row,peak-radius,peak+radius)
        precision_change=refined-peak
        if abs(precision_change)>mp.mpf('1e-21'):
            raise ValueError('Peak changed on refining the bracket at 85 digits')
    return dict(own_error=own-peak, threshold=tie, scan_intervals=len(angles)-1,
                interior_peak_count=len(peaks), refinement_change_rad=precision_change,
                probes=evidence), (angles,scores)


def plot(rows, profiles, scans, dest):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, (sector, zoom) = plt.subplots(1,2,figsize=(10.4,4.0),layout='constrained')
    origin = mp.mpf('-.1478616')
    colors = {'parent':'#b34a12','donor-parent':'#184a79'}
    for row, result, (angles,scores) in zip(rows,profiles,scans,strict=True):
        donor = row['source']=='donor-parent'
        color = colors[row['source']]
        label = 'Donor' if donor else 'Target'
        style = '--' if donor else '-'
        sector.plot([float(x) for x in angles], [float(x*1e9) for x in scores],
                    style,color=color,lw=1.7,label=label)
        peak = result['probes']['maximum']
        xs = mp.linspace(origin-mp.mpf('1.6e-7'),origin+mp.mpf('3.5e-7'),121)
        # Subtract at working precision before handing any value to Matplotlib.
        ys = [(candidate(row,x)['score']-peak['score'])*mp.mpf('1e20') for x in xs]
        zoom.plot([float((x-origin)*10**7) for x in xs],list(map(float,ys)),style,color=color,lw=1.7)
        own = result['probes'][label]
        zoom.scatter(float((own['beta']-origin)*10**7),float(-own['loss']*10**20),
                     marker='o',facecolor='white',edgecolor=color,s=48,zorder=5)
        zoom.scatter(float((peak['beta']-origin)*10**7),0,marker='+' if donor else 'x',
                     color=color,s=70,zorder=6)
    sector.axvline(0,color='.5',lw=.7,ls=':')
    sector.set(title='Full forward sector',xlabel='Angle from the 30° seed tangent, β (rad)',
               ylabel='Direction score  ($10^{-9}$ Pa² m²)',xlim=(-float(mp.pi/4),float(mp.pi/4)))
    sector.legend(frameon=False)
    zoom.set(title='Near the winning direction',xlabel='β + 0.1478616 rad  ($10^{-7}$ rad)',
             ylabel='Score minus own maximum  ($10^{-20}$ Pa² m²)')
    zoom.legend(handles=[Line2D([],[],ls='',marker=m,color=c,markerfacecolor='white',label=t)
                         for m,c,t in [('o','.25','Selected angle'),('+',colors['donor-parent'],'Donor maximum'),
                                       ('x',colors['parent'],'Target maximum')]],frameon=False,loc='lower left')
    for ax in (sector,zoom):
        ax.spines[['right','top']].set_visible(False)
        ax.grid(axis='y',alpha=.2)
    for suffix in ('pdf','png'):
        fig.savefig(dest/f'strip15-direction.{suffix}',dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('moments',type=Path)
    p.add_argument('destination',type=Path)
    p.add_argument('--common-choices',type=Path,help='Decisions of both compiled selectors on the opposite inputs')
    args = p.parse_args()
    mp.mp.dps = 65
    with args.moments.open() as stream:
        rows = list(csv.DictReader(stream))
    if {r['source'] for r in rows} != {'donor-parent','parent'} or len(rows)!=2:
        raise ValueError('Need one captured input per selector')
    probes = {('Donor' if r['source']=='donor-parent' else 'Target'):mp.mpf(float(r['angle_rad'])) for r in rows}
    choices={}
    if args.common_choices:
        with args.common_choices.open() as stream:
            records=list(csv.DictReader(stream))
        if sorted((r['input_moments'],r['production_selector']) for r in records)!=[
                ('donor-parent','parent'),('parent','donor-parent')]:
            raise ValueError('Missing common-input selector decision')
        choices={r['input_moments']:{'other_selector_same_moments':mp.mpf(float(r['beta_rad']))} for r in records}
    profiles, scans = zip(*(profile(row,probes|choices.get(row['source'],{})) for row in rows))
    args.destination.mkdir(parents=True,exist_ok=True)
    report = {r['source']:v for r,v in zip(rows,profiles,strict=True)}
    (args.destination/'peak.json').write_text(json.dumps(report,indent=2,default=lambda x:mp.nstr(x,24))+'\n')
    plot(rows,profiles,scans,args.destination)


if __name__=='__main__':
    main()
