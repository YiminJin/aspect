#!/usr/bin/env python3
"""Coordinate-based rank comparison; retained entries do not qualify a trajectory."""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from assess_native_tangent_cost import records

E = Path(__file__).resolve().parent / 'native_mpi_evidence'
SCALE = 64.00000000008


def table(path, keys=()):
    with path.open() as file:
        rows = list(csv.DictReader(file))
    def value(text):
        try:
            return float(text)
        except ValueError:
            return text
    return sorted(rows, key=lambda row: tuple(value(row[k]) for k in keys)) if keys else rows


def differences(a, b, keys):
    left, right = table(a, keys), table(b, keys)
    assert len(left) == len(right)
    result = {}
    for l, r in zip(left, right):
        assert all(l[k] == r[k] for k in keys), 'Compare stable coordinates, never DoF order'
        for k in l:
            if k in keys:
                continue
            result[k] = max(result.get(k, 0.), abs(float(l[k])-float(r[k])))
    return result


def run_record(tag):
    record = json.loads((E / (tag+'-run.json')).read_text())
    assert record['returncode'] == 0, f'{tag} did not complete'
    text = (E / (tag+'.log')).read_text()
    record['phase_costs'] = records(text, 'Endpoint cost:')
    tangents = records(text, 'Endpoint linear tangent:')
    profiles = records(text, 'Fault linear profile:')
    record['linear_tangents'] = dict(calls=len(tangents),
        seconds=sum(row['seconds'] for row in tangents),
        Krylov=sum(row['Krylov'] for row in tangents),
        inner_calls=sum(row['linear calls'] for row in tangents),
        maximum_seconds=max((row['seconds'] for row in tangents),default=0.))
    record['linear_profiles'] = {key:sum(row[key] for row in profiles) for key in profiles[0]} if profiles else {}
    # Existing native TimerOutput scopes include assembly and are nested.
    # A recovery run's summary includes both its failed and accepted attempts.
    record['native_timer_sections'] = {
        name.strip(): dict(calls=int(calls), seconds=float(seconds))
        for name, calls, seconds in re.findall(
            r'^\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*([0-9.eE+\-]+)s\s*\|', text, re.MULTILINE)}
    if 'Fixed-entry diagnostic:' in text:
        segment=text.split('Fixed-entry diagnostic:',1)[1]
        if 'Coordinated endpoint prepublication failure:' in segment:
            segment=segment.split('Coordinated endpoint prepublication failure:',1)[1]
        segment=segment.split('Fixed-entry whole-step seconds=',1)[0]
        actions=records(segment,'Endpoint linear tangent:')
        record['accepted_entry_actions']=dict(calls=len(actions),
            Krylov=sum(row['Krylov'] for row in actions),
            inner_calls=sum(row['linear calls'] for row in actions),
            seconds=sum(row['seconds'] for row in actions))
        sums=records(segment,'Fault linear profile:')
        record['accepted_entry_linear_profile']={key:sum(row[key] for row in sums) for key in sums[0]} if sums else {}
    record['accepted_entry_seconds'] = [float(v) for v in re.findall(r'Fixed-entry accepted-step seconds=(\S+)', text)]
    record['whole_entry_seconds'] = [float(v) for v in re.findall(r'Fixed-entry whole-step seconds=(\S+),', text)]
    record['failure_attempt_seconds'] = [float(v) for v in re.findall(r'Fixed-entry failure attempt seconds=(\S+)', text)]
    out = Path(record['output'])
    memory = table(out / 'mpi-memory.csv')
    record['peak_memory_MiB_by_rank'] = {rank: max(float(row['peak_RSS_KiB'])/1024 for row in memory if row['rank'] == rank)
                                      for rank in sorted({row['rank'] for row in memory})}
    record['communication_epochs'] = {}
    for file in sorted(E.glob(tag+'-epoch-*-rank-*.csv')):
        rows = table(file)
        record['communication_epochs'][file.name] = dict(seconds=sum(float(row['seconds']) for row in rows),
                                                         operations=rows)
    return record


def operator_family(tags):
    runs = [run_record(tag) for tag in tags]
    assert len({r['executable_sha256'] for r in runs}) == 1
    assert len({r['plugin_sha256'] for r in runs}) == 1
    base = Path(runs[0]['output'])
    results = []
    for record in runs:
        out = Path(record['output'])
        rows = table(out/'operators.csv')
        maxima = {k:max(abs(float(row[k])) for row in rows) for k in rows[0] if k != 'case'}
        assert len(rows) == 15
        relative = {}
        baseline = {row['case']:row for row in table(base/'operators.csv')}
        for row in rows:
            relative[row['case']] = {k:abs(float(row[k])-float(baseline[row['case']][k])) for k in row if k != 'case'}
        energy_difference = max(max(values[k] for k in ['bulk_error','cohesive_error','fracture_error','potential_error']) for values in relative.values())
        # Original dimensional form of 32 eps max(1, |J/scale|); no larger allowance.
        minimum_allowance = 32*np.finfo(float).eps*SCALE
        assert energy_difference <= minimum_allowance, 'Rank energy uncertainty exceeds original line-search allowance'
        results.append(dict(ranks=record['ranks'],maxima=maxima,rank_differences=relative,
                            maximum_energy_rank_difference=energy_difference,
                            minimum_line_search_allowance=minimum_allowance,
                            gradient_differences=differences(base/'operator-gradients.csv',out/'operator-gradients.csv',('case','y'))))
    return dict(runs=runs,results=results)


def entry_family(tags):
    runs = [run_record(tag) for tag in tags]
    assert len({r['executable_sha256'] for r in runs}) == 1
    assert len({r['plugin_sha256'] for r in runs}) == 1
    base = Path(runs[0]['output'])
    results = []
    for tag, record in zip(tags,runs):
        out = Path(record['output'])
        text = (E/(tag+'.log')).read_text()
        targets = [float(x)*SCALE for x in re.findall(r'Endpoint phase target: initial=[^,]+, target=(\S+)',text)]
        entries = table(out/'fixed-entries.csv')
        for row in entries:
            name = row['case'];values = {k:float(v) for k,v in row.items() if k != 'case'}
            assert values['healing'] == 0 and values['I_min'] > 0
            assert values['KKT'] <= targets[-1]
            fields = table(out/(name+'-phase.csv'),('x','y'))
            assert all(float(p['lower']) <= float(p['phase']) < 1 for p in fields)
            transverse_rows = {}
            for point in fields:
                transverse_rows.setdefault(float(point['y']), []).append(float(point['phase']))
            values.update(case=name,physical_KKT_target=targets[-1],
                          global_reflection=values['reflected_phase_error'] < values['phase_error'],
                          tangential_phase_variation=max(np.ptp(row) for row in transverse_rows.values()),
                          energy_decrease=values['initial_potential']-values['final_potential'])
            if 'saddle' in name:
                allowance=32*np.finfo(float).eps*max(SCALE,abs(values['initial_potential']))
                assert values['energy_decrease'] > 4*allowance
                values['line_search_allowance_J_per_m'] = allowance
            values['phase_rank_differences'] = differences(base/(name+'-phase.csv'),out/(name+'-phase.csv'),('x','y'))
            values['surface_rank_differences'] = differences(base/(name+'-surface.csv'),out/(name+'-surface.csv'),('x',))
            values['stress_rank_differences'] = differences(base/(name+'-stress.csv'),out/(name+'-stress.csv'),('x','y'))
            results.append(dict(ranks=record['ranks'],entry=values))
    return dict(runs=runs,results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--operators', nargs='+')
    parser.add_argument('--growth', nargs='+')
    parser.add_argument('--saddle', nargs='+')
    args = parser.parse_args()
    result = {}
    if args.operators:
        result['operators'] = operator_family(args.operators)
    if args.growth:
        result['growth'] = entry_family(args.growth)
    if args.saddle:
        result['saddle'] = entry_family(args.saddle)
    (E/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Wrote coordinate-based rank and cost assessment to', E/'assessment.json')
