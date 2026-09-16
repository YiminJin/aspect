"""Disjoint coupling costs, setup break-even and few-mode preconditioner cost."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
from mechanical_report import summarize
from compare import compare


def report(path):
    result = summarize(path)
    log = path.with_suffix('.log').read_text()
    totals = result['totals']
    verified = 'ASPECT_FAULT_COMPARE_COUPLING=1' in result['resources'].get('overrides', [])
    result['reference_checks_enabled'] = verified
    iterations = [int(row[0]) for row in result['accepted_linear_checks']]
    result['iteration_distribution'] = dict(mean=float(np.mean(iterations)),
        minimum=min(iterations), median=float(np.median(iterations)),
        p90=float(np.percentile(iterations, 90)), maximum=max(iterations), values=iterations)
    result['sparse'] = {}
    for action in ('B', 'G'):
        setup = totals.get(action+'_setup_s', 0.)
        applications = totals.get(action+'_sparse_calls', 0.)
        sparse = totals.get(action+'_sparse_s', 0.)
        references = totals.get(action+'_calls', 0.)
        reference = totals.get(action+'_s', 0.)
        saving = reference/references-sparse/applications if applications and references else None
        matrices = re.findall(r'Fault sparse '+action+r': rank0 entries=(\d+), bytes=(\d+)', log)
        result['sparse'][action] = dict(setup_seconds=setup, applications=applications,
            matvec_seconds=sparse, reference_seconds=reference,
            mean_break_even_applications=setup/len(matrices)/saving if matrices and saving and saving>0 else None,
            peak_rank0_bytes=max((int(m[1]) for m in matrices), default=0),
            peak_rank0_entries=max((int(m[0]) for m in matrices), default=0),
            max_relative_action_error=max(row.get(action+'_relative_error', 0.)
                                          for row in result['linearizations']) if verified else None)
    interface = re.findall(r'Fault interface: modes=(\d+), owned response bytes=(\d+), setup inclusive s=([\d.eE+\-]+)', log)
    result['interface_setup_inclusive_seconds'] = sum(float(x[2]) for x in interface)
    result['interface_setup_base_applications'] = sum(int(x[0]) for x in interface)
    result['outer_base_applications'] = totals['preconditioner_calls']-result['interface_setup_base_applications']
    result['interface_peak_rank0_response_bytes'] = max((int(x[1]) for x in interface), default=0)
    # Nested setup probes are charged to their actual B/G/base scopes. Count
    # each exclusive part once, never add inclusive setup a second time.
    parts = ('preconditioner_s', 'B_s', 'G_s', 'B_setup_s', 'G_setup_s',
             'B_sparse_s', 'G_sparse_s', 'interface_setup_s', 'interface_apply_s')
    result['dominant_cost_including_verification_seconds'] = sum(totals.get(k, 0.) for k in parts)
    result['dominant_cost_without_reference_checks_seconds'] = (
        result['dominant_cost_including_verification_seconds']
        -sum(totals.get(a+'_s', 0.) for a in ('B', 'G') if totals.get(a+'_sparse_calls', 0.)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = {'runs': {str(path): report(path) for path in args.paths}}
    if len(args.paths) == 2:
        result['comparison'] = compare(*args.paths)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for name, run in result['runs'].items():
        print(name, json.dumps({key: value for key, value in run.items()
                               if key not in ('linearizations',)}, indent=2))
