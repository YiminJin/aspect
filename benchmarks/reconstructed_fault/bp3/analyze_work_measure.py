"""Summarize the single work-measure qualification without launching ASPECT."""
import csv
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
RUN = HERE / 'work-measure-free-top-local4'


def main():
    text = (RUN / 'run.log').read_text().replace('\0', '')
    checks = {key: float(value) for key, value in
              next(csv.DictReader((RUN / 'work_measure_checks.csv').open())).items()}
    rows = [{key: float(value) for key, value in row.items()}
            for row in csv.DictReader((RUN / 'noncommitting_surface.csv').open())]
    initial = list(csv.DictReader((RUN / 'work_initial_rows.csv').open()))
    for i, row in enumerate(rows):
        measure = row['mass_diagonal'] + row['mass_upper']
        if i:
            measure += rows[i-1]['mass_upper']
        row['measure_m'] = measure
        for field in ('q', 'sigma', 'C', 'R'):
            row[field + '_Pa'] = row['weak_' + field] / measure
    free = [row for row in rows if not row['prescribed']]
    linear = []
    for match in re.finditer(r'Fault linear solve: iterations=(\d+), estimated=([^,]+), fresh=([^,]+), target=([^,]+)', text):
        iterations, estimated, fresh, target = match.groups()
        linear.append(dict(iterations=int(iterations), estimated=float(estimated),
                           fresh=float(fresh), target=float(target)))
    assert linear and all(row['fresh'] <= row['target'] for row in linear)
    nonlinear = []
    for line in text.splitlines():
        if 'Fault nonlinear residual:' in line:
            nonlinear.append({key.strip(): float(value) for key, value in
                re.findall(r'([a-z ]+)=([-+0-9.eE]+)', line.split(':', 1)[1])})
    final = re.search(r'Noncommitting fault diagnostic converged: bulk=([^,]+), surface=([^;]+)', text)
    assert final and all(float(v) < 1e-8 for v in final.groups())
    assert 'BP3 noncommitting rollback verified:' in text
    assert checks['K_relative'] < 2e-7 and checks['G_relative'] < 2e-7
    assert checks['shear_work_relative'] < 2e-10
    assert checks['affine_load_error_Pa'] < 1e-6
    assert checks['owned_qps'] == 42880*9
    assert all(row['V'] == 1e-9 for row in rows if row['prescribed'])
    result = dict(execution=json.loads((RUN / 'execution.json').read_text()),
                  checks=checks, linear=linear, nonlinear=nonlinear,
                  final_relative=dict(bulk=float(final[1]), surface=float(final[2])),
                  nodes=len(rows), free_nodes=len(free),
                  lower_active=sum(int(row['lower_active']) for row in free),
                  top=rows[-1], top_initial_residual_Pa=float(initial[-1]['residual_Pa']),
                  free_V_range=[min(row['V'] for row in free), max(row['V'] for row in free)],
                  max_free_row_residual_Pa=max(abs(row['R_Pa']) for row in free),
                  free_weak_normal_range_Pa=[min(row['sigma_Pa'] for row in free),
                                            max(row['sigma_Pa'] for row in free)])
    (RUN / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
