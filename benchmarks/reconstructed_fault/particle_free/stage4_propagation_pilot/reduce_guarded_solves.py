"""Count returned corrections and verify guard decisions in a native trial log."""
import argparse
import json
import re
from pathlib import Path


def summarize(body):
    costs = {}
    for prefix in ('Tip bound linear solve', 'Tip shifted mechanics solve', 'Tip shifted phase CG'):
        measurements = []
        for line in body.splitlines():
            if prefix + ':' in line:
                data = dict(re.findall(r'(\w+)=([^, ]+)', line))
                measurements.append({
                    'estimate': float(data.get('residual', data.get('estimate'))),
                    'explicit': float(data['explicit']), 'tolerance': float(data['tolerance']),
                    'iterations': int(data['iterations'])
                })
        costs[prefix] = dict(count=len(measurements), failed_certificates=sum(
            not m['explicit'] <= m['tolerance'] for m in measurements),
            iterations=sum(m['iterations'] for m in measurements))
    # A rejected exact correction must leave this VI loop before pass p+1.
    previous_failed = False
    for line in body.splitlines():
        if 'Tip linearized VI:' in line:
            assert not previous_failed, line
        if 'Tip bound linear solve:' in line:
            data = dict(re.findall(r'(\w+)=([^, ]+)', line))
            previous_failed = not float(data['explicit']) <= float(data['tolerance'])
        if any(marker in line for marker in ('Tip shifted phase CG:', 'Bulk tip coupled Newton ')):
            previous_failed = False
    costs['interior_trials'] = body.count('alpha=0.98999999999999999, merit=')
    costs['interior_accepted'] = body.count('accepted search: shift=0, alpha=0.98999999999999999')
    return costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    chunks = re.split(r'\*\*\* Timestep (\d+):', (args.run/'run.log').read_text())
    result = {step: summarize(body) for step, body in zip(chunks[1::2], chunks[2::2])}
    args.destination.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
