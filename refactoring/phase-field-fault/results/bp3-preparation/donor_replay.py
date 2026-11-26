"""Compare the eight accepted native observations after donor checkpoint 4229.

Use this same-writer replay as a strict reproducibility screen. A difference is
reported before spending a longer allocation. A changed build or MPI reduction
may require a separate physical comparison of the difference.
"""
import argparse
import csv
from decimal import Decimal
from pathlib import Path
import re


STEPS = range(4230, 4238)


def observations(path):
    result = {step: {} for step in STEPS}
    with path.open() as stream:
        for line in stream:
            if 'Work consistency trace:' in line:
                tokens = dict(re.findall(r'([A-Za-z_]+)=([^,\s]+)', line))
                step = int(tokens['step'])
                if step in result:
                    names = ('time_s', 'dt_s', 'matching_shear_Pa', 'matching_raw_normal_Pa',
                             'bound_Pa', 'guard_uses_matching_frame')
                    result[step]['work'] = {name: Decimal(tokens[name]) for name in names}
            elif 'BP3 7B accepted check:' in line:
                match = re.search(r'step=(\d+), stress error=([^,]+), allowance fraction=([^,]+), '
                                  r'stress scale=([^ ]+) Pa\.', line)
                if match and int(match[1]) in result:
                    result[int(match[1])]['stress'] = dict(zip(
                        ('stress_error', 'allowance_fraction', 'stress_scale_Pa'), map(Decimal, match.groups()[1:])))
            else:
                accepted = re.search(r'BP3 accepted state\s+(\d+)\s*$', line)
                if accepted and int(accepted[1]) in result:
                    frame = result[int(accepted[1])]
                    if 'accepted' in frame:
                        raise ValueError(f'{path}: repeated accepted label {accepted[1]}')
                    frame['accepted'] = True
    for step, frame in result.items():
        if set(frame) != {'work', 'stress', 'accepted'}:
            raise ValueError(f'{path}: missing accepted observation at {step}; found {sorted(frame)}')
        work, stress = frame['work'], frame['stress']
        if (not all(value.is_finite() for value in (*work.values(), *stress.values()))
                or work['dt_s'] <= 0 or work['guard_uses_matching_frame'] != 1
                or work['bound_Pa'] != Decimal('1e-5')
                or any(not 0 <= work[key] <= work['bound_Pa']
                       for key in ('matching_shear_Pa', 'matching_raw_normal_Pa'))
                or not 0 <= stress['allowance_fraction'] <= 1 or stress['stress_scale_Pa'] <= 0):
            raise ValueError(f'{path}: invalid stress or time at accepted step {step}')
    return result


def compare(reference, replay):
    old, new = observations(reference), observations(replay)
    for step in STEPS:
        for group in ('work', 'stress'):
            for name, expected in old[step][group].items():
                actual = new[step][group][name]
                if actual != expected:
                    raise ValueError(f'accepted {step} {name}: archived {expected}, replay {actual}, '
                                     f'difference {actual-expected}')
    print(f'All {len(STEPS)} accepted clocks, durations, paired-work diagnostics and independent '
          f'stress checks agree exactly. Final stress scale {new[4237]["stress"]["stress_scale_Pa"]} Pa.')


def check_tables(reference, replay):
    for name in ('accepted_steps.csv', 'stations.csv'):
        records = []
        for directory in (reference, replay):
            by_id = {}
            with (directory / name).open() as stream:
                for row in csv.DictReader(stream):
                    step = int(row['step'])
                    if step in STEPS:
                        key = (step, Decimal(row['xd'])) if name == 'stations.csv' else (step,)
                        values = {column: Decimal(value) for column, value in row.items()}
                        if key in by_id or not all(value.is_finite() for value in values.values()):
                            raise ValueError(f'{directory}/{name}: duplicate or nonfinite record at {key}')
                        by_id[key] = values
            if {key[0] for key in by_id} != set(STEPS):
                raise ValueError(f'{directory}/{name}: incomplete eight-step replay')
            records.append(by_id)
        old, new = records
        if new.keys() != old.keys():
            raise ValueError(f'{name}: station or step IDs differ')
        for key, archived in old.items():
            if new[key].keys() != archived.keys():
                raise ValueError(f'{name}: column names differ at {key}')
            for column, expected in archived.items():
                if new[key][column] != expected:
                    raise ValueError(f'{name} {key} {column}: archived {expected}, replay {new[key][column]}')
        print(f'{name}: {len(old)} rows agree in every column')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('replay', type=Path)
    parser.add_argument('--tables', nargs=2, type=Path, metavar=('PARENT', 'OUTPUT'),
                        help='also compare both native CSV series through accepted step 4237')
    args = parser.parse_args()
    try:
        compare(args.archive, args.replay)
        if args.tables:
            check_tables(*args.tables)
    except (OSError, ValueError, KeyError, ArithmeticError) as exc:
        parser.exit(1, f'{exc}\n')
