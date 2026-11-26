"""Locate the first differing golden-section comparison and check its sign on the frozen field."""
import argparse
from pathlib import Path
import mpmath as mp
from peaks import PointScore, captured_double, read_rows, write_table


def compare(directory):
    objective = PointScore(directory)
    search = read_rows(directory/'turn-search.csv')
    direct = {}

    def value(row, key):
        beta = captured_double(row[key])
        if beta not in direct:
            direct[beta] = objective.value(beta)[2]
        return direct[beta]

    paths = []
    for policy in ('0','1'):
        path = [row for row in search if row['policy'] == policy]
        for row in path:
            if bool(int(row['left_smaller'])) != (captured_double(row['fl']) < captured_double(row['fr'])):
                raise ValueError('Wrong recorded golden comparison')
            for key in ('left','right'):
                value(row, key)
        paths.append(path)
    for iteration, (p,c) in enumerate(zip(*paths,strict=True), start=1):
        if any(p[key] != c[key] for key in ('lo','hi','left','right')):
            raise ValueError('Intervals differ before the first differing decision')
        if p['left_smaller'] != c['left_smaller']:
            a,b = value(p,'left'),value(p,'right')
            result = []
            for name, left, right in [('production',captured_double(p['fl']),captured_double(p['fr'])),
                                      ('candidate',captured_double(c['fl']),captured_double(c['fr'])),
                                      ('independent',a,b)]:
                result.append(dict(comparison=iteration,method=name,lo_rad=p['lo'],hi_rad=p['hi'],
                                   left_rad=p['left'],right_rad=p['right'],left_score_Pa2_m2=left,
                                   right_score_Pa2_m2=right,right_minus_left_Pa2_m2=right-left,
                                   keep='upper' if right>left else 'lower'))
            return result
    raise ValueError('The two methods made the same comparisons')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    ap.add_argument('output',type=Path)
    args=ap.parse_args()
    mp.mp.dps=65
    rows=compare(args.directory)
    write_table(args.output, rows)
    for row in rows:
        print(row['comparison'],row['method'],mp.nstr(row['right_minus_left_Pa2_m2'],15),row['keep'])
