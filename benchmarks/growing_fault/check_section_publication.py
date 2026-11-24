"""Independent first-step cold-creep audit from the published native histories."""
import argparse
from math import asinh, copysign, exp, expm1, fsum, log, ulp
from pathlib import Path
import sys

from compare import rows


def advance_age(speed, parent, duration):
    x = abs(speed)*duration/.01
    return parent*exp(-x) + duration*(-expm1(-x)/x if x else 1.)


def resistance(rate, parent, half_step, coefficient):
    theta = advance_age(rate, parent, half_step)
    z = abs(rate)/.002 * exp((coefficient+.02*log(theta*.001/.01))/.015)
    return copysign(3.*asinh(z), rate)+1000.*rate


def check(output, coefficient=.9):
    def points(step):
        return rows(output.glob(f'points-{step:04}-r*.csv'), ('cell', 'q'))
    def strips(step):
        return rows([output/f'sections-{step:04}.csv'], ('section',))
    old, new = points(0), points(1)
    if old.keys()!=new.keys():
        raise ValueError('Changed native point identities')
    incoming, accepted = strips(0), strips(1)
    book = rows([output/'balance-0001.csv'], ('step',))[1,]
    dt = float(book['time'])
    work = []
    native_work = []
    work_budget = []
    errors = []
    for k, section in accepted.items():
        rate, force, length = (float(section[n]) for n in ('V','force','ds'))
        parent_age, parent_slip = (float(incoming[k][n]) for n in ('age','slip'))
        terms = []
        scale = []
        for key in sorted(new):
            if int(new[key]['section']) == k[0]:
                weight = float(new[key]['weight'])/(dt*length)
                after, before = (float(tab[key]['signed_shear']) for tab in (new,old))
                terms.append(weight*(after-before))
                scale.append(weight*(abs(after)+abs(before)))
        recovered = fsum(terms)
        # Binary64 point storage, multiply/divide by physical measures, and
        # the rounded local update. Conditioning is set by the sum of absolute
        # inventories divided by dt, even when their signed sum nearly vanishes.
        bound = 32.*sys.float_info.epsilon*fsum(scale)
        native_work.append(dt*length*force*recovered)
        work_budget.append(dt*length*abs(force)*bound)
        expected = resistance(rate,parent_age,dt/2,coefficient)
        native_force = resistance(recovered,parent_age,dt/2,coefficient)
        work.append(dt*length*rate*force)
        if abs(expected-force)>1.e-11 or work[-1]<0.:
            raise ValueError(f'Strip {k[0]}: published V={rate:.17g}, f={force:.17g}, signed RSF={expected:.17g} Pa; work={work[-1]:.12g} J/m')
        if abs(recovered-rate)>bound or abs(native_force-force)>.01:
            raise ValueError(f'Strip {k[0]}: integrated rate defect {recovered-rate}, budget {bound}, force defect {native_force-force} Pa')
        slip = parent_slip+dt*rate
        age = advance_age(rate,parent_age,dt)
        if abs(float(section['slip'])-slip)>2*ulp(slip) or abs(float(section['age'])-age)>3*ulp(age):
            raise ValueError(f'Strip {k[0]}: inconsistent endpoint slip or age')
        errors.append((abs(rate-recovered),bound,abs(force-expected),abs(force-native_force)))
        print(f'{k[0]}: V={rate:.12g}, native V={recovered:.12g} m/s, f={force:.12g} Pa, work={work[-1]:.12g} J/m')
    total=fsum(work)
    if abs(float(book['Df'])-total)>16*ulp(total):
        raise ValueError('Section work differs from the published friction loss')
    print('Max absolute native rate defect, rate budget, signed RSF defect, native RSF defect:',*[max(t) for t in zip(*errors)])
    if abs(fsum(native_work)-total)>fsum(work_budget) or abs(float(book['adjoint_error']))>fsum(work_budget):
        raise ValueError('Native point and section work differ beyond the transfer roundoff budget')
    print('Friction work',total,'J/m; signed work-adjoint defect',book['adjoint_error'],'J/m')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    check(parser.parse_args().output)
