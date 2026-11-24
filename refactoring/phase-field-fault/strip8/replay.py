"""Prepare common point and moment decks and independently evaluate the full-strip objective."""
import argparse
import csv
from math import cos, fsum, sin
from pathlib import Path

import mpmath as mp

CASES=('D2','T2','D1','T1','B2','B1')


def points(path):
    table={}
    for file in sorted(path.glob('points-*')):
        for line in file.read_text().splitlines():
            cell,q,*values=line.split()
            key=(cell,int(q))
            if key in table: raise ValueError(f'Duplicate point {key}')
            table[key]=tuple(map(float,values))
    if len(table)!=180: raise ValueError(f'{path}: expected all 180 strip-8 points')
    return dict(sorted(table.items()))


def moments(points):
    terms=[]
    for area,kernel,F,u,v,*_ in points.values():
        w=area*(1-F)**2*kernel
        terms.append((area*F*u,area*F*v,area*F,w*u*u,w*v*v,w*u*v,w*u,w*v,w))
    raw=[fsum(t) for t in zip(*terms)]
    mu,mv=raw[6]/raw[8],raw[7]/raw[8]
    centered=[]
    for (_,_,_,u,v,*_),t in zip(points.values(),terms):
        centered.append((t[8]*(u-mu)**2,t[8]*(u-mu)*(v-mv),t[8]*(v-mv)**2))
    return raw+[fsum(t) for t in zip(*centered)]


def prepare(work):
    dest=work/'inputs'
    dest.mkdir(exist_ok=True)
    keys=points(work/'D2').keys()
    for case in CASES:
        frame=work/case
        data=points(frame)
        if data.keys()!=keys: raise ValueError(f'{case}: changed point identities')
        with (dest/('point-'+case)).open('w') as f:
            for (cell,q),values in data.items():
                print(cell,q,*values[:5],file=f)
        (dest/('ordered-'+case)).write_text(' '.join(map(repr,moments(data))))
        live=[v for n in (81,27) for v in (frame/f'live-{n}').read_text().split()]
        if len(live)!=12: raise ValueError('Missing deployed reduction')
        (dest/('live-'+case)).write_text(' '.join(live))


# Independently solve the scalar rate equation using arbitrary precision
# and evaluate the sum of squared point tractions directly. Never use the
# centered variance or subtract two O(stress^2) terms in this reference.
class Direct:
    def __init__(self,data,width):
        self.p=[list(map(mp.mpf,v[:5])) for v in data.values()]
        self.width=mp.mpf(float(width))
        self.raw=(mp.fsum(a*F*u for a,_,F,u,_ in self.p),
                  mp.fsum(a*F*v for a,_,F,_,v in self.p),
                  mp.fsum(a*F for a,_,F,_,_ in self.p))

    def value(self,angle):
        h=mp.mpf(.0025)
        G=mp.mpf(1e6)
        B=1/(1+h*G/mp.mpf(1e8))
        s,c=mp.sin(2*angle),mp.cos(2*angle)
        scale=G*B*h*self.width/mp.cos(angle)
        free=(self.raw[0]*s+self.raw[1]*c)/scale
        mobility=self.raw[2]/scale
        def reaction(speed):
            k=abs(speed)/mp.mpf(.01)
            age=10*mp.exp(-k*h)-mp.expm1(-k*h)/k if k else 10+h
            a,b,V0=map(mp.mpf,(.015,.02,.001))
            return 200*a*mp.asinh(speed/(2*V0)*mp.exp((mp.mpf(.6)+b*mp.log(age*V0/mp.mpf(.01)))/a))+1000*speed
        # At this stage the actual creep speed is about 1.5e-13 m/s.
        velocity=mp.findroot(lambda v:v+mobility*reaction(v)-free,(mp.mpf('1e-13'),mp.mpf('2e-13')),tol=mp.mpf('1e-52'))
        f=reaction(velocity)
        score=mp.fsum(a*k*(1-F)**2*(u*s+v*c-f)**2 for a,k,F,u,v in self.p)
        return f,score


def analyze(work,out):
    mp.mp.dps=65
    angles=list(map(float,(work/'angles').read_text().split()))
    with out.open('w') as file:
        w=csv.writer(file)
        w.writerow('source selector input angle force_Pa score_Pa2_m2 Vfree mobility direct_at_selector_force high_precision_force high_precision_score'.split())
        for case in CASES:
            data=points(work/case)
            ref=Direct(data,'0.2480056452854309')
            refs={a:ref.value(mp.mpf(a)) for a in angles}
            for selector in ('donor','target'):
                for kind in ('point','live','ordered'):
                    path=work/('replay-'+selector)/f'scores-{kind}-{case}'
                    observed=[list(map(float,l.split())) for l in path.read_text().splitlines()]
                    if [r[0] for r in observed[:-1]]!=angles or len(observed)!=len(angles)+1:
                        raise ValueError(f'Incomplete scores in {path}')
                    for a,f,score,free,mobility in observed:
                        fs,ss=refs[a] if a in refs else ref.value(mp.mpf(a))
                        s,c=sin(2*a),cos(2*a)
                        direct=fsum(J*rho*(1-F)**2*((u*s+v*c)-f)**2 for J,rho,F,u,v,*_ in data.values())
                        w.writerow([case,selector,kind,a,f,score,free,mobility,direct,mp.nstr(fs,24),mp.nstr(ss,32)])


def stage_controls(work, out):
    mp.mp.dps=65
    donor,target=(points(work/c) for c in ('D2','T2'))
    incoming,donor_max,target_max,*_=map(float,(work/'angles').read_text().split())
    with out.open('w') as file:
        w=csv.writer(file)
        w.writerow(('target_columns_substituted','angle','force_Pa','score_Pa2_m2'))
        for name,cols in [('none',()),('area and kernel',(0,1)),('phase fraction',(2,)),
                          ('predictor u and v',(3,4)),('all',range(5))]:
            data={k:tuple(target[k][i] if i in cols else v for i,v in enumerate(row))
                  for k,row in donor.items()}
            ref=Direct(data,'0.2480056452854309')
            for a in (incoming,donor_max,target_max):
                force,score=ref.value(mp.mpf(a))
                w.writerow((name,a,mp.nstr(force,24),mp.nstr(score,32)))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('work',type=Path)
    ap.add_argument('--analyze',type=Path)
    ap.add_argument('--stage-controls',type=Path)
    args=ap.parse_args()
    if args.analyze: analyze(args.work,args.analyze)
    elif args.stage_controls: stage_controls(args.work,args.stage_controls)
    else: prepare(args.work)
