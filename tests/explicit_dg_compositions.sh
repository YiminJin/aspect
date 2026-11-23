#!/bin/sh
set -eu
cat
python3 -S - <<'PY'
from pathlib import Path
out = Path("output-explicit_dg_compositions")
h = .0125
profile = lambda x: (x-.4)**2-.2
state = [profile(.5*k+.25*i) for k in range(2) for i in range(3)]
# Strong P2 material derivative plus left-face jump lifting, evaluated at
# the three Lobatto nodes of each half-cell. The velocity is +1 m/s.
def rhs(values, time):
    result = []
    left = profile(-time)
    for cell in range(2):
        a,b,c = values[3*cell:3*cell+3]
        jump = left-a
        result.extend(2*(lift*jump-deriv) for lift,deriv in
                      zip((9,-1.5,3),(-3*a+4*b-c,c-a,a-4*b+3*c)))
        left=c
    return result

for step in range(3):
    if step:
        t=(step-1)*h
        first=[a+h*f for a,f in zip(state,rhs(state,t))]
        second=[(3*a+b+h*f)/4 for a,b,f in zip(state,first,rhs(first,t+h))]
        state=[(a+2*b+2*h*f)/3 for a,b,f in zip(state,second,rhs(second,t+h/2))]
    lines=(out/f'coefficients-{step}-0').read_text().splitlines()
    if list(map(float,lines[0].split())) != [step*h,h if step else 0.] or len(lines)!=5:
        raise ValueError("missing mixed fields or wrong clock")
    for line in lines[1:]:
        cell,name,basis,x0,y0,x1,y1,*values=line.split()
        k=int(2*float(x0))
        errors=[abs(float(v)-state[3*k+i%3]) for i,v in enumerate(values)]
        if name=="tracer" and not all(e<2.e-12 for e in errors):
            raise ValueError(f"wrong explicit action on {cell}: {errors}")
        if step==1 and name=="companion" and not max(errors)>5.e-5:
            raise ValueError("implicit companion lost its distinct trajectory")
(out/'mixed-check').write_text('Explicit tracer and implicit companion verified.\n')
PY
