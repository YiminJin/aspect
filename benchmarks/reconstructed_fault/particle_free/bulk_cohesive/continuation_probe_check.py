#!/usr/bin/env python3
"""High-precision local explanation of the retained predictor FD failure.

Reconstruct an equivalent local (t0,abar,phi) from an exported sticking point
and its first sliding probe. This is a local derivative control, not a new
history or an interpolated constitutive state for the native trajectory.
"""
import json
from pathlib import Path
import numpy as np
import mpmath as mp
from continuation_analysis import BASE,OUT
mp.mp.dps=70
row=np.genfromtxt(BASE/'runs/predictor-probe/bulk-tangent-switches-rank-1.csv',delimiter=',',names=True,dtype=None,encoding='utf8')[-1]
x,y=float(row['x']),float(row['y']);eps=float(row['epsilon']);phi=mp.mpf(str(row['phi0']));q0=mp.mpf(str(row['q0']));dt=mp.mpf('.00125');G=mp.mpf('1000000');beta=1/(1+dt/100);R=mp.mpf('1000');dx=.015625;dy=.0078125
ix=int(x/dx);iy=int(y/dy);u=(x-ix*dx)/dx;v=(y-iy*dy)/dy
L=lambda z:np.array([2*(z-.5)*(z-1),4*z*(1-z),2*z*(z-.5)])
dL=lambda z:np.array([4*z-3,4-8*z,4*z-1])
xx=ix*dx+dx*np.array([0,.5,1]);yy=iy*dy+dy*np.array([0,.5,1]);X,Y=np.meshgrid(xx,yy)
vx=.001*np.sin(np.pi*X)*np.sin(np.pi*Y)*(X-.5);vy=.001*np.sin(np.pi*X)*np.sin(np.pi*Y)*(Y-.5)
dgamma=float(dL(v)@vx@L(u)/dy+L(v)@vy@dL(u)/dx)
dphi=float((1-u)*(.01+.005*np.sin(np.pi*ix*dx)**2)+u*(.01+.005*np.sin(np.pi*(ix+1)*dx)**2))
def compliance(z):return 4*z*(1+z)/(1-z)**2
h0=compliance(phi);a0=h0*q0/G;dp=mp.mpf(str(dphi));dg=mp.mpf(str(dgamma));e=mp.mpf(str(eps));w1=mp.mpf(str(row['w1']));h1=compliance(phi+e*dp);a1=a0+beta*dt*w1
T0=R+(1+h1)*G*a1/h1-e*G*beta*dg

def response(e):
    ph=phi+e*dp;h=compliance(ph);trial=T0+e*G*beta*dg
    sliding=abs(h*trial-G*(1+h)*a0)>R*h
    q=(trial-R)/(1+h) if sliding else G*a0/h
    hp=4*(1+3*ph)/(1-ph)**3
    return hp*q*q/(2*G),sliding
Y0,s0=response(mp.mpf(0));hp=4*(1+3*phi)/(1-phi)**3;hpp=24*(1+phi)/(1-phi)**4
analytic=(hpp-2*hp*hp/h0)*q0*q0/(2*G)*dp
rows=[]
for value in ['1e-7','5e-8','1e-9','1e-10','1e-11']:
    ep=mp.mpf(value);Y1,s1=response(ep);Y2,s2=response(2*ep);fd=(4*Y1-Y2-3*Y0)/(2*ep)
    rows.append(dict(epsilon=float(ep),sliding0=bool(s0),sliding1=bool(s1),sliding2=bool(s2),relative_derivative_error=float(abs(fd-analytic)/max(1,abs(analytic)))))
result=dict(cell=str(row['cell']),qp=int(row['qp']),x_m=x,y_m=y,phi=float(phi),q_Pa=float(q0),trial_shear_Pa=float(T0),native_phase_direction=dphi,native_engineering_shear_direction=float(dg),sticking_margin_Pa=float(R-(T0-G*(1+1/h0)*a0)),analytic_driving_direction_Pa=float(analytic),probes=rows,
interpretation='Equivalent local control: the fixed large probes cross the actual Coulomb branch. Smaller probes stay on the sticking branch and converge to its analytic derivative. This does not retrospectively pass the failed native global probe.')
assert not s0 and rows[-1]['relative_derivative_error']<1e-8
(OUT/'predictor-local-probe-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
