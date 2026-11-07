#!/usr/bin/env python3
"""Locate measured representation differences using completed paired endpoints."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
import numpy as np
from compare_bounded_profile import load,quadrature,tensor,dot3,ROOT
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/direct_true_stress'
result={}
for a,b in [('true-c-p','true-sep-c-p-2'),('true-c-p-ref','bpf-c-p-m-ref'),('true-l-p','true-sep-l-p'),('true-c-n','bpf-c-n-m-joint')]:
    sa=load(a,True)[3];sb=load(b,True)[3];sa=sa[max(sa)];sb=sb[max(sb)]
    x,w=quadrature(sa.n,8);d=tensor(sa,x)-tensor(sb,x);err=dot3(d,d);ref=dot3(tensor(sb,x),tensor(sb,x))
    X=sb.surface[:,1:3];t=X[-1]-X[0];length=np.linalg.norm(t);t/=length;n=np.array([-t[1],t[0]])
    tangential=(x-X[0])@t;normal=abs((x-X[0])@n)
    ends=(np.minimum(abs(tangential),abs(tangential-length))<2*sb.h)&(normal<.6)
    center=(tangential>.3)&(tangential<length-.3)&(normal<.4)
    result[a+':'+b]=dict(end_band_width=2*sb.h,endpoint_fraction_squared_error=float((w[ends]@err[ends])/(w@err)),
       central_fault_band_relative_L2=float(np.sqrt((w[center]@err[center])/(w[center]@ref[center]))),
       central_fault_band_absolute_L2=float(np.sqrt(w[center]@err[center])),
       interpretation='Location diagnostic on evolved trajectories; not the common-frozen-state projection identity and not proof of a unique error cause.')
(OUT/'endpoint-error-location.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
