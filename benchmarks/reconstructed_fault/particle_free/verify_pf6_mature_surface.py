#!/usr/bin/env python3
"""Independent regularized-RSF reference for the intended mature BP3 initialization.

Surface fractions are prescribed; Q1 interpolation of exponential initial state
has a representation defect. Report that defect instead of correcting prestress.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np


def read(path):
    rows=list(csv.DictReader(path.open()))
    return {k:np.array([float(row[k]) for row in rows]) for k in rows[0]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    V,Vref,Dc,b,mu0,sigma,damping=1e-9,1e-6,.008,.015,.6,50e6,4624440.
    mu_target=.025*np.arcsinh(V/(2*Vref)*np.exp((mu0+b*np.log(Vref/V))/.025))
    def theta(fraction):
        direct=.01+.015*fraction
        return Dc/Vref*np.exp((direct*np.log(2*Vref/V*np.sinh(mu_target/direct))-mu0)/b)
    def friction(fraction,state):
        direct=.01+.015*fraction
        return direct*np.arcsinh(V/(2*Vref)*np.exp((mu0+b*np.log(Vref*state/Dc))/direct))
    result={}
    for run in a.runs:
        nodes=read(run/'surface-initialization.csv');qp=read(run/'surface-initialization-interior.csv')
        expected=np.clip((nodes['s_m']-15000.)/3000.,0,1)
        f_error=float(max(abs(nodes['fraction']-expected)))
        theta_error=float(max(abs(nodes['theta_s']/theta(expected)-1)))
        mu_error=float(max(abs(friction(qp['fraction'],qp['theta_s'])-qp['friction'])))
        assert f_error<1e-13 and theta_error<2e-13 and mu_error<1e-14
        assert max(abs(nodes['a']-(.01+.015*expected)))<1e-14
        assert min(nodes['I_h_m'])>0
        result[str(run)]=dict(nodes=len(nodes['node']),native_GP_mu_independent_Linf=mu_error,
            nodal_fraction_Linf=f_error,nodal_theta_independent_relative_Linf=theta_error,
            fraction_interpolation_Linf=float(max(abs(qp['fraction']-qp['fraction_analytic']))),
            theta_GP_relative_Linf=float(max(abs(qp['theta_s']/theta(qp['fraction_analytic'])-1))),
            nodal_prestress_balance_Linf_Pa=float(max(abs(nodes['balance_Pa']))),
            GP_prestress_balance_Linf_Pa=float(max(abs(qp['balance_Pa']))),
            min_a=float(min(nodes['a'])),max_a=float(max(nodes['a'])),
            min_I_h_m=float(min(nodes['I_h_m'])),max_I_h_m=float(max(nodes['I_h_m'])))
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
