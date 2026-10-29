#!/usr/bin/env python3
"""Compare the distributed symbol with a derived closed-interface symbol.

Identical incompressible Maxwell halfspaces: K(s)=G*|k|*s/(s+r).
This is a different geometry, used as a mature-interface limit diagnostic.
"""
import json
import numpy as np
from structural_mixed import base,G,RELAX,DC,P0
from structural_sliding import OUT,write_csv


def active_roots(b,K,freeze_state=False):
    Z=b['Z'];nu=b['V']/DC;F=0. if freeze_state else P0*b['mt']*b['Theta']/DC
    return np.roots([Z,Z*(RELAX+nu)+K-F,Z*RELAX*nu+K*nu-F*RELAX])


def main():
    b=base(1.,.0005);angle=.04;sn=np.sin(2*angle);cs=np.cos(2*angle)
    D=sn*sn-b['mu']*sn*(1+cs);Kbulk=G*b['chi']*D
    rows=[]
    for wavelength in np.geomspace(1e-5,1e4,121):
        for mode,K in [('bulk_oblique',Kbulk),('closed_interface',2*np.pi*G/wavelength)]:
            eig=active_roots(b,K)
            rows.append(dict(mode=mode,wavelength_m=wavelength,growth_s_inv=max(eig.real),K_Pa_m=K,V_m_s=b['V'],angle_rad=angle))
    angular=[]
    for angle in np.linspace(0,np.pi,361):
        sn=np.sin(2*angle);cs=np.cos(2*angle)
        for mode,pressure,state in [('full',True,True),('fixed_age_control',True,False),('fixed_pn_control',False,True),('both_fixed_control',False,False)]:
            D=sn*sn-(b['mu']*sn*(1+cs) if pressure else 0.)
            growth=max(active_roots(b,G*b['chi']*D,freeze_state=not state).real)
            angular.append(dict(mode=mode,angle_deg=np.degrees(angle),growth_s_inv=growth))
    write_csv('interface_spectrum.csv',rows);write_csv('pressure_state_controls.csv',angular)
    Z=b['Z'];nu=b['V']/DC;F=P0*b['mt']*b['Theta']/DC
    critical=max(F-Z*(RELAX+nu),(F-Z*nu)*RELAX/nu)
    (OUT/'interface_spectrum_summary.json').write_text(json.dumps(dict(bulk_oblique_growth_s_inv=max(active_roots(b,Kbulk).real),interface_critical_wavelength_m=2*np.pi*G/critical,
        Kbulk=Kbulk,fixed_age_max_growth_s_inv=max(r['growth_s_inv'] for r in angular if r['mode']=='fixed_age_control')),indent=2)+'\n')

if __name__=='__main__':main()
