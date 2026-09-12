"""Offline decomposition of a saved K3 localization-normalization defect."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def read(path, name, step):
    return np.atleast_1d(np.genfromtxt(path/f'{name}_{step}.csv', delimiter=',', names=True))


def phase_interpolator(path, step):
    data = read(path, 'phase', step)
    xs, ys = np.unique(data['x']), np.unique(data['y'])
    values = np.empty((len(xs), len(ys)))
    values[np.searchsorted(xs, data['x']), np.searchsorted(ys, data['y'])] = data['phi']
    return RegularGridInterpolator((xs, ys), values, bounds_error=True), ys


def h(phi):
    phi = np.maximum(phi, 0.)
    return 128*phi*(1+phi)/(1-phi)**2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    args = parser.parse_args()
    path = args.case
    gauss, weights = np.polynomial.legendre.leggauss(12)
    summaries = []
    for step in range(len(list(path.glob('time_*.csv')))):
        bulk, surface = read(path, 'bulk', step), read(path, 'surface', step)
        previous = read(path, 'surface', max(0, step-1))
        segments = read(path, 'segments', step)
        current_phi, ys = phase_interpolator(path, step)
        previous_phi, previous_ys = phase_interpolator(path, max(0, step-1))
        dt = float(read(path, 'time', step)[0]['dt'])
        beta = np.exp(-dt/100)
        kappa = -1e8*np.expm1(-dt/100)
        rows = []
        for x in np.unique(bulk['x']):
            column = bulk[bulk['x']==x]
            V, I = (float(np.interp(x, surface['x'], surface[key])) for key in ('V','Ih'))
            Ip, Cp = (float(np.interp(x, previous['x'], previous[key])) for key in ('Ih','C'))
            factor = beta*Cp/kappa
            scale = max(abs(V),1e-5)
            segment = np.clip(np.searchsorted(surface['x'],x,side='right')-1,0,len(segments)-1)
            center = float(np.interp(x, surface['x'],surface['y']))
            lower = center-segments['half_width_minus'][segment]
            upper = center+segments['half_width_plus'][segment]
            edges = np.unique(np.r_[ys,previous_ys,lower,upper])
            y = edges[:-1,None]+np.diff(edges)[:,None]*(1+gauss)/2
            points = np.c_[np.full(y.size,x),y.ravel()]
            hc = h(current_phi(points)).reshape(y.shape)
            hp = h(previous_phi(points)).reshape(y.shape)
            w = np.diff(edges)[:,None]*weights/2
            admitted = (y>=lower)&(y<=upper)
            instantaneous, history = hc*V/I, factor*(hc*Ip/I-hp)
            full_inst, full_hist = float(np.sum(w*instantaneous)), float(np.sum(w*history))
            supported_inst = float(np.sum(w*instantaneous*admitted))
            supported_hist = float(np.sum(w*history*admitted))
            actual_w = column['weight']/np.sum(column['weight'])
            actual_inst = float(np.dot(actual_w,column['chi']*column['V']))
            actual_hist = float(np.dot(actual_w,column['history']))
            active = column['active']>0
            qp = np.c_[np.full(len(column),x),column['y']]
            hc_qp, hp_qp = h(current_phi(qp)), h(previous_phi(qp))
            # Confirm the postprocessor's old FE phase is the saved previous
            # profile before attributing an identity defect to its projection.
            qp_history_error = float(max(abs(column['history'][active]
                -factor*(hc_qp[active]*Ip/I-hp_qp[active]))))
            qp_chi_error = float(max(abs(column['chi'][active]-hc_qp[active]/I)))
            full_defect = (V-full_inst-full_hist)/scale
            tail = (full_inst+full_hist-supported_inst-supported_hist)/scale
            quadrature = (supported_inst+supported_hist-actual_inst-actual_hist)/scale
            actual = (V-actual_inst-actual_hist)/scale
            e = float(np.sum(w*hc))/I-1
            previous_e = float(np.sum(w*hp))/Ip-1
            amplification = factor*Ip/V
            predicted = e+amplification*(e-previous_e)
            measured = (full_inst+full_hist-V)/V
            row = dict(step=step,x=x,V=V,Ih_projected=I,
                Ih_independent=float(np.sum(w*hc)),previous_Ih_projected=Ip,
                previous_Ih_independent=float(np.sum(w*hp)),
                relative_Ih_error=e,previous_relative_Ih_error=previous_e,
                history_amplification=amplification,
                predicted_full_integral_minus_V_over_V=predicted,
                measured_full_integral_minus_V_over_V=measured,
                actual_signed_defect=actual,full_signed_defect=full_defect,
                omitted_signed_tail=tail,bulk_quadrature_signed_defect=quadrature,
                full_instantaneous_m_per_s=full_inst,full_history_m_per_s=full_hist,
                supported_instantaneous_m_per_s=supported_inst,supported_history_m_per_s=supported_hist,
                actual_instantaneous_m_per_s=actual_inst,actual_history_m_per_s=actual_hist,
                qp_history_max_difference=qp_history_error,qp_chi_max_difference=qp_chi_error)
            assert abs(actual-full_defect-tail-quadrature)<1e-12
            assert abs(predicted-measured)<1e-11
            assert qp_history_error<1e-10 and qp_chi_error<1e-9
            rows.append(row)
        with (path/f'identity_audit_{step}.csv').open('w') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        worst = max(rows,key=lambda r:abs(r['actual_signed_defect']))
        interior = [r for r in rows if .046875<=r['x']<=.203125]
        summaries.append(dict(step=step,worst=worst,
            interior_max_actual_defect=max(abs(r['actual_signed_defect']) for r in interior),
            max_full_defect=max(abs(r['full_signed_defect']) for r in rows),
            max_tail=max(abs(r['omitted_signed_tail']) for r in rows),
            max_bulk_quadrature_defect=max(abs(r['bulk_quadrature_signed_defect']) for r in rows),
            max_qp_history_difference=max(r['qp_history_max_difference'] for r in rows),
            max_qp_chi_difference=max(r['qp_chi_max_difference'] for r in rows)))
    (path/'identity_audit.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
