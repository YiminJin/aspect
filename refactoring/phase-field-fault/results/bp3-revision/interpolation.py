"""Compare both ordinary pilot outputs at fixed clocks and material IDs."""
import argparse
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

def resistance(age, s):
    a = .01 + .015*np.clip((s-15000.)/3000., 0., 1.)
    return 5e7*a*np.arcsinh(.0005*np.exp((.6+.015*np.log(1.e-6*age/.008))/a)) + 4624440.e-9

def compare(before, after, output):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    with (output/"profiles.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s","id","distance_m","linear_age_s","log_age_s","linear_rate_m_s","log_rate_m_s"])
        for step, t in enumerate([0,400,900]):
            a,b = [np.loadtxt(root/f"fault-labels-{step:05}.txt") for root in (before,after)]
            np.testing.assert_array_equal(a[:,:3], b[:,:3])
            if step == 0:
                np.testing.assert_array_equal(a[:,3:5], b[:,3:5])
            s = 2*a[:,1]
            writer.writerows(zip(np.full(len(s),t), a[:,0].astype(int),s,a[:,3],b[:,3],a[:,5],b[:,5]))
            print(t, "max rate difference (m/s)", max(abs(a[:,5]-b[:,5])), "max initial-target deviations",
                  *(max(abs(v[:,5]/1e-9-1)) for v in (a,b)))
            if step in (0,2):
                for ax, center in zip(axes[step//2], (15000,18000)):
                    for data,style,label in ((a,"o-","Linear age"),(b,".-","Log age")):
                        ax.plot(s/1000,data[:,5]*1e9,style,label=label,markersize=3)
                    ax.set(xlim=((center-750)/1000,(center+750)/1000),xlabel="Down-dip distance (km)")
                    mask = abs(s-center)<=750
                    ax.set_ylim(min(a[mask,5].min(),b[mask,5].min())*1e9-.00025,
                                max(a[mask,5].max(),b[mask,5].max())*1e9+.00025)
                axes[step//2,0].set_ylabel(f"Speed at {t} s (nm/s)")
            if step==0:
                mid = .5*(s[1:]+s[:-1])
                with (output/"midpoints.csv").open("w") as midfile:
                    w=csv.writer(midfile)
                    w.writerow(["distance_m","linear_resistance_Pa","log_resistance_Pa"])
                    tractions=[resistance(age,mid) for age in ((a[:-1,3]+a[1:,3])/2,
                                                               np.sqrt(b[:-1,3]*b[1:,3]))]
                    w.writerows(zip(mid,*tractions))
                    print("max midpoint error at plate speed, 50 MPa (Pa)",*(max(abs(v-26546122.365133367)) for v in tractions))
    axes[0,0].legend()
    fig.savefig(output/"transition.png",dpi=150)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("before","after","output"):
        p.add_argument(name,type=Path)
    compare(**vars(p.parse_args()))
