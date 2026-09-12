"""Saved-data K4.2 plots; no filtering of stresses or history correction."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reference_check import HERE,Y


def main():
    cases={c:json.loads((HERE/f'k42_{c}-analysis.json').read_text()) for c in 'ABC'}
    p={c:np.load(HERE/f'k42_{c}-profiles.npz') for c in cases}
    fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for c in cases:
        ell=.15625 if c=='A' else .078125
        ax[0,0].plot(p[c]['phase_y'],p[c]['phase'],label=c)
        ax[0,1].plot(p[c]['phase_y']/ell,p[c]['phase'],label=c)
    ax[0,0].set(xlabel='y (m)',ylabel='Initialized phi',xlim=(-.4,.4))
    ax[0,1].set(xlabel='y / ell',ylabel='Initialized phi',xlim=(-3,3))
    for j,t in enumerate((4.,6.)):
        k=round(t/.125)
        ref={n:np.load(HERE/'k41'/f'{n}-4096'/'dt0.125.npz')['velocity'][k] for n in ('ell0','half')}
        ax[1,j].plot(Y,ref['half']-ref['ell0'],'k--',label='independent dt=.125')
        for c in 'BC':
            ax[1,j].plot(Y,p[c]['velocity'][k]-p['A']['velocity'][k],label=f'{c} - A')
        ax[1,j].set(xlabel='y (m)',ylabel='Width difference in ux (m/s)',title=f't = {t:g} s')
        np.savetxt(HERE/f'k42-velocity-width-t{t:g}.csv',
                   np.c_[Y,ref['half']-ref['ell0'],p['B']['velocity'][k]-p['A']['velocity'][k],
                         p['C']['velocity'][k]-p['A']['velocity'][k]],
                   delimiter=',',header='y,independent,B_minus_A,C_minus_A',comments='')
    for a in ax.flat: a.legend(); a.grid(alpha=.2)
    fig.savefig(HERE/'k42-profiles.png',dpi=160)


if __name__=='__main__': main()
