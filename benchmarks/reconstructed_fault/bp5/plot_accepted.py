"""Plot only accepted candidate states; failed step 2 is not a data point."""
import json
import numpy as np
from run_short import OUT
from analyze_mechanical_modes import table
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run=OUT/'candidate-six'
accepted=np.atleast_1d(table(run/'accepted_steps.csv'))
for name,lo,hi in [('whole',0,115.471),('transition',13,20),('top',0,2),('bottom',113.47,115.471)]:
    fig,axes=plt.subplots(1,5,figsize=(16,3.5))
    for row in accepted:
        step=int(row['step'])
        state=np.atleast_1d(table(run/f'state_work_{step}.csv'))
        weak=np.atleast_1d(table(run/f'work_weak_{step}.csv'))
        order=np.argsort(state['xd'])
        x=state['xd'][order]/1000
        values=[state['V']*1e9,state['Theta_out'],state['slip']*1000,
                weak['q']/weak['weight']-26546122.365139291,weak['sigma']/weak['weight']-50e6]
        labels=['V/Vp','Committed Theta (s)','Slip (mm)','Current work q - nominal (Pa)','Current work sigma - 50 MPa (Pa)']
        for ax,value,label in zip(axes,values,labels):
            ax.plot(x,value[order],label=f'accepted {step}: t={row["time"]:g} s',ls='-' if step==0 else '--')
            ax.set(xlim=(lo,hi),xlabel='Down dip (km)',ylabel=label)
            ax.grid(alpha=.2)
    axes[1].set_yscale('log');axes[0].legend(fontsize=7)
    fig.suptitle('Mature 2-D BP5-friction diagnostic: current stress uses incoming working history')
    fig.tight_layout();fig.savefig(OUT/f'accepted_{name}.png',dpi=150);plt.close(fig)

summary=json.loads((run/'checks.json').read_text())
print('Plotted genuinely accepted steps:',[s['step'] for s in summary])
