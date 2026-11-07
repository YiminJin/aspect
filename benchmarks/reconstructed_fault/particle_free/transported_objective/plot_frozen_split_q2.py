#!/usr/bin/env python3
"""Plot retained frozen split-Q2 accuracy and representation cost; no reruns."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/frozen-split-q2-matplotlib')
import argparse
import json
from pathlib import Path
import numpy as np
from compare_frozen_split_q2 import OUT, REFERENCE, GRIDS, METRIC, SplitQ2, load_fixture, index_basis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    new = json.loads((args.output/'comparison.json').read_text())
    old = json.loads((args.reference/'comparison.json').read_text())
    rows = {r['n']:r for r in new['cases']}
    previous = {r['n']:r for r in old['cases'] if r['q']=='cut'}
    cost = new['costs']
    fig,ax = plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for law,color,label in [(0,'tab:blue','Constant viscosity'),(1,'tab:orange','Layered viscosity')]:
        for split,style,marker in [(False,'--','x'),(True,'-','o')]:
            dofs = [cost[str(n)]['tensor_scalar_dofs' if split else 'uncut_scalar_dofs'] for n in GRIDS]
            error = [100*(rows[n] if split else previous[n])['localized_source_relative_L2'][law] for n in GRIDS]
            name = 'split Q2' if split else 'uncut DGQ2'
            ax[0,0].loglog(dofs,error,color=color,linestyle=style,marker=marker,label=f'{label}, {name}')
    ax[0,0].set(xlabel='Scalar coefficients per physical tensor',ylabel='L2 error / localized source (%)',
                title='Added local freedom removes the end-jump floor')
    ax[0,0].legend(fontsize=7)
    for metric,color,marker in [('point','tab:red','o'),('work','tab:green','s')]:
        for split,style in [(False,'--'),(True,'-')]:
            data=[]
            for n in GRIDS:
                if metric=='point':
                    data.append(100*(rows[n] if split else previous[n])['point_traction']['all']['relative_L2'][0])
                else:
                    data.append(100*rows[n]['work_traction']['split' if split else 'uncut']['all']['relative_L2'][0])
            ax[0,1].loglog(GRIDS,data,color=color,marker=marker,linestyle=style,
                          label=f'{metric.capitalize()}, {"split" if split else "uncut"}')
    ax[0,1].set(xlabel='Tensor grid width (cells)',ylabel='Relative traction error (%)',
                title='Constant viscosity: point and work traction')
    ax[0,1].legend(fontsize=8)
    f=load_fixture(args.reference/'frozen-input.npz')
    along=f.length+np.linspace(-.10,.10,2000)
    x=f.X[0]+along[:,None]*f.t
    split_coeff=np.load(args.output/'split-coefficients.npz')['n64']
    uncut_coeff=np.load(args.reference/'projection-coefficients.npz')['n64_cut']
    ids,B=index_basis(x,64)
    tau=np.einsum('pk,pkc->pc',B,uncut_coeff[ids,:,0])
    ax[1,0].plot(along-f.length,tau@(METRIC@f.S),'--',color='tab:orange',label='64² uncut best projection')
    tau=SplitQ2(f,64).evaluate(x,split_coeff)[:,0]
    ax[1,0].plot(along-f.length,tau@(METRIC@f.S),color='tab:blue',label='64² split projection')
    ax[1,0].plot(along-f.length,f.fields(x)[0][:,0]@(METRIC@f.S),':',color='black',label='Unchanged local source')
    ax[1,0].axvline(0,color='gray',linewidth=.8)
    ax[1,0].set(xlabel='Distance from right finite end (m)',ylabel='Frozen shear stress (Pa)',
                title='One-sided stress jump is represented')
    ax[1,0].legend(fontsize=8)
    overhead=[100*cost[str(n)]['history_fraction_added'] for n in GRIDS]
    ax[1,1].bar(range(4),overhead,color='tab:blue',width=.65)
    ax[1,1].set_xticks(range(4),[f'{n}²' for n in GRIDS])
    for i,(n,v) in enumerate(zip(GRIDS,overhead)):
        ax[1,1].text(i,v+.3,f'{v:.2f}%\n+{cost[str(n)]["additional_pieces"]} pieces',ha='center',fontsize=9)
    ax[1,1].set(xlabel='Tensor grid',ylabel='Additional stress-history storage (%)',ylim=(0,19),
                title='Only end-crossed cells gain coefficients')
    for a in ax.flat:a.grid(alpha=.25)
    fig.savefig(args.output/'frozen-split-q2.png',dpi=180)
    fig.savefig(args.output/'frozen-split-q2.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
