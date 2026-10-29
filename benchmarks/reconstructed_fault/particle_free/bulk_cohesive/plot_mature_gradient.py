#!/usr/bin/env python3
"""Mature sliding evidence, physical axes and separate forces/energies."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_mature_gradient import read, column
from mature_gradient_strip import OUT, ELL

history, profiles = read('history.csv'), read('profiles.csv')
labels = [('local', 'Local law'), ('ls1', 'ℓs = 1 mm'), ('ls3', 'ℓs = 3 mm'), ('ls6', 'ℓs = 6 mm')]
plt.rcParams.update({'font.size':10, 'axes.titlesize':11})

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
for case, label in labels:
    rows = [r for r in history if r['case'] == case]
    axes[0].plot(1000*column(rows, 'delta_m'), column(rows, 'traction_Pa'), label=label)
axes[0].axhline(1000., color='.5', ls=':', label='μp₀, prescribed aligned plane')
axes[0].set(xlabel='Accumulated interface slip δ [mm]', ylabel='Physical shear traction [Pa]', title='Same imposed motion; actual accumulated slip on the axis')
axes[0].legend(fontsize=8)
rows = [r for r in history if r['case'] == 'ls6' and float(r['V_m_s']) > 1e-12]
x = 1000*column(rows, 'delta_m')
axes[1].stackplot(x, column(rows, 'Rfriction_Pa'), column(rows, 'Rcohesive_Pa'), column(rows, 'Rgradient_Pa'),
                  labels=['Friction (actual conjugate planes)', 'Cohesive backstress', 'Accumulated-gradient resistance'], alpha=.8)
axes[1].plot(x, column(rows, 'traction_Pa'), 'k--', lw=1, label='Measured traction')
axes[1].set(xlabel='Accumulated interface slip δ [mm]', ylabel='Work-conjugate traction [Pa]', title='6 mm: exact mechanical power partition')
axes[1].legend(fontsize=8, loc='upper left')
fig.suptitle('Fixed mature-like phase; stronger surroundings; microfree ends\nConstant μ = 0.1, solved physical pressure; RSF state evolution is not introduced')
fig.savefig(OUT/'traction-slip.png', dpi=170); plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
for case, label in labels:
    rows = [r for r in profiles if r['case'] == case and float(r['time_s']) == 2.]
    y, rate, k = column(rows, 'y_m'), column(rows, 'w_engineering_s1'), column(rows, 'k_engineering')
    mass = column(rows, 'weight_m'); V = mass @ rate
    axes[0, 0].plot(1000*y, rate, label=label)
    axes[0, 1].plot(1000*y, rate/V, label=label)
    axes[1, 0].plot(1000*y, k, label=label)
rows = [r for r in profiles if r['case'] == 'ls6' and float(r['time_s']) == 2.]
axes[1, 1].plot(1000*column(rows, 'y_m'), column(rows, 'Xxy_Pa'), label='Gradient backstress Xxy')
axes[1, 1].plot(1000*column(rows, 'y_m'), column(rows, 'Qxy_Pa'), label='Cohesive Qxy')
for ax in axes.flat:
    ax.axvspan(-1000*ELL/2, 1000*ELL/2, color='.8', alpha=.2)
    for y in [-ELL, ELL]: ax.axvline(1000*y, color='.4', lw=.7, ls=':')
    ax.set(xlabel='Normal coordinate r [mm]', xlim=(-100, 100))
axes[0, 0].set(title='Final engineering slip-strain rate', ylabel='w = 2 W:S [s⁻¹]'); axes[0, 0].legend(fontsize=8)
axes[0, 1].set(title='Normalized rate profile: ∫ρW dr = 1', ylabel='ρW = w/V [m⁻¹]')
phase_axis = axes[0, 1].twinx()
phase_axis.plot(1000*column(rows, 'y_m'), column(rows, 'phi'), 'k--', lw=.8, alpha=.6)
phase_axis.set(ylabel='Prescribed phase φ (unchanged)', ylim=(0, 1.05))
axes[1, 0].set(title='Final accumulated engineering shear', ylabel='k = 2 K:S [1]')
axes[1, 1].set(title='Different local forces, 6 mm', ylabel='Pa'); axes[1, 1].legend(fontsize=8)
fig.suptitle('t = 2 s; imposed relative motion = 2 mm; phase does not widen\nGray core: φ = .999; dotted support: shoulder joins φ = .02 surroundings. K and W are tensor histories, not crack angles.')
fig.savefig(OUT/'cross-band.png', dpi=170); plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
for ax, case, title in [(axes[0, 0], 'local', 'Local law: dynamic recoverable storage'), (axes[0, 1], 'ls6', '6 mm: additional recoverable gradient storage')]:
    rows = [r for r in history if r['case'] == case]
    x = 1000*column(rows, 'delta_m')
    ax.stackplot(x, column(rows, 'bulk_J_m2'), column(rows, 'cohesive_J_m2'), column(rows, 'gradient_J_m2'), labels=['Bulk', 'Cohesive', 'Gradient'], alpha=.8)
    ax.set(title=title, xlabel='Accumulated interface slip δ [mm]', ylabel='J/m²'); ax.legend(fontsize=8, loc='upper left')
rows = [r for r in history if r['case'] == 'ls6']; x = 1000*column(rows, 'delta_m'); dt = column(rows, 'dt_s')
Db, Dc, Df = [np.cumsum(dt*column(rows, key)) for key in ['Db_W_m2', 'Dc_W_m2', 'Df_W_m2']]
loss = np.cumsum(sum(column(rows, key) for key in ['BE_stress_J_m2', 'BE_cohesive_J_m2', 'BE_gradient_J_m2']))
storage = sum(column(rows, key) for key in ['bulk_J_m2', 'cohesive_J_m2', 'gradient_J_m2'])
axes[1, 0].stackplot(x, Df, Db, Dc, labels=['Friction', 'Bulk Maxwell', 'Cohesive Maxwell'], alpha=.8)
axes[1, 0].set(title='6 mm: cumulative physical dissipation', xlabel='Accumulated interface slip δ [mm]', ylabel='J/m²'); axes[1, 0].legend(fontsize=8)
axes[1, 1].plot(x, np.cumsum(column(rows, 'work_J_m2')), label='Boundary reaction work')
axes[1, 1].plot(x, storage+Df+Db+Dc+loss, '--', label='Storage + physical loss + BE remainder')
axes[1, 1].plot(x, loss, ':', label='BE numerical remainder separately')
axes[1, 1].set(title='6 mm: endpoint BE work accounting', xlabel='Accumulated interface slip δ [mm]', ylabel='J/m²'); axes[1, 1].legend(fontsize=8)
fig.suptitle('Fixed initial fracture storage = 2.64093 J/m² at N = 257; ΔEfracture = 0\nGradient storage is not frictional heat. Algebraic balance and the independent Δt check are reported separately.')
fig.savefig(OUT/'energy.png', dpi=170); plt.close(fig)
