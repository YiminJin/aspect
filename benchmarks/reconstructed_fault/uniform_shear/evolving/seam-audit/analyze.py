"""Phase-only seam discrimination; no simulation or equation modification."""
import hashlib
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--corrected',action='store_true')
args=parser.parse_args()
suffix='-corrected' if args.corrected else ''
data = here / ('output'+suffix)
saved = here.parent / 'spatial0375_n128_f32'
root = here.parents[4]
hx = 0.25 / 32


def read(path):
    return np.genfromtxt(path, delimiter=',', names=True)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


report = {'definition': 'rhs is minus the production weak phase residual; density = rhs / constrained CPDI lumped test mass',
          'fixed_interior': [0.046875, 0.203125], 'hx': hx, 'states': {}}
fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
column_rows = []
baseline_particles = read(saved / 'particles_0.csv')
baseline_by_id = {int(row['id']): row for row in baseline_particles}
baseline_nodes = None
for step in (0, 2, 3):
    nodes = read(data / f'state{step}_nodes.csv')
    nodes = nodes[nodes['constrained'] == 0]
    particles = read(data / f'state{step}_particles.csv')
    xs = np.unique(nodes['x'])
    ys = np.unique(nodes['y'])
    arrays = {}
    for key in ('rhs', 'reaction', 'gradient', 'mass', 'phi'):
        arrays[key] = np.array([nodes[key][nodes['x'] == x][np.argsort(nodes['y'][nodes['x'] == x])]
                                for x in xs])
    if baseline_nodes is None:
        baseline_nodes = arrays
    interior = (xs >= 0.046875) & (xs <= 0.203125)
    distance = np.minimum(xs, 0.25-xs)
    state = {}
    for key in ('rhs', 'reaction', 'gradient'):
        density = arrays[key] / arrays['mass']
        deviation = density - density[interior].mean(axis=0)
        weak_deviation = arrays[key] - arrays[key][interior].mean(axis=0)
        energy = np.sum(deviation**2, axis=1)
        state[key] = dict(weak_l2=float(np.linalg.norm(arrays[key])),
                          weak_nonuniform_l2=float(np.linalg.norm(weak_deviation)),
                          density_nonuniform_rms=rms(deviation),
                          fixed_interior_density_nonuniform_rms=rms(deviation[interior]),
                          seam_node_density_nonuniform_rms=rms(deviation[distance == 0]),
                          max_density_nonuniform=float(np.max(np.abs(deviation))),
                          seam_energy_fraction={str(layers): float(energy[distance <= layers*hx+1e-12].sum()/energy.sum())
                                                for layers in (0, 1, 2, 3)})
        for col, x in enumerate(xs):
            column_rows.append([step, key, x, rms(density[col]), rms(deviation[col]),
                                float(np.linalg.norm(arrays[key][col])), float(arrays['mass'][col].sum())])
        axes[0, ('rhs', 'reaction', 'gradient').index(key)].semilogy(xs, np.sqrt(np.mean(deviation**2, axis=1)),
                                                                 '.-', label=f'domains {step}')
        if key == 'rhs':
            axes[1, 0].plot(ys, deviation[0], label=f'seam, state {step}')
            axes[1, 1].plot(ys, density[interior].mean(axis=0), label=f'interior, state {step}')

    state['phi_max_tangential_difference'] = float(np.max(np.ptp(arrays['phi'], axis=0)))
    state['consistency'] = dict(
        partition_unity_max=float(np.max(np.abs(particles['sum_w']-1))),
        gradient_constant_max=float(np.max(particles['grad_constant'])),
        first_moment_max_m=float(np.max(particles['first_moment_error'])),
        physical_linear_gradient_error_max=float(np.max(particles['linear_gradient_error'])),
        positive_identity_count=int(np.sum(particles['linear_gradient_error'] < 1e-9)),
        negative_identity_norm_count=int(np.sum(np.abs(particles['linear_gradient_error']-2*np.sqrt(2)) < 1e-9)),
        polygon_volume_relative_max=float(np.max(np.abs(particles['polygon_area']/particles['volume']-1))),
        total_domain_measure=float(np.sum(particles['volume'])),
        total_measure_relative_error=float(np.sum(particles['volume'])/0.25-1),
        max_abs_homogeneous_input_cpdi_grad_x=float(np.max(np.abs(particles['gphi_x']))))
    assert state['phi_max_tangential_difference'] == 0
    # Do not turn the newly measured gradient-orientation issue into a pass.
    assert state['consistency']['partition_unity_max'] < 1e-12
    assert state['consistency']['first_moment_max_m'] < 1e-10

    original = np.array([baseline_by_id[int(i)] for i in particles['id']], dtype=baseline_particles.dtype)
    volume_ratio = particles['volume']/original['volume']
    physical_distance = np.minimum(particles['x'], 0.25-particles['x'])
    column = np.clip(np.floor(particles['x']/hx).astype(int), 0, 31)
    state['particle_regions'] = {}
    for name, mask in dict(seam_cells=physical_distance<hx,
                           fixed_interior=(particles['x']>=0.046875)&(particles['x']<=0.203125)).items():
        state['particle_regions'][name] = dict(
            count=int(mask.sum()),
            volume_ratio_min=float(volume_ratio[mask].min()), volume_ratio_max=float(volume_ratio[mask].max()),
            centroid_minus_parent_y_rms=rms(particles['cy'][mask]-particles['y'][mask]),
            grad_x_rms=rms(particles['gphi_x'][mask]),
            partition_unity_max=float(np.max(np.abs(particles['sum_w'][mask]-1))),
            gradient_constant_max=float(np.max(particles['grad_constant'][mask])),
            first_moment_max_m=float(np.max(particles['first_moment_error'][mask])))
    for c in range(32):
        mask = column == c
        column_rows.append([step, 'domain_volume_ratio', (c+0.5)*hx,
                            float(volume_ratio[mask].mean()), rms(volume_ratio[mask]-1),
                            float(particles['volume'][mask].sum()), int(mask.sum())])
    axes[1, 2].plot([(c+0.5)*hx for c in range(32)],
                    [rms(volume_ratio[column == c]-1) for c in range(32)], '.-', label=f'domains {step}')

    history = read(saved / f'particles_{step}.csv')
    initial_H = np.array([baseline_by_id[int(i)]['H'] for i in history['id']])
    state['actual_saved_H_minus_H0_max'] = float(np.max(np.abs(history['H']-initial_H)))
    state['actual_max_displacement_x'] = float(np.max(np.abs(particles['x']-original['x'])))
    state['actual_max_displacement_y'] = float(np.max(np.abs(particles['y']-original['y'])))
    # A wall-clipped row of equally spaced particles gains/loses a strip of
    # width dx at its two ends. This is a geometric diagnosis, not a new rule.
    pitch = 0.25/96
    wall = (original['x']<pitch) | (original['x']>0.25-pitch)
    displacement = particles['x']-original['x']
    wall_prediction = 1+np.where(original['x']<0.125,1,-1)*displacement/pitch
    state['wall_strip_model'] = dict(
        relative_volume_error_rms=rms((volume_ratio-wall_prediction)[wall]),
        relative_volume_error_max=float(np.max(np.abs(volume_ratio[wall]-wall_prediction[wall]))),
        centroid_shift_error_max_m=float(np.max(np.abs(
            particles['cx'][wall]-particles['x'][wall]+displacement[wall]/2))))
    entry = read(saved/f'phase_input_{step+1}_rank0.csv')
    accepted_by_id = {int(row['id']): row for row in history}
    state['following_phase_entry_position_difference_max'] = max(
        max(abs(row['x']-accepted_by_id[int(row['id'])]['x']),
            abs(row['y']-accepted_by_id[int(row['id'])]['y'])) for row in entry)
    assert state['following_phase_entry_position_difference_max'] == 0
    report['states'][str(step)] = state

for ax, name in zip(axes[0], ('Total phase RHS', 'Reaction contribution', 'Gradient contribution')):
    ax.set(title=name, xlabel='x (m)', ylabel='RMS deviation from interior profile (Pa)')
    ax.legend()
axes[1, 0].set(title='Seam minus fixed-interior residual', xlabel='y (m)', ylabel='Residual density (Pa)')
axes[1, 1].set(title='Tangentially averaged interior residual', xlabel='y (m)', ylabel='Residual density (Pa)')
axes[1, 2].set(title='Domain volume departure from initial', xlabel='x (m)', ylabel='RMS relative volume change')
for ax in axes[1]:
    ax.legend()
fig.savefig(here / ('seam-audit'+suffix+'.png'), dpi=160)
with (here / ('columns'+suffix+'.csv')).open('w') as out:
    out.write('state,quantity,x,rms_or_mean,nonuniform_rms,weak_l2_or_measure,mass_or_count\n')
    for row in column_rows:
        out.write(','.join(map(str,row))+'\n')
report['domain_recovery'] = [dict(zip(row.dtype.names,map(float,row))) for row in read(data/'domain_recovery.csv')]
report['resources'] = json.loads((here/('resources'+suffix+'.json')).read_text())
files = ['source/particle/particle_domain.cc', 'source/simulator/phase_field.cc',
         'tests/phase_field_test_access.h',
         'benchmarks/reconstructed_fault/uniform_shear/uniform_shear.cc',
         'build-pf-cpdi/aspect-release',
         str((data/'parameters.prm').relative_to(root)),
         str((here/'diagnostic.cc').relative_to(root)),
         str((here/'build/libk3_phase_seam_audit.release.so').relative_to(root))]
report['sha256'] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
(here/('summary'+suffix+'.json')).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
