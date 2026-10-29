#!/usr/bin/env python3
"""Read-only event metrics on the retained Cartesian Q1 phase space."""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.ndimage import label

ELL = .046875
DX = DY = .0078125
LEVELS = [.01, .05, .1]


class Events:
    def __init__(self, initial_path):
        a = np.loadtxt(initial_path, delimiter=',', skiprows=1)
        self.x, self.y = np.unique(a[:, 1]), np.unique(a[:, 2])
        assert len(self.x) == 193 and len(self.y) == 129
        self.ids = a[:, 0].astype(int)
        self.ix, self.iy = np.searchsorted(self.x, a[:, 1]), np.searchsorted(self.y, a[:, 2])
        self.initial = self.grid(a)
        self.mass = np.ones_like(self.initial)*DX*DY
        self.mass[[0, -1], :] *= .5
        self.mass[:, [0, -1]] *= .5
        self.X, self.Y = np.meshgrid(self.x, self.y)
        p = self.initial
        self.virgin_cells = (p[:-1, :-1] == 0) & (p[1:, :-1] == 0) & (p[:-1, 1:] == 0) & (p[1:, 1:] == 0)
        self.support = ~self.virgin_cells
        self.support_front = float(self.x[1:][self.support.any(axis=0)].max())
        assert self.support_front == 1.09375

    def grid(self, a):
        assert np.array_equal(a[:, 0].astype(int), self.ids)
        result = np.empty((len(self.y), len(self.x)))
        result[self.iy, self.ix] = a[:, 3]
        return result

    def phase(self, path):
        return self.grid(np.loadtxt(path, delimiter=',', skiprows=1))

    def components(self, phi, level):
        mask = phi >= level
        labels, _ = label(mask)  # Retained four-neighbor diagnostic convention.
        seeds = np.unique(labels[:, 0]); seeds = seeds[seeds > 0]
        connected = np.isin(labels, seeds)
        return labels, connected

    def front(self, phi, connected, level, corridor=False):
        use = connected.copy()
        if corridor:
            use &= abs(self.Y-.5) <= ELL
        values = self.X[use].tolist()
        for j, i in zip(*np.where(use[:, :-1] & ~connected[:, 1:])):
            if phi[j, i+1] < level:
                values.append(self.x[i]+DX*(level-phi[j, i])/(phi[j, i+1]-phi[j, i]))
        return float(max(values)) if values else None

    def measure(self, phi, previous, time, step):
        delta = phi-self.initial
        corners = [phi[:-1, :-1], phi[1:, :-1], phi[:-1, 1:], phi[1:, 1:]]
        virgin_peak = max(float(v[self.virgin_cells].max()) for v in corners)
        virgin_integral = float(sum(np.sum(v[self.virgin_cells]) for v in corners)*DX*DY/4)
        row = {'step': step, 'time_s': time, 'loading_displacement_m': .018*(time-.05),
               'peak_increment': float(delta.max()), 'max_step_increment': float(np.max(phi-previous)),
               'integrated_increment_m2': float(np.sum(self.mass*delta)),
               'initially_zero_cell_peak': virgin_peak, 'initially_zero_cell_integral_m2': virgin_integral,
               'initially_zero_nodal_peak': float(delta[self.initial == 0].max()),
               'represented_support_front_m': self.support_front, 'target_front_m': self.support_front+ELL}
        remote = boundary = False
        for level in LEVELS:
            labels, connected = self.components(phi, level)
            row[f'connected_front_phi{level}_m'] = self.front(phi, connected, level)
            row[f'axial_front_phi{level}_m'] = self.front(phi, connected, level, True)
            j = int(np.argmin(abs(self.y-.5)))
            center = connected.copy(); center[:j] = False; center[j+1:] = False
            row[f'centerline_front_phi{level}_m'] = self.front(phi, center, level)
            disconnected = (labels > 0) & ~connected
            row[f'disconnected_area_phi{level}_m2'] = float(np.sum(self.mass[disconnected]))
            if level == .01:
                remote_area = remote_peak = 0.
                for k in np.unique(labels[disconnected]):
                    component = labels == k
                    virgin = component & (self.initial == 0)
                    area = float(np.sum(self.mass[virgin]))
                    peak = float(delta[virgin].max()) if virgin.any() else 0.
                    remote_area = max(remote_area, area); remote_peak = max(remote_peak, peak)
                    remote |= area >= 4*DX*DY or (area >= DX*DY and peak >= .05)
                grown = connected & (delta >= .01)
                distance = np.minimum.reduce([1.5-self.X, self.Y, 1-self.Y])
                boundary = bool(np.any(grown & (distance <= 2*ELL)))
                row['largest_remote_initially_zero_area_m2'] = remote_area
                row['remote_initially_zero_peak_increment'] = remote_peak
                row['new_growth_boundary_distance_m'] = float(distance[grown].min()) if grown.any() else None
        old_labels, old_connected = self.components(previous, .05)
        old_front = self.front(previous, old_connected, .05, True)
        advance = row['axial_front_phi0.05_m']-old_front
        row['step_front_advance_phi0.05_m'] = advance
        row['intact_activation'] = virgin_peak > 1e-8
        row['rapid_growth'] = row['max_step_increment'] >= .01 or advance >= DX
        row['target_reached'] = row['axial_front_phi0.05_m'] >= self.support_front+ELL
        row['significant_remote_nucleation'] = bool(remote)
        row['boundary_approach'] = boundary
        row['stop_reason'] = ('target' if row['target_reached'] else 'remote_nucleation' if remote else 'boundary_approach' if boundary else '')
        return row


def write_rows(path, rows):
    if not rows:
        return
    with Path(path).open('w') as stream:
        # Initialization rows omit energy terms which only exist after a step.
        writer = csv.DictWriter(stream, list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    # Independent zero-growth/identity control on the actual represented seed.
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('initial', type=Path)
    args = parser.parse_args(); e = Events(args.initial)
    r = e.measure(e.initial, e.initial, .13875, 127)
    assert not r['intact_activation'] and not r['rapid_growth'] and not r['stop_reason']
    assert r['integrated_increment_m2'] == 0
    print(json.dumps(r, indent=2))
