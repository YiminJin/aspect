#!/usr/bin/env python3
"""Frozen matching-reference paths: isolate time quadrature without evolution."""
import hashlib
import numpy as np
import native_endpoint as native


def main():
    directory = native.E/'prefix-2-fine-1024-onset'
    fields = np.load(directory/'fields.npz')
    history = native.old.read(directory/'history.json')
    reference = native.SharedFE().r
    keys = ['external_work', 'bulk_viscous', 'cohesive_viscous', 'friction',
            'mechanical_power', 'friction_residual_power', 'phase_power',
            'bulk_profile_power', 'normalization_power', 'external_remainder']
    rows = []
    for time in [.5, .5283203125, .529296875, .53125, .625]:
        index = int(np.flatnonzero(fields['time'] == time)[0])
        arguments = (reference, fields['phase'][index-1], fields['phase'][index],
                     fields['R'][index-1], fields['Z'][index-1],
                     fields['R'][index], fields['Z'][index],
                     fields['velocity'][index-1], history[index-1]['V'],
                     history[index-1]['dt'])
        ledgers = {order:native.old.core.path_ledger(*arguments, order=order)
                   for order in [8, 16, 32]}
        differences = {str(order):{key:abs(ledgers[order][key]-ledgers[32][key])
                                  for key in keys} for order in [8, 16]}
        assert max(differences['8'].values()) < 1e-10
        rows.append(dict(time=time, quadrature_differences=differences,
                         reference={key:ledgers[32][key] for key in keys}))
    result = dict(scope='Frozen prototype paths, same spatial operators; no physical history advanced. This estimates time quadrature, not native spatial error or contact-path consistency.',
                  input_sha256=hashlib.sha256((directory/'fields.npz').read_bytes()).hexdigest(),
                  allowance_J_per_m=1e-10, rows=rows)
    native.old.write(native.E/'path-time-quadrature.json', result)
    print(native.old.json.dumps(dict(maximum_Q8_Q32=max(max(row['quadrature_differences']['8'].values()) for row in rows),
                                    maximum_Q16_Q32=max(max(row['quadrature_differences']['16'].values()) for row in rows))))


if __name__ == '__main__':
    main()
