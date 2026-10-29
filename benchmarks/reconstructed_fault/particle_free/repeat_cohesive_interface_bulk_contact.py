#!/usr/bin/env python3
"""Fresh bounded causal runs with qualified fixed numerical supports.

Reuses the path controller with an explicit factory in this isolated process.
Original defaults, checkpoint files and production owners remain untouched.
"""
import hashlib
import numpy as np
import cohesive_interface_path as path
from investigate_cohesive_interface_bulk_contact import WORK, OUT, build, PATH_WORK


def factory(arcs,n=64,degree=2,order=6,pressure_mode='work'):
    assert pressure_mode=='work'
    signature=dict(arcs=arcs,n=n,degree=degree,order=order,lifting_width_m=.75,normal_width_m=.75,version='fixed-support-1')
    name='path-kernel-'+hashlib.sha256(path.encode(signature).encode()).hexdigest()[:16]
    return build(name,n,degree,.75,'cartesian',order,arcs,np.zeros(sum(a['cells'] for a in arcs)),
        purpose='Fresh causal workflow; fixed numerical lifting and normal test. No incoming history converted.')


def main():
    # The reused native C API is unchanged. Only this process's experimental
    # factory and archive roots change; no production or persisted default does.
    library=WORK/'build/libcohesive_interface_turn_solver.so'
    if not library.exists():library.symlink_to(PATH_WORK/'build/libcohesive_interface_turn_solver.so')
    if not (WORK/'build-work').exists():(WORK/'build-work').symlink_to('build',target_is_directory=True)
    path.WORK=WORK;path.OUT=OUT;path.kernel=factory
    path.run('causal-wide64',64,pressure_mode='work')
    path.run('causal-wide96',96,pressure_mode='work')


if __name__=='__main__':main()
