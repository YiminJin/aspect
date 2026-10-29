#!/usr/bin/env python3
"""Fresh, equilibrated controls on the same prescribed saved path.

This avoids inheriting the numerical disequilibrium of an imported coarse
history. It is fresh initialization, never a reset/conversion of an old run.
"""
import copy
import numpy as np
import cohesive_interface_path as path
from cohesive_interface_admission import bounded_multiplier
from investigate_cohesive_interface_bulk_contact import WORK, OUT, ARCS, build, PATH_WORK, save


def initial(k):
    s=k.initial();old=k.s['x_m']<6
    for key in path.HISTORY:
        value=np.array(s['histories'][key]);value[~old]=1. if key=='theta' else 0.;s['histories'][key]=value.tolist()
    for key in ['jump_offset','last_resistance']:
        value=np.array(s[key]);value[~old]=0.;s[key]=value.tolist()
    c=np.r_[s['wall_memory'],s['memory']];tied=~old
    force=(k.K@c)[1:]+k.Q.T@(k.w*np.array(s['last_resistance']))
    lam,error,peak=bounded_multiplier(k,force,tied)
    assert error<1e-8 and peak<3000
    r=np.array(s['last_resistance']);r[tied]=lam;s['last_resistance']=r.tolist()
    assert np.all(np.array(s['histories']['damage'])>=0)
    return s,dict(true_weak_Pa=error,tied_peak_Pa=peak,clock_s=0.,old_seed_prescription_unchanged=True,
        new_tied_damage_zero=True,new_fracture_energy_zero=True,Maxwell_jump_memory_zero=True,
        initial_deviator_error_bound_Pa=.0003*k.meta['affine_unit_strain_error_Pa'],
        wall_memory=.0003,scope='Fresh common-path initialization, not modification of accepted histories')


def main():
    library=WORK/'build/libcohesive_interface_turn_solver.so'
    if not library.exists():library.symlink_to(PATH_WORK/'build/libcohesive_interface_turn_solver.so')
    if not (WORK/'build-work').exists():(WORK/'build-work').symlink_to('build',target_is_directory=True)
    path.WORK=WORK;path.OUT=OUT
    cases=[('fresh-wide64','wide64',64,2,'cartesian'),('fresh-wide96','wide96',96,2,'cartesian'),
        ('fresh-fitted64','fitted64',64,2,'fitted'),('fresh-fitted96','fitted96',96,2,'fitted')]
    for label,kernel_label,n,degree,mesh in cases:
        k=build(kernel_label,n,degree,.75,mesh);s,checks=initial(k);k.close()
        filename=WORK/(label+'-initial.json');save(filename,s);save(OUT/(label+'-initial-checks.json'),checks)
        def factory(arcs,n=64,degree=2,order=6,pressure_mode='work'):
            assert arcs==ARCS and pressure_mode=='work'
            return path.PathKernel(WORK/kernel_label,ARCS,'work')
        path.kernel=factory
        # The installed path already has the bounded 3m reserve. No insertion
        # is performed; the original physical loading/hold/controller is reused.
        path.run(label,n,degree,pressure_mode='work',checkpoint=filename)


if __name__=='__main__':main()
