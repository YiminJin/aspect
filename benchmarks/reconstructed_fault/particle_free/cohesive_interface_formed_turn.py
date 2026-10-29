#!/usr/bin/env python3
"""Bounded turn: reuse the admission owner; refine only future material cells."""
import argparse
import json
import shutil
from pathlib import Path

import cohesive_interface_candidate_admission as owner
from cohesive_interface_path import ROOT,save,sha

WORK=ROOT/'build-particle-free/cohesive-interface-formed-turn'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_formed_turn'
SOURCE=ROOT/'build-particle-free/cohesive-interface-candidate-admission/half-step/final.json'
SPACING={'coarse':.09375,'fine':.046875}
_kernel=owner.kernel


def bounded_kernel(arcs,n=64,probe_order=8):
    assert shutil.disk_usage(WORK).free>128*2**20,'Disk reserve reached before new native solve'
    size=sum(p.stat().st_size for p in WORK.rglob('*') if p.is_file() and not p.is_symlink())
    assert size<1.1*2**30,'Declared new raw-output budget reached'
    k=_kernel(arcs,n,probe_order)
    # This diagnostic is unused here. Keep all center bases used in stress
    # preservation and all native operators. Its exact exporter input is saved.
    p=k.path/'field-mean.bin.gz'
    if p.exists():
        save(k.path/'omitted-mean-diagnostic.json',dict(bytes=p.stat().st_size,sha256=sha(p),
            reason='Unused cell-mean diagnostic only; no native history or operator removed',
            regenerate='Repeat the native export using geometry.csv and input.json into a new folder'))
        p.unlink()
    return k


def configure(mode):
    owner.WORK=WORK;owner.OUT=OUT;owner.CELL=SPACING[mode];owner.kernel=bounded_kernel


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=list(SPACING));p.add_argument('--label')
    p.add_argument('--checkpoint',type=Path,default=SOURCE);p.add_argument('--end',type=float,default=.440)
    p.add_argument('--dt',type=float,default=.000125)
    a=p.parse_args();configure(a.mode);label=a.label or a.mode
    config=dict(mode=a.mode,cell_length_m=owner.CELL,dt_s=a.dt,candidate_extent_m=.1875,
        start_checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),end_s=a.end,
        fixed_material_prefix=True,physical_R_m=.75,numerical_support_m=.75,source_revision='f1f9f29cf40c32d1ddcea000b724375b64853532')
    save(OUT/(label+'-input.json'),config)
    owner.run(label,cells=round(.1875/owner.CELL),dt=a.dt,cadence=.0025,checkpoint=a.checkpoint,end=a.end)


if __name__=='__main__':main()
