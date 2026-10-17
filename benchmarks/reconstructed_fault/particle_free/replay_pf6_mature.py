#!/usr/bin/env python3
"""Verify/extract compact mature evidence and independently replay its measured norms.

Offline only: no ASPECT, MPI, or archived executable is launched. Destination
must be new. Run against the supplied evidence directory and a fresh output path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('evidence',type=Path);p.add_argument('output',type=Path)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
provenance=json.loads((a.evidence/'provenance.json').read_text())
archive=a.evidence/'reference.tar.xz'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
assert digest(archive)==provenance['archive_sha256']
expected=json.loads((a.evidence/'archive-manifest.json').read_text())
seen=set()
with tarfile.open(archive,'r|xz') as tar:
    for member in tar:
        name=Path(member.name)
        assert member.isfile() and not name.is_absolute() and '..' not in name.parts and member.name in expected
        destination=a.output/name;destination.parent.mkdir(parents=True,exist_ok=True)
        h=hashlib.sha256()
        with tar.extractfile(member) as source,destination.open('xb') as target:
            for chunk in iter(lambda:source.read(1024*1024),b''):
                h.update(chunk);target.write(chunk)
        assert h.hexdigest()==expected[member.name],member.name
        seen.add(member.name)
assert seen==set(expected)
scripts=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(scripts/'verify_pf6_mature_math.py'),str(a.output/'replayed-math.json')],check=True)
subprocess.run([sys.executable,str(scripts/'analyze_pf6_mature.py'),str(a.output/'runs'),
                '--output',str(a.output/'replayed-summary.json')],check=True)
original=json.loads((a.evidence/'measured-summary.json').read_text())
replayed=json.loads((a.output/'replayed-summary.json').read_text())
assert original==replayed,'Replayed norms differ from the recorded measured summary'
# Recompute the cancellation ratios from full authoritative tensors too.
import numpy as np
sys.path.insert(0,str(scripts))
from analyze_pf6_mature import read,norm3
cancellation=json.loads((a.evidence/'cancellation-summary.json').read_text())
for label,step in (('dg-10',10),('dg-q6',3),('dg-fine',3)):
    data=read(a.output/'runs'/label/'run'/f'authoritative-stress-{step}.csv')
    tensor={'tau_xx':data['T_xx'],'tau_yy':data['T_yy'],
            'tau_zz':-data['T_xx']-data['T_yy'],'tau_xy':data['T_xy']}
    w=data['JxW']*data['chi']
    trial=float(np.sqrt(sum(w*norm3(tensor))/sum(w)))
    physical=float(np.sqrt(sum(w*norm3(data))/sum(w)))
    assert cancellation[label]==dict(localized_T_RMS_Pa=trial,localized_tau_RMS_Pa=physical,
                                     T_to_tau_norm_ratio=trial/physical)
# Verify the unfiltered traction export without a native solve.
raw=a.output/'replayed-raw-tractions.csv'
subprocess.run([sys.executable,str(scripts/'export_pf6_mature_tractions.py'),
    str(a.output/'runs/dg-10/run/authoritative-stress-10.csv'),str(raw)],check=True)
assert digest(raw)==digest(a.output/'derived/dg-10-raw-tractions.csv')
result=dict(archive_sha256=provenance['archive_sha256'],verified_members=len(seen),
            all_measured_norms_reproduced=True,independent_mathematical_checks_passed=True,
            ASPECT_or_MPI_launched=False,spatial_accuracy_qualified=False,
            time_perturbation_load_screen_passed=False,strict_reference_match_passed=False)
(a.output/'replay.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
