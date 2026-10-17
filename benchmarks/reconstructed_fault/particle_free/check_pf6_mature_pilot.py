#!/usr/bin/env python3
"""Verify preserved entry evidence/defaults and the exact final publication guard."""
import argparse,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]

def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    incoming=json.loads((a.directory/'incoming.json').read_text());changed=[];missing=[]
    for name,digest in incoming.items():
        path=ROOT/name
        if not path.is_file():missing.append(name)
        elif sha(path)!=digest:changed.append(name)
    assert not missing,missing
    allowed={'source/material_model/phase_field_fault.cc','include/aspect/material_model/phase_field_fault/fe_strain_internal.h',
      'source/material_model/phase_field_fault/dg_accepted.cc','source/material_model/phase_field_fault/dg_history.cc',
      'source/material_model/phase_field_fault/history.cc','source/material_model/phase_field_fault/normalization.cc',
      'source/simulator/core.cc','source/simulator/phase_field.cc','source/simulator/solver/reconstructed_fault_stokes.cc',
      'benchmarks/reconstructed_fault/bp3_native/coupled_checks.cc','benchmarks/reconstructed_fault/particle_free/run_pf6_mature.py',
      'doc/reconstructed_fault/current_design.md','doc/reconstructed_fault/specification.tex',
      'doc/reconstructed_fault/particle_free/STATUS.md','doc/reconstructed_fault/particle_free/SOURCE_AUDIT.md',
      'benchmarks/reconstructed_fault/particle_free/README.md'}
    assert set(changed)<=allowed,set(changed)-allowed
    path='source/material_model/phase_field_fault/fe_strain.cc';assert sha(ROOT/path)==incoming[path]
    entry=a.directory/'entry/source/material_model/phase_field_fault.cc';current=ROOT/'source/material_model/phase_field_fault.cc'
    defaults=lambda path:re.findall(r'prm.declare_entry\s*\(\s*"([^"]+)"\s*,\s*"([^"]*)"',path.read_text())
    assert defaults(entry)==defaults(current)
    for path in ['source/material_model/phase_field_fault/dg_limiter.cc','source/material_model/phase_field_fault/dg_transport.cc',
                 'source/particle/manager.cc','source/particle/particle_domain.cc','source/reconstructed_fault/manager.cc']:
        assert sha(ROOT/path)==incoming[path]
    archives={
     'pf6_mature_resolution':'92a1e76500b3da0222802ca0ba9b4b0a4f0e5777104bab48c1a7374ce552f2a6',
     'pf6_mature_dg':'eec5f1d5465026977aa4a08c7264f2e9e5695addb632b3b5327800b3558b3881',
     'pf6_domain_free':'aad666529a70a6aba4d3aae9e3a6c203198e0bebcef58dc3532786313c229e8e'}
    for name,digest in archives.items():assert sha(ROOT/'benchmarks/reconstructed_fault/particle_free/evidence'/name/'reference.tar.xz')==digest
    prior=json.loads((ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/pf6_mature_resolution/final-checks.json').read_text())
    assert prior['strict_reference_match_passed'] is False
    log=ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence/pf6-mature-aspect-unit-delivery.log'
    assert 'All tests passed (118034 assertions in 124 test cases)' in log.read_text()
    summary=json.loads((a.directory/'measured-summary.json').read_text());assert summary['functional_checks_passed']
    result=dict(entry_files=len(incoming),missing_entry_files=missing,changed_entry_files=changed,
       unchanged_material_defaults=len(defaults(current)),publication_allowance_source_unchanged=True,
       particle_domain_and_WENO_sources_unchanged=True,prior_archives=archives,strict_reference_match_passed=False,
       official_unit_cases=124,official_unit_assertions=118034,functional_checks_passed=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
