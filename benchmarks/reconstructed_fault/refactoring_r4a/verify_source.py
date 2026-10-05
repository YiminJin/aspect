#!/usr/bin/env python3
"""Prove the R4a definition move against the captured accepted source."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
e=root/'evidence'
old=(e/'solver-before.cc').read_text()
ordinary=Path('source/simulator/solver.cc').read_text()
fault=Path('source/simulator/solver/reconstructed_fault_stokes.cc').read_text()
header=Path('include/aspect/simulator/solver/stokes_operators.h').read_text()
def between(text,start,end):
    a=text.index(start)
    return text[a:text.index(end,a)]
checks={}
method='  template <int dim>\n  void\n  Simulator<dim>::solve_reconstructed_fault_stokes'
checks['complete-fault-body']=between(old,method,'\n}\n')==between(fault,method,'\n}\n')
checks['exclusive-helper']=between(old,'  namespace\n','  template <int dim>\n  double')==between(fault,'  namespace\n',method)
checks['stokes-block-declaration']=between(old,'    /**\n     * Implement multiplication','    void StokesBlock::vmult') in header
checks['schur-definitions']=between(old,'    /**\n     * Base class for Schur','\n  }\n') in header
checks['stokes-block-definitions']=between(old,'    void StokesBlock::vmult','    /**\n     * Base class for Schur')==between(ordinary,'    void StokesBlock::vmult','\n  }\n')
start='  template <int dim>\n  double'
checks['ordinary-methods']=between(old,start,method)==between(ordinary,start,'\n}\n')
checks['no-fault-driver-in-general-solver']='solve_reconstructed_fault_stokes' not in ordinary
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
entry=json.loads((e/'entry-source-hashes.json').read_text())
changed=[p for p,v in entry.items() if h(Path(p))!=v]
checks['only-authorized-existing-source-changes']=set(changed)==({'source/simulator/solver.cc','CMakeLists.txt'} & entry.keys())
protected=json.loads((e/'protected-hashes.json').read_text())
checks['reference-and-user-files-preserved']=all(Path(p).exists() and h(Path(p))==v for p,v in protected.items())
(e/'source-verification.json').write_text(json.dumps(dict(checks=checks,changed=changed),indent=2)+'\n')
print(json.dumps(checks,indent=2))
assert all(checks.values())
