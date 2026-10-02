#!/usr/bin/env python3
"""Generate a separate probe; retain the original stale-diagnostic test unchanged."""
from pathlib import Path
import sys
repo=Path(__file__).resolve().parents[4]
s=(repo/'tests/phase_field_fault_surface_singular.cc').read_text()
s=s.replace('#include "phase_field_fault_surface_singular_system.cc"', (repo/'tests/phase_field_fault_surface_singular_system.cc').read_text())
old='"Failed to factor reconstructed-fault K_V block"'
assert s.count(old)==1
# Match the current LAPACK singular-pivot diagnostic, not an arbitrary exception.
s=s.replace(old,'"GTTRF failed, info=1, singular pivot vertex=0"')
s=s.replace('          catch (const std::exception &exception)\n            {', '          catch (const std::exception &exception)\n            {\n              this->get_pcout() << "Observed surface failure: " << exception.what() << std::endl;',1)
Path(sys.argv[1]).write_text(s)
