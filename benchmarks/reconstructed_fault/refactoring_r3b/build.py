#!/usr/bin/env python3
"""Fresh R3b Release build using the qualified R3a compiler/options."""
from pathlib import Path
import subprocess
root = Path(__file__).resolve().parent
configure = ['cmake', '-S', '.', '-B', 'build-refactor-r3b',
             '-DCMAKE_C_COMPILER=/opt/openmpi/5.0.6/bin/mpicc',
             '-DCMAKE_CXX_COMPILER=/opt/openmpi/5.0.6/bin/mpic++',
             '-DCMAKE_Fortran_COMPILER=/opt/openmpi/5.0.6/bin/mpifort',
             '-DDEAL_II_DIR=/opt/dealii/9.6-local', '-DASPECT_WITH_VORO=ON',
             '-DVORO_DIR=/home/ein/local/voro++/0.4.6', '-DASPECT_WITH_NETCDF=OFF',
             '-DASPECT_ADDITIONAL_CXX_FLAGS=-fno-finite-math-only -ffp-contract=off',
             '-DCMAKE_BUILD_TYPE=Release', '-DASPECT_RUN_ALL_TESTS=ON',
             '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
for label, seconds, command in [('configure',180,configure),
                                ('build',1500,['cmake','--build','build-refactor-r3b','-j2'])]:
    subprocess.run(['python3',str(root/'run_logged.py'),label,str(seconds),*command],check=True)
