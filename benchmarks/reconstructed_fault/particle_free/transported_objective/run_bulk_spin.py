#!/usr/bin/env python3
"""Fresh bounded bulk-spin fixture using its matched candidate core/checker."""
import sys
from pathlib import Path
import run_material_native
if __name__=='__main__':
    build=run_material_native.BUILD/'bulk-spin'
    sys.argv[1:1]=['--core',str(build/'aspect-release'),'--checker',str(build/'checks/libtransported_bulk_spin.release.so')]
    if '--constitutive-spin' not in sys.argv:sys.argv.extend(['--constitutive-spin','bulk slip subtracted'])
    run_material_native.main()
