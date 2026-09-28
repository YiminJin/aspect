"""Prepare independent dt/half/quarter checkpoint copies; never launch a solve."""
import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from stage_normal_stress_diagnostic import stage, sha


def prepare(args):
    root = args.destination.resolve()
    if root.exists():
        raise ValueError('Preserve existing evidence: destination must be new')
    root.mkdir(parents=True)
    for name, fraction in [('dt', 1), ('half', .5), ('quarter', .25)]:
        dest = root/name
        stage(SimpleNamespace(checkpoint=args.checkpoint, input=args.input, job=args.job,
                              destination=dest, bp5_library=args.build/'libbp5_steady_initialization.release.so',
                              diagnostic_library=args.build/'libbp5_normal_stress_diagnostic.release.so',
                              steps=1, control=False, local_verification=False))
        lib = args.build/'libbp5_stress_cycle.release.so'
        shutil.copy2(lib, dest/lib.name)
        prm = dest/'normal_stress_diagnostic_restart.prm'
        text = prm.read_text()
        text += f'''
set Additional shared libraries = ./libbp5_steady_initialization.release.so, ./libbp5_normal_stress_diagnostic.release.so, ./libbp5_stress_cycle.release.so
subsection Particles
  set Integration scheme = rk2
end
subsection Termination criteria
  set Checkpoint on termination = false
end
subsection Postprocess
  set List of postprocessors = reconstructed fault BP3, particles, BP3 output complete, BP5 normal diagnostic, stress cycle
  subsection BP5 normal diagnostic
    set Small windows only = true
    set Raw every step = true
    set Native centerline = false
    set Wall seconds = 900
  end
  subsection Stress cycle
    set Timestep fraction = {fraction}
  end
end
'''
        prm.write_text(text)
        (dest/'cycle.json').write_text(json.dumps(dict(fraction=fraction, prm_sha256=sha(prm),
                                                     cycle_library_sha256=sha(dest/lib.name)), indent=2)+'\n')
    print(root)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'input', 'job', 'build', 'destination'):
        p.add_argument('--'+key, type=Path, required=True)
    prepare(p.parse_args())
