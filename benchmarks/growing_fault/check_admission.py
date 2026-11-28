"""Exercise an isolated, positive-phase band outside the forward disk at rest and under load."""
import argparse
import csv
import os
from pathlib import Path
import subprocess

from parameter_file import public_nine_strip_input


def check(aspect, plugin, directory):
    directory.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, DEAL_II_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    base = public_nine_strip_input()
    for loaded in (False, True):
        run = directory/('loaded' if loaded else 'rest')
        run.mkdir()
        output = run/'output'
        prm = base + f'''
set Additional shared libraries = {plugin}
set Output directory = {output}
set End time = .005
subsection Material model
 subsection Phase field fault
  set Forward seed tip = .5,.5
 end
end
subsection Initial composition model
 subsection Function
  set Function expression = 0;0;0;.2*(max(0,1-((x-.3)/.32)^2)+max(0,1-((x-1.8)/.2)^2))*max(0,1-((y-.5)/.3)^2)
 end
end
subsection Postprocess
 set List of postprocessors = growing fault samples
end
'''
        if not loaded:
            prm += '\nsubsection Boundary velocity model\n subsection Function\n  set Function expression = 0;0\n end\nend\n'
        (run/'input.prm').write_text(prm)
        with (run/'log').open('w') as log:
            result = subprocess.run([aspect, run/'input.prm'], stdout=log, stderr=subprocess.STDOUT, env=env)
        if loaded:
            if result.returncode==0 or 'Detached exterior cohesive history' not in (run/'log').read_text():
                raise ValueError(f'Expected exterior history diagnosis. See {run}/log')
            if (output/'balance-0001.csv').exists():
                raise ValueError('Published an unsupported detached endpoint')
            print('Loaded exterior history diagnosed before the first physical publication')
        else:
            result.check_returncode()
            with (output/'sections-0001.csv').open() as stream:
                flags = {int(row['section']):int(row['formed']) for row in csv.DictReader(stream)}
            with (output/'points-0001-r0.csv').open() as stream:
                exterior = [row for row in csv.DictReader(stream) if float(row['x'])>1.7 and float(row['phi'])>.05]
            if not exterior or not any(flags.values()) or any(flags[int(row['section'])] for row in exterior):
                raise ValueError('Detached positive-phase band incorrectly registered its plane')
            print(f'{len(exterior)} positive exterior samples retained unformed strips at rest')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('aspect','plugin','destination'):
        parser.add_argument(key,type=lambda p:Path(p).resolve())
    args = parser.parse_args()
    check(args.aspect, args.plugin, args.destination)
