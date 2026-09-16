"""Prepare, but NEVER execute, the single step-12 server diagnostic replay.

Use a rebuilt server binary/plugin compatible with the supplied 64-bit-index
checkpoint. Physical settings come from the actual server original.prm.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--plugin',type=Path,required=True)
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    repo=here.parents[2]
    source=here/'first_cycle_coarse'
    checkpoint=source/'restart/03'
    assert (source/'restart/last_good_checkpoint.txt').read_text().strip()=='3'
    assert checkpoint.is_dir() and (source/'original.prm').is_file()
    binary=args.binary.resolve(strict=True);plugin=args.plugin.resolve(strict=True)
    output=args.output.resolve()
    if output.exists():raise SystemExit('Output already exists; preserve previous attempts.')
    inputs={str(p):digest(p) for p in checkpoint.iterdir() if p.is_file()}
    output.mkdir(parents=True)
    shutil.copytree(checkpoint,output/'restart/03')
    for p in checkpoint.iterdir():
        if p.is_file():assert digest(output/'restart/03'/p.name)==inputs[str(p)]
    (output/'restart/last_good_checkpoint.txt').write_text('3\n')
    prm=output/'diagnostic.prm'
    prm.write_text(f'include {source}/original.prm\n'
                   f'set Additional shared libraries = {plugin}\n'
                   f'set Output directory = {output}\n'
                   'set Resume computation = true\n'
                   'subsection Termination criteria\n'
                   '  set Termination criteria = end step\n'
                   '  set End step = 12\n'
                   'end\n')
    sources=[here/'bp3.cc',here/'bp3_model.h',source/'original.prm',prm,binary,plugin,
             repo/'source/reconstructed_fault/surface_system.cc',
             repo/'source/simulator/assemblers/reconstructed_fault_stokes.cc']
    provenance=dict(checkpoint=inputs,files={str(p):digest(p) for p in sources},
                    required_index_bits=64,required_server_ranks=64,wall_cap_seconds=120,
                    executed=False)
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    q=shlex.quote
    print('Prepared and byte-verified restart/03. No ASPECT process launched.')
    print('Use the original 64-rank server allocation and compatible rebuilt binary/plugin:')
    print(f'cd {q(str(here))}')
    print(f'ASPECT_SOURCE_DIR={q(str(repo))} ASPECT_FAULT_EXPLICIT_B=1 '
          'ASPECT_FAULT_EXPLICIT_G=1 ASPECT_FAULT_SURFACE_SOLVER=tridiagonal '
          'ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC=1 '
          'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 '
          f'timeout --signal=TERM --kill-after=10s 120s ibrun {q(str(binary))} {q(str(prm))} '
          f'> {q(str(output/"replay.log"))} 2>&1')
    print('Leave interface/preconditioner experiment, reference-comparison, and profiling flags unset.')


if __name__=='__main__':main()
