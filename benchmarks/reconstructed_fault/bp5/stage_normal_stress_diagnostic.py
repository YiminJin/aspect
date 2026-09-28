"""Copy a coherent checkpoint into a new diagnostic branch; never launch ASPECT.

The input must be the actual run's fully resolved parameters.prm (or a flat
production input), and --job is the directory from which that run was launched.
No archive clocks or histories are edited, and no writable hard links are used.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def parameters(text):
    stack, result = [], {}
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip()
        if line.startswith('include '):
            raise ValueError('Use the resolved parameters.prm; includes must not be guessed')
        if line.startswith('subsection '):
            stack.append(line[len('subsection '):])
        elif line == 'end':
            stack.pop()
        elif line.startswith('set '):
            key, value = line[4:].split('=', 1)
            result[tuple(stack+[key.strip()])] = value.strip()
    if stack:
        raise ValueError('Unbalanced input')
    return result


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stage(args):
    checkpoint, job, dest = args.checkpoint.resolve(), args.job.resolve(), args.destination.resolve()
    step_text, time_text = (checkpoint/'bp3_accepted_state.txt').read_text().split()
    step, time = int(step_text), float(time_text)
    if not args.local_verification and (step != 5612 or time != 5310111071.5634108):
        raise ValueError('Expected accepted step 5612 at 5310111071.5634108 s; select the correct checkpoint')
    if dest.exists() or dest == checkpoint or checkpoint in dest.parents:
        raise ValueError('Use a new independent staging directory outside the checkpoint')
    for name in ('resume.z', 'mesh'):
        if not any(checkpoint.glob(name+'*')):
            raise ValueError(f'Incomplete checkpoint: missing {name}')
    original = args.input.read_text()
    config = parameters(original)
    if float(config[('Time stepping','BP5 state startup','Maximum logarithmic state change')]) != 0.02:
        raise ValueError('Expected unchanged 0.02 startup limiter')
    if config[('Material model','Phase field fault','Use adiabatic pressure in fault friction')] != 'false':
        raise ValueError('Expected true normal-stress feedback')
    if config[('Material model','Phase field fault','I h surface quadrature subdivisions')] != '8':
        raise ValueError('Expected the qualified eight-panel production input')
    source_hashes = {str(p.relative_to(checkpoint)): sha(p) for p in checkpoint.rglob('*') if p.is_file()}
    dest.mkdir(parents=True)
    shutil.copytree(checkpoint, dest/'output-normal-diagnostic/restart/01')
    (dest/'output-normal-diagnostic/restart/last_good_checkpoint.txt').write_text('1\n')
    for name, digest in source_hashes.items():
        assert sha(dest/'output-normal-diagnostic/restart/01'/name) == digest
    metadata = checkpoint/'bp3_output_metadata'
    if metadata.is_dir():
        for p in metadata.iterdir():
            if p.is_file():
                shutil.copy2(p, dest/'output-normal-diagnostic'/p.name)
    # The first resumed dt is already in the checkpoint. If available, retain
    # its preceding controller record; do not recompute or replace that clock.
    original_output=checkpoint.parent.parent
    for name in ('timestep_selection.csv','state_startup_predictor.csv'):
        source=original_output/name
        if source.is_file():
            with source.open(newline='') as stream:
                reader=csv.DictReader(stream)
                # The production predictor writes its header only at step 0;
                # a prior restart into an empty directory may be headerless.
                if 'accepted_step' not in (reader.fieldnames or []):
                    if name!='state_startup_predictor.csv' or len(reader.fieldnames or [])!=6:
                        raise ValueError(f'Unknown controller record columns: {source}')
                    stream.seek(0)
                    reader=csv.DictReader(stream,fieldnames=('accepted_step','time','maximum_dt','proposed_dt','measure','limit'))
                rows=[r for r in reader if int(r['accepted_step'])<=step]
                columns=reader.fieldnames
            with (dest/'output-normal-diagnostic'/name).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=columns)
                writer.writeheader();writer.writerows(rows)
    (dest/'production_input.prm').write_text(original)
    template = Path(__file__).with_name('normal_stress_diagnostic_restart.prm').read_text()
    template = template.replace('= 5612', f'= {step}').replace('= 5310111071.5634108', f'= {time:.17g}')
    template = template.replace('set New accepted steps = 5', f'set New accepted steps = {args.steps}')
    if args.control:
        template = template.replace('set Capture stress split = true', 'set Capture stress split = false')
    # Inline the unchanged input literally. ASPECT's regex-based include
    # expansion interprets $0/$& inside resolved-parameter documentation as
    # replacement expressions; embedding avoids that unrelated parser issue.
    template = template.replace('include production_input.prm',original)
    # Copy exact scientific inputs; override paths only, retaining all contents.
    overrides=[]
    for key, name in [(('Fault reconstruction','Prescribed faults file'),'fault.txt'),
                      (('Mesh refinement','BP3 saved mesh','Target cells file'),'target_cells.txt'),
                      (('Postprocess','BP3','Bottom normalization completion file'),'completion.txt')]:
        source=Path(config[key]); source=source if source.is_absolute() else job/source
        target=dest/'fixture'/name; target.parent.mkdir(exist_ok=True)
        shutil.copy2(source,target)
        prefix=''.join('subsection '+s+'\n' for s in key[:-1])
        overrides.append(prefix+f'set {key[-1]} = fixture/{name}\n'+'end\n'*(len(key)-1))
    if config.get(('Postprocess','BP3','Mature prestress file'),''):
        raise ValueError('Loading-driven steady initializer must not reload captured prestress')
    libraries=[]
    for library in config[('Additional shared libraries',)].split(','):
        source=Path(library.strip())
        if 'libbp5_normal_stress_diagnostic' in source.name:
            continue  # Always install the explicitly supplied observer below.
        if 'libbp5_steady_initialization' in source.name:
            source=args.bp5_library.resolve()
        elif not source.is_absolute():
            source=job/source
        shutil.copy2(source,dest/source.name); libraries.append('./'+source.name)
    if not any('libbp5_steady_initialization' in p for p in libraries):
        raise ValueError('Production BP5 plugin is missing')
    source=args.diagnostic_library.resolve()
    shutil.copy2(source,dest/source.name); libraries.append('./'+source.name)
    overrides.append('set Additional shared libraries = '+', '.join(libraries)+'\n')
    # Preserve any additional required postprocessors from the actual run.
    post=', '.join(p.strip() for p in config[('Postprocess','List of postprocessors')].split(',')
                   if p.strip()!='BP5 normal diagnostic')
    overrides.append('subsection Postprocess\nset List of postprocessors = '+post+', BP5 normal diagnostic\nend\n')
    (dest/'normal_stress_diagnostic_restart.prm').write_text(template+'\n'+'\n'.join(overrides))
    manifest=dict(checkpoint=str(checkpoint), checkpoint_step=step, checkpoint_time=time,
                  original_input=str(args.input.resolve()), original_input_sha256=sha(args.input),
                  checkpoint_sha256=source_hashes, control=args.control, new_steps=args.steps,
                  staged_input_sha256={str(p.relative_to(dest)):sha(p) for p in (dest/'fixture').iterdir()},
                  libraries={name:sha(dest/name[2:]) for name in libraries})
    (dest/'staging.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Staged unchanged accepted step {step}, time {time:.17g}; no simulation launched.\n{dest}')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','input','job','destination','bp5-library','diagnostic-library'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=5,choices=range(1,9))
    p.add_argument('--control',action='store_true')
    p.add_argument('--local-verification',action='store_true',help='Allow a small local checkpoint instead of server step 5612')
    stage(p.parse_args())
