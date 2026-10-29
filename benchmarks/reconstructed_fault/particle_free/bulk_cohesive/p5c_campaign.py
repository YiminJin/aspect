#!/usr/bin/env python3
"""Selected native continuation and read-only accepted-state observation.

Native ASPECT owns solves, acceptance, histories and checkpoints. This driver
copies verified checkpoints, launches bounded stages, and requests the existing
collective user-request termination criterion after diagnostic events.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import zlib
import numpy as np
from p5c_campaign_events import Events, write_rows

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT/'build-particle-free/p5c-campaign'
AXIAL = ROOT/'build-particle-free/p5c-axial'
OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/p5c_continuation'
WRAPPER = ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py'
BINARY = ROOT/'build-particle-free/p5c-native/aspect-release'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def event_reason(row, events):
    """A matched replay ignores its target, never a simultaneous safety stop."""
    if not events:
        return ''
    if events == 'replay':
        return ('remote_nucleation' if row['significant_remote_nucleation']
                else 'boundary_approach' if row['boundary_approach'] else '')
    return row['stop_reason']


def metadata(directory, mode):
    raw = zlib.decompress((directory/'resume.z').read_bytes()[16:])
    tag = f'bulk cohesive {mode} tensor cell history v1'.encode()
    q, step, publications, physical_time = struct.unpack_from('<IIQd', raw, raw.index(tag)+len(tag))
    assert q == 8 and step == publications
    return {'quadrature': q, 'step': step, 'publications': publications, 'time_s': physical_time}


def checkpoint(run, mode, step=None):
    pointer = run/'restart/last_good_checkpoint.txt'
    number = int(pointer.read_text())
    paths = [run/'restart'/f'{number:02}', *sorted((run/'restart').glob('[0-9][0-9]'))]
    for path in dict.fromkeys(paths):
        try:
            m = metadata(path, mode)
            if step is None or m['step'] == step:
                return path, m
        except (zlib.error, ValueError, FileNotFoundError):
            continue  # Only the ring entry currently being written may be partial.
    raise RuntimeError(f'No complete {mode} checkpoint at step {step} in {run}')


def preserve(run, mode, label, step=None):
    destination = WORK/'checkpoints'/f'{mode}-{label}'
    if destination.exists():
        return destination
    source, m = checkpoint(run, mode, step)
    partial = destination.with_name(destination.name+'.partial')
    target = partial/'restart'/source.name
    target.parent.mkdir(parents=True)
    hashes = {p.name: sha(p) for p in source.iterdir() if p.is_file()}
    shutil.copytree(source, target)
    for name, digest in hashes.items():
        assert sha(target/name) == digest == sha(source/name)
    (partial/'restart/last_good_checkpoint.txt').write_text(str(int(source.name))+'\n')
    phase = run/f'phase-{m["step"]:04}.csv'
    if phase.exists():
        shutil.copy2(phase, partial/'phase.csv')
    m.update(source=str(source.relative_to(ROOT)), mode=mode, files=hashes)
    save(partial/'metadata.json', m)
    partial.rename(destination)
    return destination


def initialize():
    assert not (WORK/'entry.json').exists(), 'Preserve the existing campaign'
    WORK.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    old = json.loads((AXIAL/'entry.json').read_text())
    protected = {name: sha(ROOT/name) for name in old['protected']}
    assert protected == old['protected']
    artifacts = [BINARY, ROOT/'build-particle-free/p5c-native/plugin/libbulk_p5c_checks.release.so',
                 ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/p5c_native.zip',
                 ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/p5c_axial.zip']
    save(WORK/'entry.json', {'revision': old['revision'], 'protected': protected,
                            'artifacts': {str(p.relative_to(ROOT)): sha(p) for p in artifacts},
                            'new_plugin_sha256': sha(WORK/'plugin/libbulk_p5c_campaign_checks.release.so')})
    for mode in ['fixed', 'radial']:
        run = AXIAL/'runs'/f'{mode}-active'
        parent = preserve(run, mode, 'initial127', 127)
        m = json.loads((parent/'metadata.json').read_text())
        assert abs(m['time_s']-.13875) < 1e-12
        shutil.copy2(AXIAL/'runs'/f'{mode}-prefix'/'phase-0000.csv', WORK/f'{mode}-initial.csv')
    # Test the detector on analytic synthetic fields, never used by mechanics.
    detector = Events(WORK/'radial-initial.csv'); p = detector.initial.copy()
    zero = detector.measure(p, p, .13875, 127)
    assert not zero['stop_reason'] and not zero['intact_activation']
    target = p.copy(); mask = (abs(detector.Y-.5)<.02)&(detector.X>=1.04)&(detector.X<=1.1484375)
    target[mask] = np.maximum(target[mask], .06)
    assert detector.measure(target, p, .14, 129)['target_reached']
    remote = p.copy(); remote[(abs(detector.X-1.35)<.012)&(abs(detector.Y-.8)<.012)] = .02
    assert detector.measure(remote, p, .14, 129)['significant_remote_nucleation']
    save(OUT/'detector-controls.json', {'zero_growth': zero, 'synthetic_connected_target': True, 'synthetic_remote_component': True})
    save(WORK/'state.json', {'phase': 'prepared', 'runs': [], 'modes': {}})


def stage(mode, name, parent, end_time, dt=.000625, loading_end=.2, events=True, reference_run=None):
    state = json.loads((WORK/'state.json').read_text())
    parent_meta = json.loads((parent/'metadata.json').read_text())
    parent_step, parent_time = parent_meta['step'], parent_meta['time_s']
    intervals = round((end_time-parent_time)/dt)
    assert intervals > 0 and abs(parent_time+intervals*dt-end_time) < 1e-12
    end_step = parent_step+intervals
    run = WORK/'runs'/name
    assert not run.exists(), f'Preserve {run}'
    shutil.copytree(parent/'restart', run/'restart')
    shutil.copy2(parent/'phase.csv', run/f'phase-{parent_step:04}.csv')
    prm = f'''include $ASPECT_SOURCE_DIR/build-particle-free/p5c-axial/{mode}-active.prm
set Additional shared libraries = $ASPECT_SOURCE_DIR/build-particle-free/p5c-campaign/plugin/libbulk_p5c_campaign_checks.release.so
set Output directory = $ASPECT_SOURCE_DIR/build-particle-free/p5c-campaign/runs/{name}
set End time = 0.210
set Resume computation = true
set Maximum time step = {dt:.17g}
set Maximum first time step = {dt:.17g}
subsection Boundary velocity model
 set Prescribed velocity boundary indicators = left y:p5c campaign loading, right y:p5c campaign loading, bottom:p5c campaign loading, top:p5c campaign loading
end
subsection Checkpointing
 set Steps between checkpoint = 1
end
subsection Termination criteria
 set Termination criteria = end step, user request
 set End step = {end_step}
end
'''
    input_path = WORK/f'{name}.prm'; input_path.write_text(prm)
    command = ['python3', str(WRAPPER), 'p5c-campaign-'+name, '21600', 'env',
               'ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1', 'ASPECT_BULK_COMPACT_OUTPUT=1',
               f'ASPECT_P5C_LOADING_END={loading_end:.17g}', '/opt/openmpi/5.0.6/bin/mpirun', '-np', '4',
               str(BINARY), str(input_path)]
    process = subprocess.Popen(command, cwd=ROOT)
    state['phase'] = 'running '+name; state['process'] = process.pid
    save(WORK/'state.json', state)
    detector = Events(WORK/f'{mode}-initial.csv')
    previous = detector.phase(parent/'phase.csv')
    seen = parent_step; rows = []; requested = None
    mode_state = state['modes'].setdefault(mode, {})
    try:
        while True:
            resources = run/'resources.csv'
            if resources.exists():
                records = list(csv.DictReader(resources.open()))
                for resource in records:
                    if not resource.get('freeze') or int(resource['step']) <= seen:
                        continue
                    step = int(resource['step']); t = float(resource['time'])
                    assert step == seen+1, (seen, step)
                    assert resource['freeze'] == '1' and resource['particles'] == '221184'
                    assert resource['particle_fingerprint'] == '13847991597444700578'
                    phi = detector.phase(run/f'phase-{step:04}.csv')
                    row = detector.measure(phi, previous, t, step)
                    row['loading_displacement_m'] = .018*(min(t, loading_end)-.05)
                    if reference_run is not None:
                        for prefix in ['phase', 'means', 'phase-forces', 'boundary-reactions']:
                            assert sha(run/f'{prefix}-{step:04}.csv') == sha(reference_run/f'{prefix}-{step:04}.csv'), (name, prefix, step)
                    if step in [128, 129] and dt == .000625:
                        for prefix in ['phase', 'means', 'phase-forces', 'boundary-reactions']:
                            assert sha(run/f'{prefix}-{step:04}.csv') == sha(AXIAL/'runs'/f'{mode}-active'/f'{prefix}-{step:04}.csv'), (name, prefix, step)
                    for flag, key in [('intact_activation', 'first_intact'), ('rapid_growth', 'first_rapid')]:
                        if row[flag] and key not in mode_state:
                            if seen == parent_step:
                                saved = WORK/'checkpoints'/f'{mode}-{key}'
                                shutil.copytree(parent, saved)
                            else:
                                saved = preserve(run, mode, key, seen)
                            mode_state[key] = {'bracket_s': [t-dt, t], 'checkpoint': str(saved.relative_to(ROOT)), 'event_step': step}
                    stop_event = event_reason(row, events)
                    if stop_event and requested is None:
                        requested = {'step': step, 'time_s': t, 'reason': stop_event}
                        (run/'terminate-aspect').touch()
                    rows.append(row); previous = phi; seen = step
                    write_rows(run/'events.csv', rows)
                    save(WORK/'progress.json', {'name': name, 'mode': mode, **row})
                    save(WORK/'state.json', state)
                    print(json.dumps({'run': name, 'step': step, 'time_s': t,
                          'peak_increment': row['peak_increment'], 'max_step_increment': row['max_step_increment'],
                          'front_phi05_m': row['axial_front_phi0.05_m'], 'virgin_peak': row['initially_zero_cell_peak'],
                          'stop_requested': requested}), flush=True)
            code = process.poll()
            if code is not None:
                # A final resource row can become visible concurrently with exit.
                if resources.exists() and records and int(records[-1]['step']) > seen:
                    continue
                break
            time.sleep(1)
    except BaseException:
        (run/'terminate-aspect').touch()
        process.wait(timeout=600)
        raise
    final = preserve(run, mode, name+'-terminal')
    final_meta = json.loads((final/'metadata.json').read_text())
    if requested and code == 0:
        assert final_meta['step'] <= requested['step']+1
    record = {'name': name, 'mode': mode, 'dt_s': dt, 'loading_end_s': loading_end,
              'planned_end_s': end_time, 'actual_end_s': final_meta['time_s'], 'last_publication': final_meta['publications'],
              'exit_code': code, 'requested_stop': requested, 'checkpoint': str(final.relative_to(ROOT)),
              'parent': str(parent.relative_to(ROOT))}
    if reference_run is not None:
        record['bitwise_reference_run'] = str(reference_run.relative_to(ROOT))
        record['bitwise_compared_exports'] = 4*len(rows)
    state['runs'].append(record); state['process'] = None; state['phase'] = 'completed '+name
    save(WORK/'state.json', state); save(OUT/'campaign-state.json', state)
    print(json.dumps(record), flush=True)
    return record, final


def radial():
    parent = WORK/'checkpoints/radial-initial127'
    for end in [.155, .175, .200]:
        record, parent = stage('radial', f'radial-to{round(end*1000):03}', parent, end)
        # Native stage endpoint is an observation only: no loading interruption.
        save(OUT/f'radial-{round(end*1000):03}-summary.json', record)
        if record['exit_code'] or record['requested_stop']:
            break
    save(WORK/'radial-loading-end.json', record)


def resume_radial():
    """Continue the verified shutdown checkpoint without a physical hold."""
    from p5c_campaign_analysis import intermediate
    pause = json.loads((WORK/'shutdown-checkpoint.json').read_text())
    parent = ROOT/pause['checkpoint']
    incoming = json.loads((parent/'metadata.json').read_text())
    assert incoming['step'] == incoming['publications'] == 153
    assert abs(incoming['time_s']-.155) < 1e-12
    for end in [.175, .200]:
        record, parent = stage('radial', f'radial-resume-to{round(end*1000):03}', parent, end)
        save(OUT/f'radial-{round(end*1000):03}-summary.json', record)
        intermediate()
        if record['exit_code'] or record['requested_stop']:
            break
    save(WORK/'radial-loading-end.json', record)


def capture_rapid_checkpoints():
    """Supplement the first-event copy with recoverable recent rapid events."""
    saved = []
    for run in sorted((WORK/'runs').glob('*')):
        if run.name.startswith(('replay-', 'onset-')) or not (run/'events.csv').exists():
            continue
        mode = 'radial' if run.name.startswith('radial') else 'fixed'
        for row in csv.DictReader((run/'events.csv').open()):
            if row['rapid_growth'] != 'True':
                continue
            number = int(row['step']); label = f'rapid-step{number:04}'
            if (WORK/'checkpoints'/f'{mode}-{label}').exists():
                continue
            try:
                saved.append(str(preserve(run, mode, label, number-1).relative_to(ROOT)))
            except RuntimeError:
                pass  # The first-rapid copy remains; old ring entries are not invented.
    return saved


def recover_and_complete():
    """Recover the event checkpoint lost when the interactive observer died.

    The native trajectory survived that interruption. Retain its extra states
    as diagnostics, reproduce its intended prefix exactly, and never turn the
    unobserved extra loading into a new physical comparison endpoint.
    """
    from p5c_campaign_analysis import stage_summary, intermediate, final
    orphan = WORK/'runs/radial-resume-to175'
    command_record = ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence/p5c-campaign-radial-resume-to175.json'
    deadline = time.monotonic()+900
    (orphan/'terminate-aspect').touch()
    while not command_record.exists():
        assert time.monotonic() < deadline, 'Original native run has not stopped; do not launch another run.'
        time.sleep(1)
    detector = Events(WORK/'radial-initial.csv')
    parent = WORK/'checkpoints/radial-user-shutdown'
    previous = detector.phase(parent/'phase.csv'); rows = []
    for resource in csv.DictReader((orphan/'resources.csv').open()):
        assert resource['freeze'] == '1'
        phi = detector.phase(orphan/f'phase-{int(resource["step"]):04}.csv')
        rows.append(detector.measure(phi, previous, float(resource['time']), int(resource['step'])))
        previous = phi
    write_rows(orphan/'events-recovered.csv', rows)
    first_stop = next(row for row in rows if row['stop_reason'])
    retained = preserve(orphan, 'radial', 'interrupted-observer-extra-terminal')
    meta = json.loads((retained/'metadata.json').read_text())
    state = json.loads((WORK/'state.json').read_text())
    diagnostic = {'name': orphan.name, 'mode': 'radial', 'actual_end_s': meta['time_s'],
                  'last_publication': meta['publications'], 'checkpoint': str(retained.relative_to(ROOT)),
                  'exit_code': json.loads(command_record.read_text())['exit_code'],
                  'first_event': first_stop,
                  'scope': 'Observer lost on intentional turn interruption; native extra continuation retained, excluded from selected endpoint/control.'}
    state.setdefault('interrupted_native_runs', []).append(diagnostic)
    save(WORK/'state.json', state); save(OUT/'observer-interruption.json', diagnostic)
    record, recovered = stage('radial', 'radial-recovered-target', parent, first_stop['time_s'],
                              reference_run=orphan)
    save(WORK/'radial-loading-end.json', record)
    intermediate()
    finish_controls()
    select_replays()
    replay_plan()
    final()
    from p5c_campaign_temporal import main as temporal
    temporal()


def start_recovery():
    """Keep the event observer alive across an interactive turn interruption."""
    log = WORK/'controller-recovery.log'
    assert not log.exists(), 'Inspect the existing controller before starting another one.'
    with log.open('w') as stream:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'recover-complete'],
                                 cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    save(WORK/'controller.json', {'pid': child.pid, 'action': 'recover-complete',
                                'log': str(log.relative_to(ROOT))})
    print(json.dumps({'controller_pid': child.pid, 'log': str(log)}, indent=2))


def watch_replays():
    """Protect replays in the already-running, pre-correction Python process.

    This read-only observer uses the same native collective user-request stop;
    future controllers use event_reason directly. It never launches a solve.
    """
    deadline = time.monotonic()+43200
    while not (OUT/'temporal-summary.json').exists():
        assert time.monotonic() < deadline, 'Replay safety observer exceeded its bounded lifetime.'
        state = json.loads((WORK/'state.json').read_text())
        phase = state['phase']
        if phase.startswith(('running replay-', 'running onset-')):
            run = WORK/'runs'/phase.removeprefix('running ')
            path = run/'events.csv'
            if path.exists():
                rows = list(csv.DictReader(path.open()))
                if rows:
                    row = rows[-1]
                    reason = event_reason({key: row.get(key) == 'True' for key in
                                           ['significant_remote_nucleation', 'boundary_approach']}, 'replay')
                    if reason:
                        (run/'terminate-aspect').touch()
                        record = run/'replay-safety-stop.json'
                        if not record.exists():
                            save(record, {'reason': reason, 'step': row['step'], 'time_s': row['time_s']})
                            print(record.read_text(), flush=True)
        time.sleep(1)
    print('All native replays complete; safety observer finished.', flush=True)


def start_replay_watch():
    log = WORK/'replay-safety-observer.log'
    assert not log.exists(), 'Preserve the existing safety observer.'
    with log.open('w') as stream:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'watch-replays'],
                                 cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    save(WORK/'replay-safety-observer.json', {'pid': child.pid, 'log': str(log.relative_to(ROOT))})
    print('Started independent replay safety observer:', child.pid)


def finish_controls():
    """Matched fixed loading and each admissible endpoint's physical hold."""
    from p5c_campaign_analysis import intermediate
    primary = json.loads((WORK/'radial-loading-end.json').read_text())
    radial_end = primary['actual_end_s']
    state = json.loads((WORK/'state.json').read_text())
    state['loading_endpoints'] = {'radial': radial_end}
    save(WORK/'state.json', state)
    radial_parent = ROOT/primary['checkpoint']
    stage('radial', 'radial-hold', radial_parent, radial_end+.005,
          loading_end=radial_end, events=False)
    intermediate()
    parent = WORK/'checkpoints/fixed-initial127'
    previous_time = .13875
    endpoints = sorted(set([t for t in [.155, .175] if t < radial_end-1e-12]+[radial_end]))
    for end in endpoints:
        if end <= previous_time+1e-12:
            continue
        record, parent = stage('fixed', f'fixed-to{round(end*1e6):06}', parent, end, loading_end=radial_end)
        intermediate(); previous_time = record['actual_end_s']
        if record['exit_code'] or record['requested_stop']:
            break
    state = json.loads((WORK/'state.json').read_text())
    state['loading_endpoints']['fixed'] = previous_time
    save(WORK/'state.json', state)
    stage('fixed', 'fixed-hold', parent, previous_time+.005,
          loading_end=previous_time, events=False)
    intermediate()


def replay_plan():
    """Run the explicitly recorded local windows, without changing histories."""
    from p5c_campaign_analysis import intermediate
    plan = json.loads((WORK/'replay-plan.json').read_text())
    for item in plan:
        stage(item['mode'], item['name'], ROOT/item['parent'], item['end_s'],
              dt=.0003125, loading_end=item['loading_end_s'], events='replay')
        intermediate()


def select_replays():
    """Select local windows from actual accepted events, not a parameter search."""
    state = json.loads((WORK/'state.json').read_text()); capture_rapid_checkpoints()
    plan = []
    for mode in ['radial', 'fixed']:
        records = [r for r in state['runs'] if r['mode'] == mode and not r['name'].startswith(('replay-', 'onset-'))]
        events = []
        for record in records:
            path = WORK/'runs'/record['name']/'events.csv'
            if path.exists():
                events += [dict(row, source_run=record['name']) for row in csv.DictReader(path.open())]
        if not events:
            continue
        principal = max(events, key=lambda r: float(r['max_step_increment']))
        event_time, event_step = float(principal['time_s']), int(principal['step'])
        try:
            parent = preserve(WORK/'runs'/principal['source_run'], mode, 'pre-principal', event_step-1)
        except RuntimeError:
            candidates = []
            for path in (WORK/'checkpoints').glob(f'{mode}-*'):
                if (path/'metadata.json').exists():
                    meta = json.loads((path/'metadata.json').read_text())
                    if meta['time_s'] < event_time-1e-12:
                        candidates.append((meta['time_s'], path))
            parent = max(candidates)[1]
        incoming = json.loads((parent/'metadata.json').read_text())['time_s']
        loading_end = state['loading_endpoints'][mode]
        available_end = max(r['actual_end_s'] for r in records)
        end = min(available_end, loading_end+.005 if loading_end <= event_time+.00125 else event_time+.00125)
        first = state['modes'].get(mode, {}).get('first_intact')
        if first and float(first['bracket_s'][1]) < incoming-1e-12:
            onset_end = min(first['bracket_s'][1]+.00125, available_end)
            plan.append({'mode': mode, 'name': 'onset-'+mode, 'parent': first['checkpoint'],
                         'end_s': onset_end, 'loading_end_s': loading_end,
                         'reason': 'short intact-activation check; same incoming tensor histories'})
        plan.append({'mode': mode, 'name': 'replay-'+mode, 'parent': str(parent.relative_to(ROOT)),
                     'end_s': end, 'loading_end_s': loading_end,
                     'principal_event_time_s': event_time, 'principal_max_step_increment': float(principal['max_step_increment']),
                     'reason': 'largest accepted phase increment; preserve physical loading/hold endpoint'})
    save(WORK/'replay-plan.json', plan); save(OUT/'replay-plan.json', plan)
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['initialize', 'radial', 'resume-radial', 'start-recovery', 'recover-complete', 'start-replay-watch', 'watch-replays', 'controls', 'select-replays', 'replay'])
    args = parser.parse_args()
    {'initialize': initialize, 'radial': radial, 'resume-radial': resume_radial,
     'start-recovery': start_recovery, 'recover-complete': recover_and_complete, 'controls': finish_controls,
     'start-replay-watch': start_replay_watch, 'watch-replays': watch_replays,
     'select-replays': select_replays, 'replay': replay_plan}[args.action]()
