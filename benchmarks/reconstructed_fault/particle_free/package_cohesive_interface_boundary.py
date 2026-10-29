#!/usr/bin/env python3
"""Validate and package the completed boundary experiment; never runs a solver."""
import ast
import csv
import difflib
import hashlib
import json
import math
from pathlib import Path
import py_compile
import shutil
import subprocess
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / 'benchmarks/reconstructed_fault/particle_free'
OUT = BENCH / 'evidence/cohesive_interface_boundary'
WORK = ROOT / 'build-particle-free/cohesive-interface-boundary'
DOC = ROOT / 'doc/reconstructed_fault/particle_free'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    assert Path.cwd() == ROOT
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
    assert branch == 'pf-rsf-particle-free'
    protected = json.loads((OUT / 'entry-protection.json').read_text())
    archives = json.loads((OUT / 'entry-archives.json').read_text())
    changed = [p for p, h in protected.items() if sha(ROOT / p) != h]
    old_changed = [p for p, h in archives.items() if sha(OUT.parent / p) != h]
    assert not changed and not old_changed, (changed, old_changed)
    write('preservation.json', dict(protected_file_count=len(protected),
          changed_protected_files=changed, prior_archive_count=len(archives),
          changed_archives=old_changed, production_BP3_unchanged=True))

    # Compare with task entry, not HEAD, because extensive prior local work exists.
    modified = []
    patch = []
    with zipfile.ZipFile(WORK / 'entry-source.zip') as z:
        for p in z.namelist():
            old = z.read(p).decode()
            new = (ROOT / p).read_text()
            if old != new:
                modified.append(p)
                patch.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                             fromfile='task-entry/' + p, tofile='current/' + p))
            if p.endswith(('STATUS.md', 'SOURCE_AUDIT.md')):
                assert new.endswith(old), p
            else:
                assert old == new or p.endswith('/cohesive_interface_native.cc'), p
    (OUT / 'source-entry.patch').write_text(''.join(patch))
    added = [str(p.relative_to(ROOT)) for p in sorted(BENCH.glob('*cohesive_interface_boundary.py'))]
    added += [str(p.relative_to(ROOT)) for p in sorted(DOC.glob('COHESIVE_INTERFACE_BOUNDARY_*.md'))]
    write('changed-files.json', dict(baseline_revision=revision, modified_since_task_entry=modified,
          added_for_task=added, preserved_preexisting_changes=True,
          note='Evidence files and package are additional generated task outputs.'))

    inputs = OUT / 'inputs'
    checks = OUT / 'native-checks'
    inputs.mkdir(exist_ok=True)
    checks.mkdir(exist_ok=True)
    costs = []
    for p in sorted(OUT.glob('*-run.json')):
        r = json.loads(p.read_text())
        label = p.name.removesuffix('-run.json')
        for suffix in ['.prm', '-environment.json', '-validate.log', '-aspect.log']:
            shutil.copy2(WORK / (label + suffix), inputs / (label + suffix))
        assert sha(inputs / (label + '.prm')) == r['parameter_sha256']
        raw = WORK / 'runs' / label / 'output'
        for f in raw.glob('*.csv'):
            if f.name in {'operator_checks.csv', 'raw_operator_checks.csv', 'operator_arithmetic.csv',
                          'field_checks.csv', 'trace_checks.csv', 'derivative_checks.csv',
                          'derivative-extended.csv', 'lifecycle.csv', 'restart_loaded.csv',
                          'loading-plan-extension.csv', 'predictor-fallback.csv',
                          'rejected-contact-proposals.csv'}:
                target = checks / label
                target.mkdir(exist_ok=True)
                shutil.copy2(f, target / f.name)
        setup = raw / 'operator_checks.csv'
        costs.append(dict(command=label, wall_seconds=r['seconds'],
                          setup_seconds=float(rows(setup)[0]['setup_seconds']) if setup.exists() else None,
                          observed_RSS_MiB=r['observed_peak_RSS_MiB'], exit_code=r['exit_code'],
                          stop=r['stop'], timeout_seconds=r['timeout_s'],
                          RSS_note='Process VmHWM sampled every 5 s; short failed jobs under-sampled.'))
    write('cost-summary.json', dict(commands=costs, total_native_seconds=sum(r['wall_seconds'] for r in costs),
          runs_overlapped=False, rank_count=1, thread_count=1, compilation_analysis_excluded=True))
    for name in ['configure.log', 'build.log']:
        shutil.copy2(WORK / name, checks / name)

    selected = []
    provenance = {}
    for name in ['longer', 'event-half']:
        data = rows(OUT / (name + '-history.csv'))
        clocks = [float(r['time_s']) for r in data]
        assert all(b > a for a, b in zip(clocks, clocks[1:]))
        for t in [2.45, 2.50, 2.55, 2.58]:
            r = min(data, key=lambda r: abs(float(r['time_s']) - t))
            assert abs(float(r['time_s']) - t) < 1e-10
            selected.append(r)
        chain = ['build-particle-free/cohesive-interface-domain/runs/period-16/output',
                 'build-particle-free/cohesive-interface-boundary/runs/longer/output']
        chain += ['build-particle-free/cohesive-interface-boundary/runs/' +
                  ('longer-completion' if name == 'longer' else 'event-half') + '/output']
        provenance[name] = dict(logical_accepted_states=len(data), source_chain=chain,
              merge_rule='At each restart retain preceding accepted times through restart_loaded clock; '
                         'take new states after that clock. Never interpolate histories.',
              load_end_s=2.55, end_s=2.58)
    with (OUT / 'summary.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(selected[0]))
        w.writeheader()
        w.writerows(selected)
    checkpoints = []
    for p in sorted((WORK / 'preserved').rglob('*')):
        if p.is_file():
            checkpoints.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size))
    write('trajectory-provenance.json', dict(trajectories=provenance,
          separately_retained_checkpoint_files=checkpoints,
          raw_checkpoints_in_compact_archive=False,
          field_rule='Use native snapshot only from its own accepted branch segment.'))

    qualification = json.loads((OUT / 'continuation-verification.json').read_text())
    assert all(qualification['gates'].values())
    assert all(qualification[k]['publications_once'] for k in ['longer', 'half'])
    assert json.loads((OUT / 'reflection-verification.json').read_text())['passed']
    assert json.loads((OUT / 'post-hold-extension-rejected-run.json').read_text())['expected_failure_passed']
    csv_checks = {}
    for p in sorted(OUT.glob('*.csv')):
        data = rows(p)
        for r in data:
            assert None not in r, p
            for value in r.values():
                assert value is not None, p
                try:
                    v = float(value)
                except ValueError:
                    continue
                assert math.isfinite(v), (p, r)
        csv_checks[p.name] = len(data)
    images = {}
    for p in sorted(OUT.glob('*')):
        if p.suffix not in ['.png', '.gif']:
            continue
        with Image.open(p) as im:
            images[p.name] = dict(width=im.width, height=im.height, frames=getattr(im, 'n_frames', 1))
            im.verify()
    assert images['longer-evolution.gif']['frames'] > 20
    write('artifact-verification.json', dict(csv_rows=csv_checks, images=images,
          declared_gates_passed=True, previous_evidence_preserved=True))

    # Small source snapshot with local Python dependency closure. External packages,
    # existing ASPECT build and raw solver archives are intentionally not bundled.
    sources = {ROOT / p for p in added}
    sources.update(ROOT / p for p in modified)
    sources.update(BENCH.glob('cohesive_interface_*.inc'))
    sources.update([BENCH / 'cohesive_interface_evolution.h', BENCH / 'cohesive_interface_native.prm',
                    BENCH / 'CMakeLists.txt', ROOT / 'include/aspect/reconstructed_fault/cohesive_interface.h',
                    ROOT / 'source/reconstructed_fault/cohesive_interface.cc'])
    queue = [p for p in sources if p.suffix == '.py']
    while queue:
        p = queue.pop()
        py_compile.compile(str(p), doraise=True)
        for node in ast.walk(ast.parse(p.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                    [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                for directory in [BENCH, BENCH / 'bulk_cohesive']:
                    dep = directory / (name.replace('.', '/') + '.py')
                    if dep.is_file() and dep not in sources:
                        sources.add(dep)
                        queue.append(dep)
    files = sources | {p for p in OUT.rglob('*') if p.is_file() and not p.is_symlink()
                       and not any(q.is_symlink() for q in p.parents)}
    files -= {OUT / 'manifest.json', OUT / 'checksums.sha256'}
    records = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(files)}
    write('manifest.json', dict(revision=revision, branch=branch, files=records,
          native_plugin_sha256=sha(WORK / 'build/libcohesive_interface_native.release.so'),
          purpose='Compact evidence and source snapshot; existing checkout/build and raw solver data required for reruns.'))
    files.add(OUT / 'manifest.json')
    (OUT / 'checksums.sha256').write_text(''.join(sha(p) + '  ' + str(p.relative_to(ROOT)) + '\n' for p in sorted(files)))
    files.add(OUT / 'checksums.sha256')
    archive = OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(files):
            z.write(p, str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    archive.with_suffix('.zip.sha256').write_text(sha(archive) + '  ' + archive.name + '\n')
    print(json.dumps(dict(archive=str(archive.relative_to(ROOT)), bytes=archive.stat().st_size,
                         files=len(files), preservation_passed=True, gates=qualification['gates']), indent=2))


if __name__ == '__main__':
    main()
