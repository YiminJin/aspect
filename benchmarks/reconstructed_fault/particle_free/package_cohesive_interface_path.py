#!/usr/bin/env python3
"""Package the bounded causal-path experiment without duplicating raw kernels."""
import ast
import csv
import difflib
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image

from cohesive_interface_path import ROOT, WORK, OUT, PathKernel, save
from analyze_cohesive_interface_path import FINAL, V1

BENCH = ROOT / 'benchmarks/reconstructed_fault/particle_free'
DOC = ROOT / 'doc/reconstructed_fault/particle_free'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    assert Path.cwd() == ROOT
    branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    assert branch == 'pf-rsf-particle-free'
    protected = json.loads((OUT / 'entry-protection.json').read_text())
    archives = json.loads((OUT / 'entry-archives.json').read_text())
    assert all(sha(ROOT / p) == h for p, h in protected.items())
    assert all(sha(OUT.parent / p) == h for p, h in archives.items())
    save(OUT / 'preservation.json', dict(protected_files=len(protected), old_archives=len(archives),
         changed_protected_files=[], changed_archives=[], production_BP3_unchanged=True))

    modified, patch = [], []
    with zipfile.ZipFile(WORK / 'entry-source.zip') as archive:
        for name in archive.namelist():
            old, new = archive.read(name).decode(), (ROOT / name).read_text()
            if name.endswith(('STATUS.md', 'SOURCE_AUDIT.md')):
                assert new.endswith(old), name
            if old != new:
                modified.append(name)
                patch.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                             fromfile='task-entry/' + name, tofile='current/' + name))
    (OUT / 'source-entry.patch').write_text(''.join(patch))
    added = set(BENCH.glob('*cohesive_interface_path*.py'))
    added.add(BENCH / 'cohesive_interface_path.cc')
    added.update(DOC.glob('COHESIVE_INTERFACE_PATH_*.md'))
    save(OUT / 'changed-files.json', dict(revision=revision, modified_since_task_entry=modified,
         added_for_task=[str(p.relative_to(ROOT)) for p in sorted(added)],
         production_source_changes=False, evidence_files_additional=True))
    sources = {ROOT / p for p in modified} | added
    sources.update([ROOT / 'source/reconstructed_fault/cohesive_interface.cc',
                    ROOT / 'include/aspect/reconstructed_fault/cohesive_interface.h',
                    BENCH / 'cohesive_interface_evolution.h',
                    BENCH / 'cohesive_interface_turn_solver.cc'])
    queue = [p for p in sources if p.suffix == '.py']
    while queue:
        path = queue.pop()
        py_compile.compile(str(path), doraise=True)
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                dependency = BENCH / (name.replace('.', '/') + '.py')
                if dependency.is_file() and dependency not in sources:
                    sources.add(dependency)
                    queue.append(dependency)

    diagnostics = OUT / 'diagnostics'
    diagnostics.mkdir(exist_ok=True)
    for path in WORK.glob('*.log'):
        shutil.copy2(path, diagnostics / path.name)
    for name in ['entry-source.zip', 'entry-status.txt', 'c2-v1-source.zip']:
        shutil.copy2(WORK / name, diagnostics / name)
    for path in WORK.glob('*/failure.json'):
        dest = diagnostics / path.parent.name
        dest.mkdir(exist_ok=True)
        for name in ['failure.json', 'last-accepted.json', 'birth-failed-incoming.json']:
            source = path.parent / name
            if source.exists():
                shutil.copy2(source, dest / name)

    operators = OUT / 'operators'
    operators.mkdir(exist_ok=True)
    checkpoints = OUT / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    selected_kernels = {}
    for label in FINAL:
        dest = checkpoints / label
        dest.mkdir(exist_ok=True)
        names = ['initial.json', 'final.json', 'events.json']
        if label == 'work96':
            names += ['before-extension-3.json', 'after-extension-3.json']
        for name in names:
            source = WORK / label / name
            shutil.copy2(source, dest / name)
            if name == 'events.json':
                continue
            state = json.loads(source.read_text())
            key = state['kernel_folder']
            if key not in selected_kernels:
                kernel = PathKernel(WORK / key, state['geometry'], 'work')
                np.savez_compressed(operators / (key + '.npz'), K=kernel.K, Q=kernel.Q,
                    Nwork=kernel.N, raw_Nminus=kernel.raw_Nm, raw_Nplus=kernel.raw_Np,
                    x_m=kernel.s['x_m'], y_m=kernel.s['y_m'], weights_m=kernel.w,
                    tx=kernel.s['tx'], ty=kernel.s['ty'], nodes=kernel.nodes,
                    cell_center_x_m=kernel.centers['x_m'], cell_center_y_m=kernel.centers['y_m'])
                kernel.close()
                selected_kernels[key] = dict(raw_folder=str((WORK / key).relative_to(ROOT)))
            if label == 'work96' and name in ['initial.json', 'after-extension-3.json', 'final.json']:
                kernel = PathKernel(WORK / key, state['geometry'], 'work')
                coefficients = np.r_[state['wall_memory'], state['memory']]
                physical = np.einsum('ijk,j->ik', kernel.mean, coefficients)
                np.savez_compressed(operators / ('work96-' + name.removesuffix('.json') + '-physical.npz'),
                    centers_x_m=kernel.centers['x_m'], centers_y_m=kernel.centers['y_m'],
                    mean_tau_xx_yy_zz_xy_pprime_Pa=physical)
                kernel.close()
    save(operators / 'INDEX.json', dict(kernels=selected_kernels,
         note='Compact operators and actual physical fields; full basis tables remain in raw-kernel-index.json. '
              'Checkpoint JSON uses the isolated benchmark owner, not production ASPECT restart.'))
    shutil.copy2(WORK / 'geometry-checkpoint.json', checkpoints / 'geometry-checkpoint.json')

    raw, raw_bytes, kernel_seconds = {}, 0, 0.
    for folder in sorted(WORK.glob('kernel-*')):
        if not folder.is_dir():
            continue
        files = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
                 for p in folder.iterdir() if p.is_file()}
        raw_bytes += sum(item['bytes'] for item in files.values())
        metadata = json.loads((folder / 'native.json').read_text()) if (folder / 'native.json').exists() else None
        if metadata:
            kernel_seconds += metadata['seconds']
        raw[str(folder.relative_to(ROOT))] = dict(metadata=metadata, files=files)
    save(OUT / 'raw-kernel-index.json', raw)
    runs = [json.loads((OUT / (label + '-summary.json')).read_text()) for label in FINAL + V1]
    save(OUT / 'cost-summary.json', dict(completed_indexed_runs=len(runs),
         completed_indexed_evolution_seconds=sum(r['seconds'] for r in runs),
         final_control_seconds=sum(r['seconds'] for r in runs if r['label'] in FINAL),
         complete_native_kernel_seconds=kernel_seconds, raw_native_bytes=raw_bytes,
         owner_peak_RSS_MiB=max(r['peak_RSS_MiB'] for r in runs),
         native_peak_RSS_measured=False, sequential_timing_runs=True, BLAS_threads=1,
         note='Owner RSS is process high-water mark, not per-kernel or combined native peak. '
              'Evolution clocks exclude initial-kernel construction; include newly needed extension kernels '
              'and may reuse cached operators. Includes preserved superseded controls; not cold starts '
              'or a full-event performance extrapolation.'))
    lifecycle = json.loads((OUT / 'lifecycle.json').read_text())
    assert lifecycle['native_context_JSON_restart_bitwise']
    assert lifecycle['actual_native_contact_rejection_unchanged']
    assert lifecycle['correct_retry_after_failed_trial_bitwise']
    assessment = json.loads((OUT / 'assessment.json').read_text())
    assert not assessment['all_selected_practical_targets_pass']
    assert assessment['gates']['extension-step']['integral_growth']
    counts, images = {}, {}
    for path in OUT.glob('*.csv'):
        with path.open() as stream:
            data = list(csv.DictReader(stream))
        for row in data:
            assert None not in row
            for value in row.values():
                assert value is not None
                try:
                    number = float(value)
                except ValueError:
                    continue
                assert math.isfinite(number), (path, value)
        counts[path.name] = len(data)
    for path in sorted(OUT.iterdir()):
        if path.suffix not in ['.png', '.gif']:
            continue
        with Image.open(path) as image:
            images[path.name] = dict(width=image.width, height=image.height,
                                     frames=getattr(image, 'n_frames', 1))
            image.verify()
    assert images['successive-deflection.gif']['frames'] > 10
    save(OUT / 'artifact-verification.json', dict(csv_rows=counts, images=images,
         python=sys.version, packages={p: importlib.metadata.version(p)
         for p in ['numpy', 'scipy', 'matplotlib', 'Pillow']}, checks_passed=True))
    subprocess.run(['git', 'diff', '--check', '--', str(BENCH / 'CMakeLists.txt'),
                    str(DOC / 'STATUS.md'), str(DOC / 'SOURCE_AUDIT.md')], check=True)
    files = sources | {p for p in OUT.rglob('*') if p.is_file() and not p.is_symlink()}
    files -= {OUT / 'manifest.json', OUT / 'checksums.sha256'}
    save(OUT / 'manifest.json', dict(revision=revision, branch=branch,
         purpose='Compact source/evidence for the matching ASPECT worktree, not a standalone '
                 'distribution or production checkpoint. Historical failures and raw indices retained.',
         files={str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=sha(p))
                for p in sorted(files)}))
    files.add(OUT / 'manifest.json')
    (OUT / 'checksums.sha256').write_text(''.join(sha(p) + '  ' + str(p.relative_to(ROOT)) + '\n'
                                                for p in sorted(files)))
    files.add(OUT / 'checksums.sha256')
    archive = OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for path in sorted(files):
            package.write(path, str(path.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
    archive.with_suffix('.zip.sha256').write_text(sha(archive) + '  ' + archive.name + '\n')
    print(json.dumps(dict(archive=str(archive.relative_to(ROOT)), bytes=archive.stat().st_size,
                         files=len(files), protected=len(protected), old_archives=len(archives)), indent=2))


if __name__ == '__main__':
    main()
