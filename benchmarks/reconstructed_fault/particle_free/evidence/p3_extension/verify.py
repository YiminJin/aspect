#!/usr/bin/env python3
"""Replay the compact P3 native-component evidence without an ASPECT build."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import numpy as np

here = Path(__file__).resolve().parent
root = here.parents[4]
spec = importlib.util.spec_from_file_location('p3_reference', root/'benchmarks/reconstructed_fault/particle_free/assess_p3_extension.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
manifest = json.loads((here/'archive-manifest.json').read_text())
archive = root/manifest['path']
assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['sha256']
with tempfile.TemporaryDirectory(prefix='p3-extension-review-') as directory:
    data = Path(directory)
    with tarfile.open(archive, 'r:xz') as source:
        members = source.getmembers()
        assert {member.name for member in members} == set(manifest['members'])
        for member in members:
            assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            value = source.extractfile(member).read()
            assert hashlib.sha256(value).hexdigest() == manifest['members'][member.name]['sha256']
            target = data/member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
    results = {name: reference.assess(data/name) for name in ['serial', 'mpi2']}
    for name in ['old', 'new', 'gauss12', 'restart']:
        a = reference.load_state(data/'serial', name)
        b = reference.load_state(data/'mpi2', name)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
    for name in ['before.archive', 'after.archive']:
        assert (data/'serial'/name).read_bytes() == (data/'mpi2'/name).read_bytes()
print(json.dumps(dict(archive_members_verified=len(members), independent_checks_passed=True,
                      rank_fields_and_archives_identical=True,
                      stress_relative_L2=results['serial']['reconstructed_full_tensor_relative_L2'],
                      strict_reference_match_passed=False), indent=2))
