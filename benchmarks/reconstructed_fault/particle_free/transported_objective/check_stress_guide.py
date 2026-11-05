#!/usr/bin/env python3
"""Check the maintained guide's links, entry symbols and runnable example arguments."""
import json
from pathlib import Path
import re
import shlex

ROOT = Path(__file__).resolve().parents[4]
DOC = ROOT/'doc/reconstructed_fault/particle_free'
OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/stress_consolidation_guide'


def main():
    pages = [DOC/'MATURE_MOVING_FAULT.md', DOC/'STRESS_CONSOLIDATION_DOCUMENTATION_REPORT.md']
    pages += sorted((DOC/'mature_moving_fault').glob('*.md'))
    checked = []
    for page in pages:
        text = page.read_text()
        assert re.sub(r'`[^`]*`', '', text).count('$$') % 2 == 0, page
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if '://' in target:
                continue
            name, _, anchor = target.partition('#')
            path = (page.parent/name).resolve() if name else page
            assert path.exists(), (page, target)
            if anchor and path.suffix == '.md':
                headings = re.findall(r'^#+ (.+)$', path.read_text(), re.M)
                slugs = [re.sub(r'[^\w\- ]', '', x.lower()).replace(' ', '-') for x in headings]
                assert anchor in slugs, (page, target, slugs)
            checked.append(dict(page=str(page.relative_to(ROOT)), target=target))
    symbols = {
        'localize_surface_values': 'source/material_model/phase_field_fault/dg_objective.cc',
        'make_material_reaction': 'source/material_model/phase_field_fault/stress_reaction.cc',
        'project_mature_reaction': 'source/material_model/phase_field_fault/stress_reaction.cc',
        'refresh_material_reaction': 'source/material_model/phase_field_fault/stress_reaction.cc',
        'solve_reconstructed_fault_stokes_inner': 'source/simulator/solver/reconstructed_fault_stokes.cc',
        'prepare_material_history': 'source/material_model/phase_field_fault/material_history.cc',
        'stage_material_geometry': 'source/material_model/phase_field_fault/material_geometry.cc',
        'transport_material_profile': 'source/material_model/phase_field_fault/profile_transport.cc',
        'transport_material_history': 'source/material_model/phase_field_fault/stress_evolution.cc',
        'rotate_material_history': 'source/material_model/phase_field_fault/stress_evolution.cc',
        'physical_history_stress': 'source/material_model/phase_field_fault/physical_stress.cc',
        'project_material_tensor': 'source/material_model/phase_field_fault/material_tensor_projection.cc',
        'project_to_normal_profiles_unchecked': 'source/reconstructed_fault/utilities.cc',
        'stage_material_motion': 'source/reconstructed_fault/manager.cc',
        'mature_stress_endpoint': 'include/aspect/material_model/phase_field_fault/stress_law.h',
    }
    for symbol, source in symbols.items():
        assert symbol in (ROOT/source).read_text(), symbol
    config = (DOC/'mature_moving_fault/configuration.md').read_text()
    examples = []
    for block in re.findall(r'```sh\n(.*?)```', config, re.S):
        command = shlex.split(block.replace('\\\n', ' '))
        if command[0] != 'python3':
            continue
        assert (ROOT/command[1]).is_file()
        case = {'guide-direct': 'dc', 'guide-layered': 'dl'}[command[2]]
        run = ROOT/'build-particle-free/transported-objective/stage-c/material-runs'/('consolidation-'+case)
        metadata = json.loads((run/'command.json').read_text())
        assert metadata['exit_code'] == 0
        actual = metadata['arguments']
        tokens = iter(command[3:])
        flags = {}
        for token in tokens:
            key = token[2:].replace('-', '_')
            value = True if isinstance(actual[key], bool) else next(tokens)
            expected = actual[key]
            if isinstance(expected, (int,float)) and not isinstance(expected,bool):
                value = float(value)
            assert value == expected, (case,key,value,expected)
            flags[key] = value
        assert flags['representation'] == 'direct physical stress'
        assert flags['ranks'] <= 4
        prm = (run/'input.prm').read_text()
        for selector in ['Material stress representation = direct physical stress',
                         'Bulk history representation = DG trial stress',
                         'Stress rotation = Jaumann DG material Lie']:
            assert selector in prm
        # Any unmentioned runner argument must equal its default; this list
        # states the effective defaults used in these two maintained examples.
        defaults = dict(initial_dt=.1,schedule=None,spacing=.1,reject=False,
                        true_checks=False,audit=False,isolation=False,
                        entry_control=False,restart='None')
        for key, default in defaults.items():
            if key not in flags:
                assert actual[key] == default, (case,key)
        assert 'read_text' not in (ROOT/command[1]).read_text().split("text=f'''")[0]
        examples.append(dict(label=command[2],verified_run=str(run.relative_to(ROOT)),
                             command=command,only_label_differs=True))
    result = dict(pages=len(pages),relative_links=len(checked),symbols=symbols,
                  examples=examples,links=checked)
    (OUT/'documentation-checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(pages=len(pages),links=len(checked),symbols=len(symbols),examples=len(examples))))


if __name__ == '__main__':
    main()
