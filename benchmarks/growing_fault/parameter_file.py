"""Read ASPECT-resolved benchmark parameters and write scoped overrides."""
import json
import re
from pathlib import Path


def subsection(name, settings):
    return f'\nsubsection {name}\n{settings}\nend\n'


def function(expression):
    return subsection('Function', f'set Function expression = {expression}')


def resolved_function(output, model):
    parameters = json.loads((output/'parameters.json').read_text())
    return parameters[model]['Function']


def wall_components(output):
    wall = resolved_function(output, 'Boundary velocity model')['Function expression']['value']
    components = [value.strip() for value in wall.split(';')]
    if len(components) != 2 or not all(components):
        raise ValueError(f'{output}: expected two Cartesian wall velocity components')
    return components


def composition_names(output):
    parameters = json.loads((output/'parameters.json').read_text())
    names = [name.strip() for name in parameters['Compositional fields']['Names of fields']['value'].split(',')]
    phase = (parameters['Phase field']['Phase composition']['value'] or
             parameters['Postprocess']['Phase field surface energy']['Phase composition']['value'] or 'phase_field')
    if phase not in names:
        raise ValueError(f'{output}: configured phase {phase!r} missing from {names}')
    return names, phase


def public_nine_strip_input():
    """Use the frozen four-field specimen as the physical base for new parameter variants."""
    frozen = Path(__file__).with_name('onecut.prm').read_text()
    old = ' set Model name = growing fault\n subsection Growing fault\n'
    if frozen.count(old) != 1:
        raise ValueError('Cannot identify the material in the frozen nine-strip input')
    return frozen.replace(old, ' set Model name = phase field fault\n subsection Phase field fault\n')


def strip_material(parameters):
    """Read the resolved growing parameters in current and archived parameters.json files."""
    material = parameters['Material model']
    canonical = material.get('Phase field fault', {})
    if 'Forward seed tip' in canonical:
        return canonical
    # Older writers kept strip geometry and resolved RSF coefficients here.
    result = dict(material['Growing fault'])
    if 'Friction normal stress' in material:
        result['Friction normal stress'] = material['Friction normal stress']
    return result


def take_material_subsection(text, name):
    """Return the rest of the input and the contents of Material model / name."""
    lines = text.splitlines(keepends=True)
    path = []
    for i, line in enumerate(lines):
        token = line.partition('#')[0].strip()
        words = token.split(maxsplit=1)
        if words[:1] == ['subsection']:
            path.append(words[1])
            if path == ['Material model', name]:
                start = i
        elif token == 'end':
            if path == ['Material model', name]:
                return ''.join(lines[:start] + lines[i+1:]), ''.join(lines[start+1:i])
            path.pop()
    raise ValueError(f'Missing subsection Material model / {name}')


def legacy_mature_input(current):
    """Move the mature control's viscosity to the subsection understood by old writers."""
    outer, fault = take_material_subsection(current, 'Phase field fault')
    viscosity = re.findall(r'(?m)^\s*(set Viscosity\s*=.*)$', fault)
    if len(viscosity) != 1:
        raise ValueError('Expected one Maxwell viscosity in the mature control')
    return outer + subsection('Material model',
                              subsection('Phase field fault', fault.replace(viscosity[0], '')) +
                              subsection('Simple model', viscosity[0]))
