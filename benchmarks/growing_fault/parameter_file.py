"""Read ASPECT-resolved benchmark parameters and write scoped overrides."""
import json


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
