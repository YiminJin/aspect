"""Build a scratch Release executable that records the Q8 inputs and reduced Q1 tangent.

Use the worktree and its matching configured unity build. Run this script from
the scratch directory. The uninstrumented executable and its objects are preserved.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess


def insert_before(source, anchor, addition):
    if source.count(anchor) != 1:
        raise ValueError(anchor)
    return source.replace(anchor, addition + anchor)


def instrument_section_solve(original, moving):
    # Observe the same assembly as the corresponding directional-difference
    # check. At step 2 the stationary check uses Picard iteration 0. The moving
    # check uses iteration 2, after the interior stress projection is populated.
    condition = 'this->get_timestep_number()==2 && this->get_nonlinear_iteration()==0'
    if moving:
        condition = ('this->get_timestep_number()==2 && this->get_nonlinear_iteration()==2'
                     ' && !projected_response.empty()')

    open_points = r"""
    std::ofstream review_points;
    if (CAPTURE_ITERATE)
      {
        review_points.open(this->get_output_directory()+"review-points-"+std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator())));
        review_points << std::setprecision(17);
      }
"""
    source = insert_before(original, '    bool locally_closed = true;',
                           open_points.replace('CAPTURE_ITERATE', condition))

    # Read the actual completed sample passed to the section residual. The
    # optional last column measures the projected minus unprojected stress (Pa).
    append = 'samples.push_back({transfer,'
    if source.count(append) != 1:
        raise ValueError(append)
    end = source.index('});', source.index(append)) + len('});')
    record_point = r"""
              if (review_points.is_open())
                {
                  const auto &point = samples.back();
                  review_points << cell->id() << ' ' << q << ' ' << fe.quadrature_point(q) << ' ' << phase[q]
                                << ' ' << point.transfer.foot.first_vertex << ' ' << point.transfer.foot.xi
                                << ' ' << point.transfer.chi << ' ' << point.volume << ' ' << point.driving_shear
                                << ' ' << point.compression;
"""
    if moving:
        record_point += r"""
                  Stress direct = material.viscous_stress(strain_rate[q],x);
                  for (unsigned int i=0; i<parent.size(); ++i)
                    direct += basis(q,i)*parent[i]*material.stress_relaxation_factor(x);
                  review_points << ' ' << (stress-direct).norm();
"""
    record_point += r"""
                  review_points << '\n';
                }
"""
    source = source[:end] + record_point + source[end:]

    # The rows have been reduced across ranks and the tridiagonal solve has
    # not overwritten them. Newton iterations 0 and 1 also reveal the first
    # accepted fraction of the rate correction.
    bands = r"""
        if (CAPTURE_ITERATE && iteration<2)
          {
            const auto rank = Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
            std::ofstream out(this->get_output_directory()+"review-bands-"+std::to_string(iteration)+"-"+std::to_string(rank));
            out << std::setprecision(17);
            for (unsigned int j=0; j<n; ++j)
              out << j << ' ' << labels[j].position << ' ' << material.active_section().profile.normal_integrals()[j] << ' '
                  << material.label_ages().at(j) << ' ' << rates[j] << ' ' << diagonal[j] << ' ' << (j+1<n?upper[j]:0.)
                  << ' ' << rhs[j] << ' ' << scale[j] << ' ' << material.effective_maxwell_viscosity(labels[j].position) << '\n';
          }
"""
    return insert_before(source, '        if (residual <= 64.*std::numeric_limits<double>::epsilon())',
                         bands.replace('CAPTURE_ITERATE', condition))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worktree", type=Path)
    parser.add_argument("--build", type=Path, help="Configured Release build for the sources being recorded")
    parser.add_argument("--moving", action="store_true")
    args = parser.parse_args()
    scratch = Path.cwd()
    repo = args.worktree.resolve()
    build = args.build.resolve() if args.build else repo / 'build-slice'
    source = repo / 'source/reconstructed_fault/stokes_coupling.cc'
    instrumented = scratch / source.name
    instrumented.write_text('#include <fstream>\n#include <iomanip>\n'
                            + instrument_section_solve(source.read_text(), args.moving))

    # Compile one translation unit and reuse all other objects from this build.
    target = build / 'CMakeFiles/aspect.exe.release.dir'
    unity = next(path for path in (target/'Unity').glob('*.cxx') if str(source) in path.read_text())
    replacement = scratch / unity.name
    replacement.write_text(unity.read_text().replace(str(source), str(instrumented)))
    commands = json.loads((build/'compile_commands.json').read_text())
    compile_command = shlex.split(next(entry['command'] for entry in commands if entry['file'] == str(unity)))
    object_index = compile_command.index('-o') + 1
    ordinary_object = compile_command[object_index]
    compile_command[object_index] = str(scratch/'capture.o')
    compile_command[compile_command.index(str(unity))] = str(replacement)
    subprocess.run(compile_command, cwd=build, check=True)

    link = shlex.split((target/'link.txt').read_text())
    link[link.index(ordinary_object)] = str(scratch/'capture.o')
    link[link.index('-o')+1] = str(scratch/'aspect-review')
    link = [arg if not arg.startswith('-Wl,--dependency-file=') else '-Wl,--dependency-file='+str(scratch/'link.d')
            for arg in link]
    subprocess.run(link, cwd=build, check=True)


if __name__ == '__main__':
    main()
