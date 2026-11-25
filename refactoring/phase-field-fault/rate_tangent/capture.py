"""In a scratch directory, build a Release unity binary that records the real section matrix.
Pass the pf-rsf worktree path and the matching configured unity build.
"""
from pathlib import Path
import argparse, shlex, subprocess, json
scratch=Path.cwd()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("worktree", type=Path)
parser.add_argument("--build", type=Path, help="Configured Release build for the sources being recorded")
parser.add_argument("--moving", action="store_true")
args = parser.parse_args()
repo = args.worktree.resolve()
build = args.build.resolve() if args.build else repo/'build-slice'
source=repo/'source/reconstructed_fault/stokes_coupling.cc'
original=source.read_text()
def inject(text, anchor, addition):
    if text.count(anchor)!=1: raise ValueError(anchor)
    return text.replace(anchor,addition+anchor)
s=inject(original,'    bool locally_closed = true;',r'''
    std::ofstream review_points;
    if (this->get_timestep_number()==2 && this->get_nonlinear_iteration()==0)
      {
        review_points.open(this->get_output_directory()+"review-points-"+std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator())));
        review_points << std::setprecision(17);
      }
''')
s=inject(s,'              samples.push_back({transfer,fe.JxW(q)',r'''
              if (review_points.is_open())
                review_points << cell->id() << ' ' << q << ' ' << fe.quadrature_point(q) << ' ' << phase[q] << ' '
                              << transfer.foot.first_vertex << ' ' << transfer.foot.xi << ' ' << transfer.chi << ' '
                              << fe.JxW(q) << ' ' << shear + stress*symmetrize(frame.dyad)
                              << ' ' << compression << '\n';
''')
s=inject(s,'        if (residual <= 64.*std::numeric_limits<double>::epsilon())',r'''
        if (this->get_timestep_number()==2 && this->get_nonlinear_iteration()==0 && iteration<2)
          {
            const auto rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
            std::ofstream o(this->get_output_directory()+"review-bands-"+std::to_string(iteration)+"-"+std::to_string(rank));
            o << std::setprecision(17);
            for (unsigned int j=0;j<n;++j)
              o << j << ' ' << accepted[j].position << ' ' << material.section().profile.normal_integrals()[j] << ' '
                << material.label_ages().at(j) << ' ' << rates[j] << ' ' << diagonal[j] << ' ' << (j+1<n?upper[j]:0.)
                << ' ' << rhs[j] << ' ' << scale[j] << ' ' << material.stiffness(accepted[j].position) << '\n';
          }
''')
if args.moving:
    # The third Picard preparation at the second endpoint has a frozen Q8
    # reaction. Record its stress defect relative to the unprojected point law.
    s = s.replace("this->get_timestep_number()==2 && this->get_nonlinear_iteration()==0",
                  "this->get_timestep_number()==2 && this->get_nonlinear_iteration()==2 && !reaction.empty()")
    s = inject(s, "              if (review_points.is_open())", r"""
              Stress direct = material.viscous_stress(strain[q],x);
              for (unsigned int i=0; i<parent.size(); ++i)
                direct += basis(q,i)*parent[i]*material.relaxation(x);
""")
    s = s.replace("<< compression << '\\n'", "<< compression << ' ' << (stress-direct).norm() << '\\n'")
(scratch/source.name).write_text('#include <fstream>\n#include <iomanip>\n'+s)
unity=next(p for p in (build/'CMakeFiles/aspect.exe.release.dir/Unity').glob('*.cxx') if str(source) in p.read_text())
replacement=scratch/unity.name
replacement.write_text(unity.read_text().replace(str(source),str(scratch/source.name)))
cfg=next(d for d in json.loads((build/'compile_commands.json').read_text()) if d['file']==str(unity))
cmd=shlex.split(cfg['command']); obj=cmd[cmd.index('-o')+1]
cmd[cmd.index('-o')+1]=str(scratch/'capture.o'); cmd[cmd.index(str(unity))]=str(replacement)
subprocess.run(cmd,cwd=build,check=True)
cmd=shlex.split((build/'CMakeFiles/aspect.exe.release.dir/link.txt').read_text())
cmd[cmd.index(obj)]=str(scratch/'capture.o'); cmd[cmd.index('-o')+1]=str(scratch/'aspect-review')
cmd=[entry if not entry.startswith('-Wl,--dependency-file=') else '-Wl,--dependency-file='+str(scratch/'link.d') for entry in cmd]
subprocess.run(cmd,cwd=build,check=True)
