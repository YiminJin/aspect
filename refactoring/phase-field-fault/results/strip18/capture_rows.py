"""Link one isolated Release object to record the phase rows that passed both production gates."""
import argparse
import json
import re
from pathlib import Path
import shlex
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('build',type=lambda p:Path(p).resolve())
parser.add_argument('scratch',type=lambda p:Path(p).resolve())
args=parser.parse_args()
out=args.scratch
out.mkdir(parents=True,exist_ok=False)
repo=Path(__file__).resolve().parents[4]
source=repo/'source/simulator/solver/midpoint_fault.cc'
# Observe the returned midpoint solution after its convergence gates and before
# field publication. Both the original and current call differ only in spacing.
publication = r'^ *system\.write_candidate_fields\(x, *solution\);$'
original = source.read_text()
calls = list(re.finditer(publication, original, re.MULTILINE))
if len(calls) != 1:
    raise ValueError('Expected one returned-state publication site')
probe=r'''
            {
              const auto rank=Utilities::MPI::this_mpi_process(system.comm);
              const auto c=sim.introspection().component_indices.compositional_fields[system.field];
              const auto offset=solution.get_block_indices().block_start(system.phase_block);
              std::vector<types::global_dof_index> ids(sim.get_fe().n_dofs_per_cell());
              std::ofstream rows(sim.get_output_directory()+"rows-"+Utilities::int_to_string(sim.get_timestep_number(),4)
                                 +"-r"+std::to_string(rank)+".csv");
              rows<<std::setprecision(17)<<"cell,i,lower,upper,stage,mass,residual,h,B\n";
              for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
                if (cell->is_locally_owned())
                  {
                    cell->get_dof_indices(ids);
                    for (unsigned int j=0;j<ids.size();++j)
                      {
                        const auto [component,local]=sim.get_fe().system_to_component_index(j);
                        if (component!=c) continue;
                        const auto i=system.phase_begin+ids[j]-offset;
                        rows<<cell->id()<<','<<local<<','<<system.lower[i]<<','<<system.upper[i]<<','<<x[i]<<','
                            <<system.mass[i]<<','<<system.Kf*returned.residual[i]<<','<<system.h<<','<<system.B<<'\n';
                      }
                  }
            }
'''
publication_offset = calls[0].start()
(out/source.name).write_text(original[:publication_offset] + probe + original[publication_offset:])
target=args.build/'CMakeFiles/aspect.exe.release.dir'
unity,=[f for f in (target/'Unity').glob('*.cxx') if str(source) in f.read_text()]
replacement=out/unity.name
replacement.write_text(unity.read_text().replace(str(source),str(out/source.name)))
entry,=[s for s in json.loads((args.build/'compile_commands.json').read_text()) if s['file']==str(unity)]
cmd=shlex.split(entry['command'])
obj=cmd[cmd.index('-o')+1]
cmd[cmd.index('-o')+1]=str(out/'rows.o')
cmd[cmd.index(str(unity))]=str(replacement)
subprocess.run(cmd,cwd=args.build,check=True)
link=shlex.split((target/'link.txt').read_text())
link[link.index(obj)]=str(out/'rows.o')
link[link.index('-o')+1]=str(out/'aspect-rows')
link=[s if not s.startswith('-Wl,--dependency-file=') else '-Wl,--dependency-file='+str(out/'link.d') for s in link]
subprocess.run(link,cwd=args.build,check=True)
