"""Observe the pre-turn iterate at entry to the original executable's selector.

Compile an observer with the corresponding build flags and selector source.
A scratch header adds one nonvirtual member for reading the stage. After the
observation, the executable runs its own selector and GDB reads its MPI sums.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DONOR = 'aspect/simulator/solver/bulk_tip.h'
TARGET = 'aspect/simulator/solver/midpoint_system.h'
SYMBOL = {
    'donor':'_ZN6aspect8internal15BulkTipCouplingILi2EE20update_virgin_planesERKN6dealii16TrilinosWrappers3MPI11BlockVectorE',
    'target':'_ZN6aspect12GrowingFault14MidpointSystem25select_prospective_planesERKSt6vectorIdSaIdEE',
}


def between(text, start, end):
    i = text.index(start)
    return text[i:text.index(end,i)]


def build(role, old, work):
    base = old/'source' if role=='donor' else ROOT
    header = DONOR if role=='donor' else TARGET
    out = work/('build-'+role)
    out.mkdir(parents=True, exist_ok=True)
    decl = (base/'include'/header).read_text()
    includes = '#include "observer.h"\n'
    if role=='donor':
        decl = decl.replace('bool update_virgin_planes(', 'void inspect(const Vector &, ::StripReplay &);\n        bool update_virgin_planes(')
        source = (base/'source/simulator/solver/bulk_tip_assembly.cc').read_text()
        code = between(source,'#include <aspect/simulator/solver/bulk_tip.h>', '    template <int dim>\n    BulkTipCoupling<dim>::BulkTipCoupling')
        code += 'template <int dim> void BulkTipCoupling<dim>::inspect(const Vector &x, ::StripReplay &io) {\n'
        start = source.index('bool BulkTipCoupling<dim>::update_virgin_planes')
        body = between(source[start:],'      const auto ns=', '            const double change=')
        pos = '            strains.resize(values.n_quadrature_points);'
        body = body.replace(pos,'''            if (io.name=="native") for (unsigned int j=0;j<ids.size();++j) {
              const auto [component,local]=fe.system_to_component_index(j);
              if (component<3 || component==pc)
                io.coefficients<<cell->id()<<' '<<component<<' '<<local<<' '<<field[ids[j]]<<'\\n';
            }
'''+pos)
        body=body.replace('if (formed_sections[k] ||','if (k!=8 || formed_sections[k] ||')
        body=body.replace('if (!formed_sections[k])','if (k==8 && !formed_sections[k])')
        body=body.replace('if (model.localize_tip_direction &&\n                std::none_of(stations.begin(),stations.end(),[&](const auto k){return eligible_sections[k];})) continue;', '')
        begin=body.index('                const double u=uv')
        end=body.index('                const double a=',begin)
        body=body[:begin]+'''                const auto p=io.point(cell->id().to_string(),q,
                    {values.JxW(q),weights->second[q],F,uv[0],uv[1]},phi,predictor[0][0],predictor[1][1],predictor[0][1]);
                const double u=p.u,v=p.v,w=p.area,rho=p.kernel;
'''+body[end:].replace('(1.-F)','(1.-p.fraction)').replace('w*F','w*p.fraction')
        body=body.replace('            const auto m=&moments[9*k];\n            // Remote','            io.reduced(&moments[9*k],&centered[3*k]);\n            const auto m=&moments[9*k];\n            // Remote')
        body=body.replace('            const auto objective=', '            double force=0.;\n            const auto objective=')
        body=body.replace('++candidate_scores;','strip_free=free; strip_mobility=mobility;').replace('++candidate_rsf_solves;','').replace('              const double expanded=', '              force=f;\n              const double expanded=')
        code += body + '''
            io.angles.push_back(angle);
            for (double test:io.angles) { const double value=objective(test);
              if (!io.rank) io.scores<<test<<' '<<force<<' '<<value<<' '<<strip_free<<' '<<strip_mobility<<'\\n'; }
          }
    }
  }
}
'''
        obj='aspect::internal::BulkTipCoupling<2>'
        vec=obj+'::Vector'
        condition='op.get_timestep_number()==1 && op.orientation_updates==0 && op.eligible_sections.at(8)'
        comm='op.get_mpi_communicator()'
    else:
        # Record the arguments at the call to the linked production root.
        onset='aspect/reconstructed_fault/onset_score.h'
        osrc=(base/'include'/onset).read_text()
        key='  /** Evaluate the unnormalized'
        osrc=osrc.replace(key,'''  inline SectionRoot observe_root(const Friction &law,double free,double mobility,double age,double h,double pressure) {
    strip_free=free; strip_mobility=mobility;
    return solve_section(law,free,mobility,age,h,pressure);
  }
'''+key).replace('const double f=solve_section(', 'const double f=observe_root(')
        (out/onset).parent.mkdir(parents=True,exist_ok=True)
        (out/onset).write_text(osrc)
        decl = decl.replace('bool select_prospective_planes(', 'void inspect(const Values &, ::StripReplay &);\n    bool select_prospective_planes(')
        source=(base/'source/simulator/solver/midpoint_direction.cc').read_text()
        code=source[:source.index('  void MidpointSystem::freeze_direction_neighborhood')]
        code+='void MidpointSystem::inspect(const Values &x, ::StripReplay &io) {\n'
        body=between(source[source.index('bool MidpointSystem::select_prospective_planes'):],
                     '    const auto &geometry=', '          if (score(selected)')
        body=body.replace('return false;', 'return;')
        pos='    for (const auto &[id,cell]:cells)'
        body=body.replace(pos,'''    if (io.name=="native") {
      const auto &fe=material.get_fe();
      const auto c=material.introspection().component_indices.compositional_fields[field];
      const auto offset=material.get_solution().get_block_indices().block_start(phase_block);
      std::vector<Index> ids(fe.n_dofs_per_cell());
      for (const auto &cell:material.get_dof_handler().active_cell_iterators()) if (cell->is_locally_owned()) {
        cell->get_dof_indices(ids);
        for (unsigned int j=0;j<ids.size();++j) {
          const auto [component,local]=fe.system_to_component_index(j);
          if (component<3 || component==c)
            io.coefficients<<cell->id()<<' '<<component<<' '<<local<<' '<<x[component==c?phase_begin+ids[j]-offset:ids[j]]<<'\\n';
        }
      }
    }
'''+pos)
        body=body.replace('if (!eligible[k]) continue;', 'if (k!=8 || !eligible[k]) continue;')
        body=body.replace('if (eligible[k])', 'if (k==8 && eligible[k])')
        a=body.index('          const double w=p.area')
        b=body.index('        }',a)
        body=body[:a]+'''          const auto z=io.point(id.to_string(),q,{p.area,p.kernel,F,u,v},phi,predictor[0][0],predictor[1][1],predictor[0][1]);
          const double w=z.area*(1.-z.fraction)*(1.-z.fraction)*z.kernel;
          const auto local=onset_moments(z.area,z.kernel,z.fraction,z.u,z.v);
          for (unsigned int j=0;j<9;++j) moments[9*k+j]+=local[j];
          if (w>0.) samples.push_back({k,w,z.u,z.v});
'''+body[b:]
        body=body.replace('          const auto *m=&moments[9*k];','          io.reduced(&moments[9*k],&centered[3*k]);\n          const auto *m=&moments[9*k];')
        body=body.replace('          const auto score=', '          double force=0.;\n          const auto score=')
        body=body.replace('return candidate_onset(', 'const auto law=candidate_onset(').replace('),angle).score;', '),angle); force=law.resistance; return law.score;')
        code+=body+'''
          const double angle=(score(selected)-score(angles[k])>128.*std::numeric_limits<double>::epsilon()*(m[3]+m[4])
                              && std::abs(selected-angles[k])>1.e-7) ? selected : angles[k];
          io.angles.push_back(angle);
          for (double test:io.angles) { const double value=score(test);
            if (!io.rank) io.scores<<test<<' '<<force<<' '<<value<<' '<<strip_free<<' '<<strip_mobility<<'\\n'; }
        }
  }
}
'''
        obj='aspect::GrowingFault::MidpointSystem'
        vec=obj+'::Values'
        condition='op.material.get_timestep_number()==1 && op.angles.at(8)==0. && op.eligible.at(8)'
        comm='op.comm'
    code += f'''
extern "C" int inspect_strip8({obj} &op, const {vec} &x) {{
  if (!({condition})) return 0;
  std::ifstream modes("modes");
  for (std::string mode; modes >> mode;) {{
    StripReplay io(mode, dealii::Utilities::MPI::this_mpi_process({comm}));
    op.inspect(x,io);
  }}
  return 1;
}}
'''
    h=out/header
    h.parent.mkdir(parents=True,exist_ok=True)
    h.write_text(decl)
    src=out/'inspect.cc'
    src.write_text(includes+code)
    kind='debug' if role=='debug' else 'release'
    tree='build-slice-debug' if role=='debug' else 'build-slice'
    flags=(ROOT/tree/f'CMakeFiles/aspect.exe.{kind}.dir/flags.make').read_text()
    opts=lambda name:shlex.split(re.search(rf'^{name} = (.*)$', flags,re.M)[1])
    subprocess.run(['/opt/gcc/12.4.0/bin/g++','-shared','-fPIC','-fno-access-control',
                    '-I'+str(out),'-I'+str(HERE),'-I'+str(base/'include'),*opts('CXX_INCLUDES'),
                    *opts('CXX_DEFINES'),*opts('CXX_FLAGS'),'-o',out/f'libinspect.{kind}.so',src],check=True)


def run(role, old, work, name, ranks, modes):
    out=work/name
    out.mkdir()
    (out/'modes').write_text('\n'.join(modes))
    if not (work/'angles').exists():
        angles=[0.]
        for layout in ('donor-onecut','review','donor-serial','release1','debug','debug2'):
            file=(old/layout/('output/tip-planes-0001.csv' if layout.startswith('donor')
                              else 'principal/output/sections-0001.csv'))
            with file.open() as stream:
                row,=[r for r in csv.DictReader(stream) if r['section']=='8']
            angles.append(float(row['beta']))
        (work/'angles').write_text('\n'.join(map(repr,angles)))
    (out/'angles').write_text((work/'angles').read_text())
    if (work/'inputs').exists():
        (out/'inputs').symlink_to(work/'inputs',target_is_directory=True)
    exe=old/'reference/tip/runtime/aspect-release' if role=='donor' else ROOT/'build-slice/aspect-release'
    kind='debug' if role=='debug' else 'release'
    if role=='debug': exe=ROOT/'build-slice-debug/aspect-debug'
    prm=(old/('donor-onecut/input.prm' if role=='donor' else
              'debug2/principal/input.prm' if role=='debug' else 'review/principal/input.prm')).read_text()
    prm=re.sub(r'(?m)^(set Additional shared libraries\s*=.*)$',r'\g<1>, '+str(work/('build-'+role)/f'libinspect.{kind}.so'),prm)
    prm+=f'\nset Output directory = {out}/output\n'
    (out/'input.prm').write_text(prm)
    (out/'observe.gdb').write_text(f'''set pagination off
set debuginfod enabled off
set confirm off
set print thread-events off
set disable-randomization off
source {HERE/'reductions.py'}
break *{SYMBOL['target' if role=='debug' else role]}
commands
  silent
  set $done = ((int (*)(void*,void*)) &inspect_strip8)((void*)$rdi,(void*)$rsi)
  if $done
    disable 1
    python Reductions() if os.environ.get('OMPI_COMM_WORLD_RANK')=='0' else None
  end
  continue
end
run
''')
    env=dict(os.environ)
    if role=='donor':
        env.update(json.loads((old/'reference/runs/localized-onecut/command.json').read_text())['environment'])
    env.update(DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    cmd=['/opt/openmpi/5.0.6/bin/mpirun','--map-by','core','--bind-to','core','-np',str(ranks),
         'gdb','-nx','-batch','-return-child-result','-x','observe.gdb','--args',str(exe),str(out/'input.prm')]
    with (out/'log').open('w') as log:
        subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    if not (out/('scores-'+modes[-1])).stat().st_size:
        raise RuntimeError('No direction decision observed')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('old',type=lambda p:Path(p).resolve())
    ap.add_argument('scratch',type=lambda p:Path(p).resolve())
    ap.add_argument('role',choices=('donor','target','debug'))
    ap.add_argument('--run')
    ap.add_argument('--ranks',type=int,choices=(1,2),default=2)
    ap.add_argument('--modes',nargs='+',default=['native'])
    args=ap.parse_args()
    if args.run:
        run(args.role,args.old,args.scratch,args.run,args.ranks,args.modes)
    else:
        build(args.role,args.old,args.scratch)
