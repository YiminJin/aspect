"""Build a scratch ASPECT executable which observes both selectors on each solved midpoint state.

The archived patch supplies the candidate point score. Both scores run the search
extracted from the current solver, with the same material, moments and incoming
angle. Only the selected policy updates the direction. Use a fresh run directory.
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess

here = Path(__file__).resolve().parent
repo = here.parents[3]
solver = Path('source/simulator/solver/midpoint_direction.cc')
assembly = Path('source/simulator/solver/midpoint_fault.cc')
header = Path('include/aspect/reconstructed_fault/onset_score.h')


def substitute(text, key, replacement):
    if text.count(key) != 1:
        raise ValueError(f'Expected exactly one capture site: {key}')
    return text.replace(key, replacement)


def instrument(scratch):
    patch = here.parent / 'interface-c2/stable-score.patch'
    for name in (solver, header, Path('unit_tests/material_sections.cc')):
        (scratch / name.parent).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, scratch / name)
    for options in (['--dry-run'], []):
        subprocess.run(['patch', '-p1', '--batch', *options, '-i', patch], cwd=scratch, check=True)

    # The patched function and its original coexist in this one translation unit.
    patched = (scratch / header).read_text()
    patched = patched.replace('_aspect_onset_score_h', '_c1_isolated_candidate_h')
    patched = substitute(patched, 'namespace aspect::GrowingFault', 'namespace aspect::GrowingFault::isolated_candidate')
    (scratch / 'candidate.h').write_text(patched)
    source = (repo / solver).read_text()
    source = '#include <fstream>\n#include <iomanip>\n#include <cstdlib>\n#include "candidate.h"\n' + source
    anchor = '    Values moments(9*angles.size())'
    source = substitute(source, anchor, r'''
    const auto rank = Utilities::MPI::this_mpi_process(comm);
    const auto dir = sim.get_output_directory();
    const bool diagnose = angles.size() == 9 && material.history().step == 0 && angles[8] == 0.;
    const bool use_candidate = std::getenv("DIR_C1_PUBLISH_CANDIDATE") != nullptr;
    std::ofstream pts, sums;
    if (diagnose)
      {
        const auto suffix = std::to_string(rank);
        pts.open(dir+"turn-points-"+suffix);
        sums.open(dir+"turn-sums-"+suffix);
        pts << std::setprecision(17);
        sums << std::setprecision(17);
        std::ofstream fields(dir+"turn-fields-"+suffix);
        fields << std::setprecision(17);
        for (const auto &[id,cell] : cells)
          for (unsigned int j=0; j<cell.indices.size(); ++j)
            fields << id << ' ' << j << ' ' << x[cell.indices[j]] << '\n';
      }
''' + anchor)
    anchor = '          const auto local=onset_moments'
    source = substitute(source, anchor, r'''          if (diagnose && k==8)
            pts << id << ' ' << q << ' ' << p.area << ' ' << p.kernel << ' ' << F << ' ' << u << ' ' << v << ' ' << phi << '\n';
''' + anchor)
    anchor = '    Utilities::MPI::sum(moments,comm,moments);'
    source = substitute(source, anchor, r'''    if (diagnose)
      {
        for (unsigned int j=72; j<81; ++j) sums << moments[j] << ' ';
        sums << '\n';
      }
''' + anchor)
    anchor = '    bool changed=false;'
    source = substitute(source, anchor, r'''    if (diagnose)
      {
        for (unsigned int j=72; j<81; ++j) sums << moments[j] << ' ';
        for (unsigned int j=24; j<27; ++j) sums << centered[j] << ' ';
        sums << '\n';
        for (double a : {0.,.23,-.14788742977350777,-.14788657313582751})
          {
            const auto r=score_plane(8,&moments[72],&centered[24],a);
            sums << a << ' ' << r.resistance << ' ' << r.score << '\n';
          }
      }
''' + anchor)

    patched_solver = (scratch / solver).read_text()
    fn = patched_solver.split('OnsetScore MidpointSystem::score_plane', 1)[1].split('\n  {\n', 1)[1].split('\n  }', 1)[0]
    candidate_function = fn.replace('    return candidate_onset(', '    return isolated_candidate::candidate_onset(')
    if fn == candidate_function:
        raise ValueError('Could not isolate archived candidate')
    production_function = source.split('OnsetScore MidpointSystem::score_plane', 1)[1].split('\n  {\n', 1)[1].split('\n  }', 1)[0]
    # The installed observer also calls score_plane() on synthetic off-peak
    # states. Route that call to the experiment's selected score as well.
    candidate_at_probe = candidate_function.replace(
        'return isolated_candidate::candidate_onset(', 'const auto result = isolated_candidate::candidate_onset(')
    source = substitute(source, production_function,
                        '    if (std::getenv("DIR_C1_PUBLISH_CANDIDATE"))\n      {\n' + candidate_at_probe
                        + '\n        return {result.score,result.resistance};\n      }\n' + production_function)
    begin = source.index('          const auto score =')
    search_begin = source.index('          const double limit=', begin)
    selected = '          const double selected=.5*(lo+hi);'
    search_end = source.index(selected, search_begin) + len(selected)
    search = source[search_begin:search_end]
    # Log unaltered pre-comparison scores to identify the first golden update
    # whose outcomes differ at bit-identical left/right trial angles.
    comparison = '              if (fl<fr)'
    search = substitute(search, comparison, r'''              if (diagnose && rank==0 && k==8)
                {
                  comparison_log << policy << ',' << lo << ',' << hi << ',' << left << ',' << right
                                 << ',' << fl << ',' << fr << ',' << (fl<fr) << '\n';
                }
''' + comparison)
    update_begin = source.index('          if (score(selected)', search_end)
    update_end = source.index('\n            }', update_begin) + len('\n            }')
    gate = source[update_begin:update_end].split('if (',1)[1].split(')\n            {',1)[0]
    code = r'''
          const auto production = [&](const unsigned int k, const double *moments,
                                      const double *centered_moments, const double angle) {
PRODUCTION_BODY
          };
          const auto candidate = [&](const unsigned int k, const double *moments,
                                     const double *centered_moments, const double angle) {
CANDIDATE_BODY
          };
          const auto full_root = [&](const double a) {
            const double scale = G*B*h*material.sections().length(k,a);
            return material.friction().solve_strip(k,(m[0]*std::sin(2.*a)+m[1]*std::cos(2.*a))/scale,
                                                    m[2]/scale,h,friction_pressures.values[k]);
          };
          const auto evaluate = [&](const unsigned int policy, const double a) -> OnsetScore {
            if (policy == 0) return production(k,m,&centered[3*k],a);
            const auto r=candidate(k,m,&centered[3*k],a);
            return {r.score,r.resistance};
          };
          std::ofstream comparison_log;
          if (diagnose && rank==0 && k==8)
            {
              comparison_log.open(dir+"turn-search.csv");
              comparison_log << std::setprecision(17) << "policy,lo,hi,left,right,fl,fr,left_smaller\n";
            }
          double proposals[2];
          bool turns[2];
          for (unsigned int policy=0; policy<2; ++policy)
            {
              const auto score = [&](const double a) { return evaluate(policy,a).score; };
SEARCH_BODY
              proposals[policy] = selected;
              turns[policy] = GATE;
              if (rank == 0)
                {
                  std::ofstream out(dir+"turn-decisions.csv",std::ios::app);
                  if (out.tellp()==0)
                    out << "step,k,incoming,policy,bin,lo,hi,angle,score,resistance,accepted,ds,pressure,rate,update_threshold\n";
                  const auto r=evaluate(policy,selected);
                  out << std::setprecision(17) << sim.get_timestep_number() << ',' << k << ',' << angles[k] << ','
                      << policy << ',' << best << ',' << -limit+(best-1)*spacing << ',' << -limit+(best+1)*spacing << ','
                      << selected << ',' << r.score << ',' << r.resistance << ',' << turns[policy] << ','
                      << material.sections().length(k,selected) << ',' << friction_pressures.values[k] << ','
                      << full_root(selected).rate << ',' << 128.*std::numeric_limits<double>::epsilon()*(m[3]+m[4]) << '\n';
                }
            }
          if (diagnose && k==8 && rank==0)
            {
              std::ofstream out(dir+"turn-scores.csv");
              out << std::setprecision(17) << "policy,where,angle,resistance,score,rate\n";
              for (unsigned int policy=0; policy<2; ++policy)
                for (const std::pair<const char *, double> &trial :
                     {std::make_pair("incoming",angles[k]),std::make_pair("off_peak",.23),
                      std::make_pair("production",proposals[0]),std::make_pair("candidate",proposals[1])})
                  {
                    const auto r=evaluate(policy,trial.second);
                    out << policy << ',' << trial.first << ',' << trial.second << ',' << r.resistance << ','
                        << r.score << ',' << full_root(trial.second).rate << '\n';
                  }
            }
          if (turns[use_candidate])
            {
              angles[k]=proposals[use_candidate];
              changed=true;
            }'''
    code = code.replace('PRODUCTION_BODY', production_function).replace('CANDIDATE_BODY', candidate_function).replace('SEARCH_BODY', search).replace('GATE',gate)
    (scratch / solver.name).write_text(source[:begin]+code+source[update_end:])
    source = (repo / assembly).read_text()
    anchor = '    // In exact arithmetic'
    source = substitute(source, anchor, r'''    if (publish && ns==9 && sim.get_timestep_number()==1 && !Utilities::MPI::this_mpi_process(comm))
      std::ofstream(sim.get_output_directory()+"root-input") << std::setprecision(17) << transfer.free_rate[8] << ' '
        << transfer.mobility[8] << ' ' << angles[8] << ' ' << friction_pressures.values[8] << '\n';
''' + anchor)
    (scratch / assembly.name).write_text(source)


def build_binary(build, scratch):
    target = build / 'CMakeFiles/aspect.exe.release.dir'
    unity, = [p for p in (target/'Unity').glob('*.cxx') if str(repo/solver) in p.read_text()]
    contents = unity.read_text()
    for original in (solver, assembly):
        contents = substitute(contents, str(repo/original), str(scratch/original.name))
    replacement = scratch / unity.name
    replacement.write_text(contents)
    entry, = [c for c in json.loads((build/'compile_commands.json').read_text()) if c['file']==str(unity)]
    cmd = shlex.split(entry['command'])
    obj = cmd[cmd.index('-o')+1]
    cmd[cmd.index('-o')+1] = str(scratch/'direction.o')
    cmd[cmd.index(str(unity))] = str(replacement)
    subprocess.run(cmd, cwd=build, check=True)
    link = shlex.split((target/'link.txt').read_text())
    link[link.index(obj)] = str(scratch/'direction.o')
    link[link.index('-o')+1] = str(scratch/'aspect-directions')
    link = [arg if not arg.startswith('-Wl,--dependency-file=') else '-Wl,--dependency-file='+str(scratch/'link.d') for arg in link]
    subprocess.run(link, cwd=build, check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('build', type=Path)
    p.add_argument('scratch', type=Path)
    args = p.parse_args()
    scratch = args.scratch.resolve()
    scratch.mkdir(parents=True, exist_ok=False)
    instrument(scratch)
    build_binary(args.build.resolve(), scratch)
