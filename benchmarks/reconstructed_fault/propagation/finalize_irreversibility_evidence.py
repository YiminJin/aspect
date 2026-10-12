#!/usr/bin/env python3
"""Pin the bounded task, retaining the incoming manifest and all original failures."""
from pathlib import Path
import hashlib,json,subprocess,difflib,shutil
root=Path(__file__).resolve().parents[3];b=root/'build-propagation-integration';base=b/'irreversibility-baseline';here=Path(__file__).parent;e=here/'irreversibility_evidence';docs=root/'doc/reconstructed_fault/propagation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((base/'manifest.json').read_text())
for name,h in old['hashes'].items():assert sha(base/name)==h,name
production=['include/aspect/phase_field.h','source/simulator/phase_field.cc']
for target,names in [('production-revision.diff',production),('specification.diff',['doc/reconstructed_fault/specification.tex'])]:
 (e/target).write_text(''.join(''.join(difflib.unified_diff((base/name).read_text().splitlines(True),(root/name).read_text().splitlines(True),fromfile='incoming/'+name,tofile='current/'+name)) for name in names))
for name,h in old['hashes'].items():
 if name.startswith(('source/','include/')) and name not in production:assert sha(root/name)==h,name
tracked=subprocess.check_output(['git','diff','--name-only','de7beaa65846c7372711f7922abf1c5d9a745ee6'],cwd=root,text=True).splitlines()
untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=root,text=True).splitlines()
all_names=sorted(set(tracked+untracked));inventory=dict(baseline=old['baseline'],tracked=tracked,untracked=untracked,categories={
 'production':[n for n in all_names if n.startswith(('source/','include/'))],
 'test_diagnostic':[n for n in all_names if n.startswith(('tests/','unit_tests/')) or (n.startswith('benchmarks/') and not n.endswith('.md'))],
 'documentation':[n for n in all_names if n.startswith('doc/') or n=='AGENTS.md' or (n.startswith('benchmarks/') and n.endswith('.md'))]},latest_stage='P2 direct irreversibility',latest_production_changes=production)
(docs/'source_change_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
# Retain small reviewable fields inside evidence, alongside full native records.
copy=e/'records';copy.mkdir(exist_ok=True)
for tag,k in [('125',8),('0625',16),('03125',32)]:
 for kind in ['irreversibility_dt','irreversibility_replay_guard_dt']:
  label=kind+tag;p=b/('output-'+label);d=copy/label;d.mkdir(exist_ok=True)
  names=['feedback.csv','material-local.csv','material-particles.csv','recoverable-energy.csv','coupling-work.csv']
  if kind=='irreversibility_dt':names+=['material-phase-accepted-0.csv',f'material-phase-accepted-{k}.csv',f'material-normalization-accepted-{k}.csv']
  else:names+=['irreversibility-replay.csv','irreversibility-replay-metrics.csv','replay-normalization-box.csv','replay-normalization-irreversible.csv']
  for name in names:
   if (p/name).exists():shutil.copy2(p/name,d/name)
files={root/n for n in all_names if (root/n).is_file()};files.update([root/'AGENTS.md',docs/'source_change_inventory.json',b/'aspect-release',b/'irreversibility-final-aspect',b/'irreversibility-final-plugin.so',b/'plugin/libirreversibility_feedback.release.so',b/'plugin/libmature_compat.release.so'])
files.update(p for p in e.rglob('*') if p.is_file() and p.name!='manifest.json')
files.update(b.glob('irreversibility*.log'));files.update(b.glob('irreversibility*.prm'))
for p in b.glob('output-irreversibility*'):
 files.update(p.glob('*.csv'))
 if (p/'parameters.prm').exists():files.add(p/'parameters.prm')
files.discard(e/'manifest.json')
r=dict(baseline=old['baseline'],branch='fault-propagation',previous_snapshot='build-propagation-integration/irreversibility-baseline',preserved_previous_artifacts=len(old['hashes']),production_changes=production,core_sha256=sha(b/'aspect-release'),phase_admissible_set_revision=True,initial_relaxation_unchanged=True,physical_lower_bound='immutable last accepted phase',direct_irreversibility_checks_pass=True,common_end_time_comparison_complete=True,accepted_loaded_steps=56,same_H_replays=3,identical_old_csvs=556,native_cohesive_assertions=622,mature_default_and_enabled_identical_csvs=4,temporal_agreement_improves=True,temporal_sensitivity_negligible=False,normalization_spatial_accuracy_qualified=False,full_energetic_consistency_qualified=False,full_damage_feedback_baseline_qualified=False,new_selector_MPI_restart_qualified=False,stress_scaling_run=False,P3_run=False,hashes={str(p.relative_to(root)):sha(p) for p in sorted(files)})
(e/'manifest.json').write_text(json.dumps(r,indent=2)+'\n');print('Pinned',len(files),'current artifacts; preserved',len(old['hashes']),'incoming artifacts.')
