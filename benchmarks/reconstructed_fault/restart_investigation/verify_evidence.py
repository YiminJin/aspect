#!/usr/bin/env python3
"""Check diagnostic equivalence, original source preservation and checkpoint hashes."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent;repo=root.parents[2];e=root/'evidence'
records={}
for name in ('debug-native-one','debug-no-observer-native-one'):
 text=(e/f'{name}-gdb.log').read_text(errors='replace')
 assert 'INSPECTION_ERROR' not in text
 assert 'received signal SIGSEGV' in text and 'manager.cc:1077' in text
 records[name]=[json.loads(line.split('MANAGER_STATE ',1)[1]) for line in text.splitlines() if line.startswith('MANAGER_STATE ')]
last=[next(r for r in records[name] if r['where']=='before trial values') for name in records]
assert last[0]==last[1]
state=last[0]
assert state['faults']==1 and state['vertices']==[8] and state['prescribed_outer_size']==0
assert state['solve_active'] and state['trial_active']
for field in ('committed','current','trial','incoming'):
 assert state[field]==dict(size=1,rows=[dict(size=8,values=[1e-12]*8)])
after=next(r for r in records['debug-no-observer-native-one'] if r['where']=='after restart rebuild')
assert after['prescribed_outer_size']==0 and not after['solve_active'] and not after['trial_active']
assert after['trial']==dict(size=1,rows=[dict(size=0,values=[])])
for name in ('original-native-one','no-observer-native-one'):
 assert json.loads((e/f'{name}.json').read_text())['exit_code']==139
original=(e/'original-native-one.log').read_text(errors='replace')
assert 'Stage-J checkpoint histories, V, geometry, and bulk: verified' in original
# The diagnostic copy disconnects exactly one optional observer; assertions remain unchanged.
original=(repo/'tests/phase_field_fault_stage_j_restart.cc').read_text()
copy=(root/'plugin/no_restore_observer.cc').read_text()
expected=original.replace('#include "phase_field_fault_stage_j.cc"','#include "'+str(repo/'tests/phase_field_fault_stage_j.cc')+'"')
expected=expected.replace('      signals.start_timestep.connect(&verify_restored_state<dim>);','      // Diagnostic only: omit the optional restored-fingerprint observer.\n      (void)signals;')
assert copy==expected
for filename in ('entry-source-hashes.json','checkpoint-hashes.json'):
 for path,digest in json.loads((e/filename).read_text()).items():
  assert hashlib.sha256((repo/path).read_bytes()).hexdigest()==digest,path
(e/'manager-states.json').write_text(json.dumps(records,indent=2)+'\n')
summary=dict(original_one_rank_segv=True,original_restored_assertions_passed=True,observer_disconnected_same_segv=True,identical_pre_call_state=True,missing_prescribed_container_at_deserialization_exit=True,source_header_tests_unchanged=len(json.loads((e/'entry-source-hashes.json').read_text())),original_checkpoint_preserved=True)
(e/'investigation-verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
