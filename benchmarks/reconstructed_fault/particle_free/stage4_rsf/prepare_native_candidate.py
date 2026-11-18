#!/usr/bin/env python3
"""Write the selected fresh native specimen/control; launch nothing.

This is a fresh conditioning experiment, NOT a restart of the growth checkpoint
and NOT a claim that a cold start realizes the reduced operating point.
"""
from stability import ROOT,OUT,DC
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
result=json.loads((OUT/'stability.json').read_text());p=result['pilot']
fixture=(HERE.parent/'first_free_turn/positive.prm').read_text()
# Use the already selected post-.04 velocity, constant throughout the fresh test.
old='0.01*(y-2)-if(t>0.04,0.0041954981558863999,0)*(x-4);if(t>0.04,0.0041954981558863999,0)*(y-2)'
new='0.01*(y-2)-0.0041954981558863999*(x-4);0.0041954981558863999*(y-2)'
assert fixture.count(old)==1
fixture=fixture.replace(old,new)
configs={}
for law in ['weakening','neutral']:
 for perturbed in [False,True]:
  name=law+('-perturbed' if perturbed else '-background')
  age=p['initial_age']*(1.01 if perturbed else 1.)
  text=fixture
  edits={'set Characteristic slip distance = .0005':f"set Characteristic slip distance = {p['Dc']:.17g}",
   'set Bulk tip initial age = .5':f'set Bulk tip initial age = {age:.17g}',
   'set Initial time step = .01':f"set Initial time step = {p['dt']/2:.17g}",
   'set Evolution effect parameters = .02':f"set Evolution effect parameters = {(.02 if law=='weakening' else .015):.17g}",
   'set Reference friction coefficients = .6':f"set Reference friction coefficients = {(.6 if law=='weakening' else p['neutral_mu0']):.17g}"}
  for old,new in edits.items():assert text.count(old)==1,old;text=text.replace(old,new)
  text='# Stage 4 selected fresh synthetic conditioning input; no production conversion.\n# Hold phase using run_native_candidate.py for the first RSF screen.\n'+text
  target=HERE/(name+'.prm');target.write_text(text)
  configs[name]=dict(input=str(target.relative_to(ROOT)),Dc=p['Dc'],age=age,dt=p['dt'],end=.02,physical_intervals=int(__import__('math').ceil(.02/p['dt'])))
profile=ROOT/'build-particle-free/bulk-tip/runs/stage4-fixed-phase-stiffness-homogeneous/output/tip-profile.csv'
import csv
with profile.open() as f:rows=list(csv.DictReader(f))
seconds=float(rows[-1]['step_seconds'])
for value in configs.values():value['cost_from_one_retained_step_hours']=value['physical_intervals']*seconds/3600
record=dict(configurations=configs,measured_reference_step_seconds=seconds,
 qualification='Prepared and runnable, not executed or qualified. Fresh zero stress/A; declared equilibrated peak-.60 seed; constant post-.04 boundary velocity from t0. Initial uniform age is scaled with Dc. The background starts in cold creep, not at the reduced operating point.',
 comparison='Four fresh histories form one weakening/control perturbation pair: each has its own unperturbed background. The common perturbation is +1% initial section age everywhere, including ahead of the seed. No age resets occur at registration.',
 cost_note='Linear estimate from one measured fixed-phase interval, not a timing guarantee. Cold-prefix and growing-rate solver costs can differ. Existing numerical work is not resumed automatically.',
 free_phase='The same input may be launched with --free-phase from t0 only, after a credible native RSF comparison. Do not release a frozen late checkpoint as a substitute.',
 decision='Do not launch a free-growth pilot: the augmented cell-scale instability is unresolved and a matched native low-speed preload/drive is missing. These are runnable cold-start conditioning alternatives, not the qualified next physical specimen.')
(OUT/'native-candidate.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
