#!/usr/bin/env python3
"""Measure bounded native progress with one-second observation uncertainty."""
import argparse
import json
from pathlib import Path
import time

parser=argparse.ArgumentParser();parser.add_argument('case');args=parser.parse_args()
directory=Path(__file__).resolve().parent/'native_endpoint_evidence'
log=directory/f'{args.case}.log';finished=directory/f'{args.case}-run.json'
out=directory/f'{args.case}-progress.json'
assert not out.exists(), 'Preserve previous cost evidence'
start=time.perf_counter();offset=0;pending='';events=[]
while True:
    # Read only appended bytes: rereading a growing tangent log each second
    # would add avoidable observer work to the cost measurement.
    if log.exists():
        with log.open() as file:
            file.seek(offset);pending+=file.read();offset=file.tell()
    lines=pending.split('\n');pending=lines.pop()
    for line in lines:
        if any(text in line for text in ['*** Timestep','Endpoint phase iteration',
                                        'Endpoint complete state committed','Endpoint phase target']):
            events.append(dict(seconds=time.perf_counter()-start,line=line.strip()))
    out.write_text(json.dumps(dict(observation_uncertainty_seconds=1.,events=events),indent=2)+'\n')
    if finished.exists():break
    time.sleep(1)
