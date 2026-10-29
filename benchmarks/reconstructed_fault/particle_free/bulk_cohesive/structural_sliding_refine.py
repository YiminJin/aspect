#!/usr/bin/env python3
"""Focused follow-up at unchanged physical parameters, saved separately."""
import json,csv
from structural_sliding import *


def main():
    histories=[];profiles=[];metas=[];comparisons=[]
    # Recompute 256 at common dt to retain every comparison time in memory.
    for peak in [.8,1.]:
        previous,_,_=run(256,peak,.002)
        for cells in [512,1024]:
            fine,p,meta=run(cells,peak,.002);comparison=compare(previous,fine,'spatial')
            histories+=fine[4::5];profiles+=p;metas.append(meta);comparisons+=comparison
            if all(r['passed'] for r in comparison):break
            previous=fine
        half,p,meta=run(cells,peak,.001);histories+=half[9::10];profiles+=p;metas.append(meta);comparisons+=compare(fine,half,'temporal')
    write_csv('refined_histories.csv',histories);write_csv('refined_profiles.csv',profiles);write_csv('refined_comparison.csv',comparisons)
    (OUT/'refined_summary.json').write_text(json.dumps(dict(runs=metas,comparisons=comparisons),indent=2)+'\n')

if __name__=='__main__':main()
