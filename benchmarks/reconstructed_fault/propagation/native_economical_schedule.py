#!/usr/bin/env python3
"""Same endpoint equations, fresh shared seed, prescribed nested time schedule."""
from pathlib import Path
import argparse
import numpy as np
import native_endpoint as native

E=Path(__file__).resolve().parent/'native_tangent_evidence'

def schedule(level):
    # Fine around first slip (.0889 s) and damage (.5273 s), independently
    # calibrated from the retained uniform trajectory, not by changing loading.
    pieces=[(0.,.078125,64),( .078125,.09375,1024*level),
            (.09375,.5,64*level),(.5,.5234375,128*level),
            (.5234375,.625,1024*level)]
    return np.concatenate([a+np.arange(1,round((b-a)*d)+1)/d for a,b,d in pieces])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--level',type=int,choices=[1,2],default=1)
    args=parser.parse_args()
    native.run(1024*args.level,schedule=schedule(args.level),output=E/f'economical-{args.level}')
