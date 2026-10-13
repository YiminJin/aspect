#!/usr/bin/env python3
"""Reuse the retained MPI runner for four sequential scoped QN controls."""
import argparse

import cheaper_coupling as test
import run_native_mpi as runner


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--kind',choices=['core','saddle'],required=True)
    parser.add_argument('--ranks',type=int,choices=[1,4],required=True)
    parser.add_argument('--wall-cap',type=float,default=3600.)
    parser.add_argument('--tag',help='Unique label preserving a previous failed launch')
    args=parser.parse_args()
    runner.E=test.E/'native'
    args.input=runner.E/f'{args.kind}.prm'
    args.tag=args.tag or f'quasi-newton-{args.kind}-n{args.ranks}'
    args.tangential=16;args.end_time=None;args.failure_step=None;args.profile=True
    raise SystemExit(runner.run(args))
