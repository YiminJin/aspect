#!/usr/bin/env python3
"""Sequential retained-entry direct-Schur screens. No trajectory is launched."""
import argparse
from pathlib import Path
import run_native_mpi as runner

HERE = Path(__file__).resolve().parent
E = HERE / 'direct_mechanical_evidence'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['batch', 'growth', 'saddle'])
    parser.add_argument('--tag', required=True)
    parser.add_argument('--wall-cap', type=float, default=1200)
    args = parser.parse_args()
    if args.stage != 'batch':
        import json
        decision = json.loads((E / 'batch-assessment.json').read_text())
        assert decision['cost_gate_pass'], 'Full entries require a favorable setup-inclusive batch'
        if args.stage == 'saddle':
            assert json.loads((E / 'growth-assessment.json').read_text())['pass'], 'Verify growth first'
    runner.E = E
    options = argparse.Namespace(input=E / (args.stage + '.prm'), tag=args.tag,
                                 ranks=1, tangential=16, wall_cap=args.wall_cap,
                                 profile=False, failure_step=None, end_time=None)
    raise SystemExit(runner.run(options))


if __name__ == '__main__':
    main()
