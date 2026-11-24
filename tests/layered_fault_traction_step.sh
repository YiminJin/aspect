#!/usr/bin/env python3
"""Check the coupled gate when the first physical Stokes residual already passes."""
import re
import sys

log = sys.stdin.read().split('Timestep 1:')[-1]
residuals = [float(x) for x in re.findall(r'after nonlinear iteration \d+: (\S+)',log)]
changes = [float(x) for x in re.findall(r'Relative change in returned fault stress projection: (\S+)',log)]
if len(residuals)<2 or not residuals[0]<.001 or not changes or not changes[0]>.4:
    raise ValueError('Expected a post-solve projection veto followed by another Stokes assembly')
print('The returned fault stress required another Stokes assembly before publication.')
