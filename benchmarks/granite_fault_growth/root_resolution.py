"""Test the production signed-root certificate across adjacent binary64 rates.

Inputs are the magnitudes of the free slip rate (m/s) and mobility (m/s/Pa)
captured on the first granite interval. All fixed parameters are the specimen's.
"""
import argparse
import json
import math
import struct
import sys

import mpmath as mp

mp.mp.dps = 90


def bits(v):
    return struct.unpack(">Q", struct.pack(">d", v))[0]


def value(i):
    return struct.unpack(">d", struct.pack(">Q", i))[0]


def inspect_root(free, mobility):
    free = abs(free)
    a, b, dc, v0, mu, damping = 0.0137, 0.0156, 1e-6, 1e-6, 0.75, 4.2031734e6
    age, h, pressure = 1., 7.5e-6/2, 7e7
    certificate = mp.mpf(32*sys.float_info.epsilon)*free

    def stored_traction(v):
        x = v*h/dc
        theta = age*math.exp(-x) + h*(-math.expm1(-x)/x)
        zlog = math.log(v)-math.log(2*v0) + (mu+b*math.log(theta*v0/dc))/a
        small = math.exp(-abs(zlog))
        regularized = zlog+math.log1p(math.hypot(1., small)) if zlog > 0 else math.asinh(small)
        return pressure*a*regularized + damping*v

    def residual(v, traction):
        return mp.mpf(v)+mp.mpf(mobility)*traction-free

    def independent(v):
        # Direct signed law at the stage age, evaluated with 90 decimal digits.
        x = v*h/dc
        theta = age*mp.exp(-x) - h*mp.expm1(-x)/x
        z = v/(2*mp.mpf(v0))*mp.exp((mu + b*mp.log(theta*v0/dc))/a)
        return residual(v, mp.mpf(pressure)*a*mp.asinh(z)+damping*v)

    lo, hi = bits(1e-100), bits(free)
    assert residual(value(lo), stored_traction(value(lo))) < 0 < residual(value(hi), stored_traction(value(hi)))
    while hi-lo > 1:
        middle = (lo+hi)//2
        if residual(value(middle), stored_traction(value(middle))) < 0:
            lo = middle
        else:
            hi = middle
    bracket = [value(lo), value(hi)]
    errors = [float(residual(v, stored_traction(v))) for v in bracket]
    root = mp.findroot(independent, tuple(map(mp.mpf, bracket)), tol=mp.mpf("1e-76"))
    assert abs(independent(root)) < certificate*mp.mpf("1e-35")
    ratios = [error/float(certificate) for error in errors]
    assert ratios[0] < -1 and ratios[1] > 1, ratios
    return dict(free_m_per_s=free, mobility_m_per_s_Pa=mobility, pressure_Pa=pressure, age_s=age, h_s=h,
                independent_rate_m_per_s=float(root), independent_resistance_Pa=float((free-root)/mobility),
                adjacent_binary64_rates=bracket, stored_residuals_m_per_s=errors,
                certificate_m_per_s=float(certificate), residual_over_certificate=ratios,
                independent_residual_at_rounded_root=float(independent(mp.mpf(float(root)))))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("free", type=float)
    p.add_argument("mobility", type=float)
    args = p.parse_args()
    print(json.dumps(inspect_root(args.free, args.mobility), indent=2))
