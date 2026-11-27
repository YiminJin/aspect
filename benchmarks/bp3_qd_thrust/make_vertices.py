"""Write SEAS BP3-QD (2021) equations 18, 25 and 26 as an ASPECT vertex table."""
import math
import sys

plate, v0, dc, f0, b = 1e-9, 1e-6, 0.008, 0.6, 0.015
sigma, radiation = 50e6, 2670 * 3464 / 2
a_deep = 0.025
tau0 = sigma * a_deep * math.asinh(plate / (2 * v0) *
                                   math.exp((f0 + b * math.log(v0 / plate)) / a_deep)) + radiation * plate

if __name__ == "__main__":
    print("# SEAS BP3-QD 60 degree thrust. IDs increase down dip from the surface.")
    print("id x_m y_m initial_age_s")
    stations = [*range(0, 15000, 250), *range(15000, 18000, 100), *range(18000, 40001, 250)]
    for i, s in enumerate(stations):
        a = 0.010 + (a_deep - 0.010) * min(1, max(0, (s - 15000) / 3000))
        log_argument = math.log(2 * v0 / plate * math.sinh((tau0 - radiation * plate) / (a * sigma)))
        age = dc / v0 * math.exp((a * log_argument - f0) / b)
        print(i, format(0.5 * s, ".17g"), format(math.sqrt(0.75) * (40000 - s), ".17g"),
              format(age, ".17g"))
    print(f"Uniform background shear = {tau0:.17g} Pa", file=sys.stderr)
