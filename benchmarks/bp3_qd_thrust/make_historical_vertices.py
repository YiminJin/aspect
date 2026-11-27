"""Write the historical fixed-segment BP3 vertices in surface-to-base material order."""
import argparse
import math
from pathlib import Path

from make_vertices import plate, v0, dc, f0, b, tau0, radiation, sigma, a_deep

# Each input segment in the donor fault.txt is resampled separately, bottom first.
ANCHORS = [(28867.513459481292, 0.), (20000., 15358.983848622454),
           (9000., 34411.542731880108), (7500., 37009.618943233421), (0., 50000.)]


def vertices(anchors, spacing):
    points = []
    for a, z in zip(anchors, anchors[1:]):
        n = math.ceil(math.dist(a, z) / spacing)
        for i in range(n):
            points.append(tuple(x + i/n*(y-x) for x,y in zip(a,z)))
    points.append(anchors[-1])
    return points[::-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing", type=float, default=10.)
    parser.add_argument("--height", type=float, default=50000., help="Cropped local contact control height in metres")
    parser.add_argument("--verify-shipped", type=Path, help="Verify material IDs and equilibrium ages across math libraries")
    args = parser.parse_args()
    anchors = ANCHORS if args.height == 50000. else [(args.height/math.sqrt(3),0.),(0.,args.height)]
    def output_rows():
        yield "# Historical BP3 right-dipping thrust; ordered surface to base."
        yield "id x_m y_m initial_age_s"
        points = vertices(anchors,args.spacing)
        for i, (x,y) in enumerate(points):
            s = math.dist((x,y), points[0])
            a = .01 + (a_deep-.01)*min(1.,max(0.,(s-15000.)/3000.))
            log_argument = math.log(2*v0/plate*math.sinh((tau0-radiation*plate)/(a*sigma)))
            age = dc/v0*math.exp((a*log_argument-f0)/b)
            yield f"{i} {x:.17g} {y:.17g} {age:.17g}"

    if args.verify_shipped:
        expected = list(output_rows())
        saved = args.verify_shipped.read_text().splitlines()
        if len(saved)!=len(expected) or saved[:2]!=expected[:2]:
            raise ValueError("Historical table size or header differs.")
        for a,b in zip(expected[2:],saved[2:]):
            i,x,y,age = a.split()
            j,u,v,state = b.split()
            if (i!=j or not math.dist((float(x),float(y)),(float(u),float(v)))<=1e-8 or
                    not math.isclose(float(age),float(state),rel_tol=1e-13,abs_tol=0.)):
                raise ValueError(f"Historical coordinate or equilibrium age differs at material ID {i}.")
    else:
        print(*output_rows(),sep="\n")
