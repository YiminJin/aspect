"""Check the installed plotter on the default and four edited growing inputs."""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np


def check_detail(plot, output, origin_label, strip, signed_degrees):
    fig = plot.plot_seed(output)
    detail = fig.axes[1]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    annotations = {text.get_text(): text for text in detail.texts}
    sense = 'counterclockwise' if signed_degrees > 0 else 'clockwise'
    turn = f'{abs(signed_degrees):.2f}° {sense}'
    for label in (origin_label, f'Station {strip}', turn):
        annotation = annotations[label]
        x, y = annotation.xy
        if not detail.viewLim.contains(x, y):
            raise ValueError(f'{label} marker is outside the detail at ({x}, {y}) m')
        # Check both ends of the rendered label in display coordinates.
        for corner in annotation.get_window_extent(renderer).get_points():
            if not detail.bbox.contains(*corner):
                raise ValueError(f'{label} annotation extends beyond the detail')
    segment = next(line for line in detail.lines if line.get_marker() == 'o')
    x, y = segment.get_data()
    angle = np.arctan2(y[1] - y[0], x[1] - x[0])
    np.testing.assert_allclose(angle, np.deg2rad(signed_degrees), atol=1e-7, rtol=1e-7)
    print(f'{origin_label} to station {strip}: {turn}; '
          f'offset {y[1]:.9f} m within {detail.get_ylim()} m')
    plot.plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('installed_plotter', 'default', 'reverse', 'large', 'translated', 'legacy'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.installed_plotter.resolve().parent))
    import plot_seed as plot
    check_detail(plot, args.default, 'Station 7', 8, -8.472218387)
    check_detail(plot, args.reverse, 'Station 7', 8, 8.472263747)
    check_detail(plot, args.large, 'Prescribed tip', 7, -25.357850691)
    check_detail(plot, args.translated, 'Station 7', 8, -8.472218387)
    check_detail(plot, args.legacy, 'Station 7', 8, -8.472218854)

    # A nodal Q2 reconstruction of the same coefficients also interpolates
    # both ends of an interval and reproduces constants. The translated
    # nonuniform phase must expose this substitution at interior native points.
    def nodal(x):
        return np.stack(((1-x)*(1-2*x), 4*x*(1-x), x*(2*x-1)), axis=-1)

    with patch.object(plot, 'bernstein', nodal):
        try:
            plot.plot_seed(args.translated)
        except AssertionError as error:
            if 'Bernstein phase on cell' not in str(error):
                raise
            print(error)
        else:
            raise ValueError('The wrong nodal basis passed the native Bernstein phase check')


if __name__ == '__main__':
    main()
