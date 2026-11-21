# Bounded material-plane comparison

Use the first-free-turn positive fixture, changing its box repetitions from
128 × 128 to 32 × 16. The current run selects `material strips`. In the paired
input change that one entry to `material strips Coulomb formation`. Start each
from the same fresh horizontal seed. The native driver is
`../bulk_tip/run.py`, using a matching executable and checker. For this pair
the common flags were `--dt .005 --end .10 --ranks 2 --bind-cores
--time-update midpoint --phase-balance --integrity-stop --checkpoint-steps 4
--seconds 480`. The integrity termination and nonlinear failure strategy in
the input apply throughout.

`replay.py OUTPUT --mu .6 --confinement 200 --csv NEW.csv` independently
scans the conjugates at 0.01 degree spacing using each newborn's preceding
accepted stress. Use `--parent-step 7` to sample before the load tilt.
The signed normal traction relative to scalar confinement is a diagnostic of
the orientation experiment. The midpoint pressure multiplier has its own
gauge and time location.

`compare.py CURRENT_RUN COULOMB_RUN --out DIRECTORY` compares the longest
common accepted prefix before either remote/boundary integrity trigger.
It uses native cellwise polynomials for the absolute phase contour, checks
formed-plane retention and age/slip continuity, and measures both signed and
accumulated absolute physical-work errors. The JSON retains the last accepted
step of each run even when the common comparison ends earlier.
