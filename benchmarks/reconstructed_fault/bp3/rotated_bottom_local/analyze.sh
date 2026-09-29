#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p analysis
for name in A B A-half B-half; do
  run="output-$name"
  [[ -f "$run/local_metrics.csv" ]] || continue
  printf '%s\n' 'case,step,time,deep_rms,deep_rate_rms,deep_max,deep_max_xd,deep_Vmin,deep_Vmax,interior_rms,interior_rate_rms,interior_max,interior_max_xd,interior_Vmin,interior_Vmax,upper_rms,upper_rate_rms,upper_max,upper_max_xd,upper_Vmin,upper_Vmax' > "analysis/$name-norms.csv"
  while IFS=, read -r step time rest; do
    [[ "$step" == step ]] && continue
    awk -v name="$name" -v step="$step" -v time="$time" -f integrate_metrics.awk "$run/local_fault_$step.csv" >> "analysis/$name-norms.csv"
    last_step=$step
  done < "$run/local_metrics.csv"
  cp "$run/local_fault_$last_step.csv" "analysis/$name-final-fault.csv"
  { head -1 "$run/bottom_${last_step}_rank0.csv"
    awk 'FNR>1' "$run"/bottom_"$last_step"_rank*.csv | sort -t, -k1,1g
  } > "analysis/$name-final-bottom.csv"
  cp "$run/local_metrics.csv" "analysis/$name-metrics.csv"
  cp "$run/corner_metrics.csv" "analysis/$name-corners.csv"
  cp "$run/accepted_steps.csv" "analysis/$name-solves.csv"
done
gnuplot plots.gnuplot
gnuplot -e 'pdf=1' plots.gnuplot
