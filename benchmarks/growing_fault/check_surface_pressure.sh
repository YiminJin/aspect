#!/usr/bin/env bash
# Compare release surface normalization and its checkpoint guard outside the fault solver.
# Usage: bash check_surface_pressure.sh OLD_ASPECT NEW_ASPECT tests/simpler_box.prm FRESH_DIRECTORY
set -eu
test "$#" = 4
old=$(realpath "$1")
new=$(realpath "$2")
fixture=$(realpath "$3")
mkdir "$4"
scratch=$(realpath "$4")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
for p in 0 100; do
  for version in old new; do
    exe=$new
    if [ "$version" = old ]; then exe=$old; fi
    case_dir=$scratch/$version$p
    mkdir -p "$case_dir"
    sed '/^set Surface pressure/d' "$fixture" > "$case_dir/input.prm"
    if [ "$p" = 100 ]; then echo 'set Surface pressure = 100' >> "$case_dir/input.prm"; fi
    cat >> "$case_dir/input.prm" <<'PRM'
set Output directory = output
set Pressure normalization = surface
subsection Mesh refinement
  set Initial global refinement = 2
end
subsection Postprocess
  subsection Visualization
    set Output format = gnuplot
    set List of output variables = adiabat
  end
end
PRM
    (cd "$case_dir" && "$exe" input.prm > log 2>&1)
  done
  cmp "$scratch/old$p/output/statistics" "$scratch/new$p/output/statistics"
  for version in old new; do
    awk 'NF && !/^#/ {print $5}' "$scratch/$version$p/output/solution/solution-00000.0000.gnuplot" > "$scratch/$version$p/p"
  done
  cmp "$scratch/old$p/p" "$scratch/new$p/p"
  # Simpler material, no fault. The stock compute-profile plugin reads the
  # surface value through SimulatorAccess and the visualization samples it.
  awk -v expected="$p" '/^# <x>/ {for(i=2;i<=NF;i++) if($i=="<adiabatic_pressure>") col=i-1}
    NF && !/^#/ && $2==1 {if(!col || $col!=expected) exit 1; found++}
    END {if(!found) exit 1; print found,"top samples, plugin surface pressure",expected,"Pa"}' \
    "$scratch/new$p/output/solution/solution-00000.0000.gnuplot"
done
paste "$scratch/new0/p" "$scratch/new100/p" | awk '{e=$2-$1-100; if(e<0)e=-e; if(e>max)max=e} END {print NR,"pressure samples, max printed shift error",max+0,"Pa"; exit(NR==0 || max>0.0006)}'

mkdir -p "$scratch/writer"
cp "$scratch/old100/input.prm" "$scratch/writer/input.prm"
cat >> "$scratch/writer/input.prm" <<'PRM'
set End time = .005
set Maximum first time step = .005
subsection Checkpointing
  set Steps between checkpoint = 1
  set Number of checkpoints to keep = 2
end
PRM
(cd "$scratch/writer" && "$old" input.prm > log 2>&1)
for p in 100 0; do
 target="$scratch/restore$p"
 mkdir -p "$target/output/restart"
 cp -a "$scratch/writer/output/restart/01" "$target/output/restart/"
 cp "$scratch/writer/input.prm" "$target/input.prm"
 cat >> "$target/input.prm" <<PRM
set Surface pressure = $p
set Resume computation = true
subsection Checkpointing
  set Resume checkpoint = 1
end
PRM
 code=0
 (cd "$target" && "$new" input.prm > log 2>&1) || code=$?
 if [ "$p" = 100 ]; then test "$code" = 0; else
   test "$code" != 0
   grep -Fq 'surface_pressure == parameters.surface_pressure' "$target/log"
 fi
done
echo "Stock checkpoint at 100 Pa continued at 100 Pa and rejected a switch to 0 Pa."
