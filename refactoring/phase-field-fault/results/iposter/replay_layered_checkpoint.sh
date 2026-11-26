#!/usr/bin/env bash
# Resume the pre-Assignment-D default and check its accepted 25 ms frame.
set -eu
prefix=$(realpath "${1:?installed ASPECT prefix}")
checkpoint=$(realpath "${2:?old 12.5 ms checkpoint generation 01}")
reference=$(realpath "${3:?uninterrupted public output directory}")
replay=$(realpath -m "${4:?new scratch directory}")
example="$prefix/benchmarks/layered_moving_fault"
export ASPECT_SOURCE_DIR="$prefix" DEAL_II_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir "$replay"
for mode in same changed; do
    dir="$replay/$mode"
    mkdir -p "$dir/output/restart"
    cp -R "$checkpoint" "$dir/output/restart/01"
    cat >"$dir/resume.prm" <<EOF
include $example/layered.prm
set Additional shared libraries = $example/liblayered_moving_fault.release.so
set Output directory = $dir/output
set Resume computation = true
subsection Checkpointing
  set Resume checkpoint = 1
end
EOF
done
cat >>"$replay/changed/resume.prm" <<'EOF'
subsection Postprocess
  subsection Layered background
    set Wall velocity contrast = 1
  end
end
EOF
(cd "$replay/same" && "$prefix/bin/aspect" resume.prm >run.log 2>&1)
for frame in material-2-rank0.csv material-2-surface-0.csv background-2.csv load-2.txt; do
    cmp "$reference/$frame" "$replay/same/output/$frame"
done
# A checkpoint-only restart writes only its new accepted clock.
tail -1 "$reference/layered-background.csv" >"$replay/expected-clock.csv"
sed -n '2p' "$replay/same/output/layered-background.csv" >"$replay/restarted-clock.csv"
cmp "$replay/expected-clock.csv" "$replay/restarted-clock.csv"
if (cd "$replay/changed" && "$prefix/bin/aspect" resume.prm >run.log 2>&1); then
    echo 'A different wall contrast was accepted' >&2
    exit 1
fi
grep -F 'Layered background checkpoint identity mismatch.' "$replay/changed/run.log"
test ! -e "$replay/changed/output/material-2-surface-0.csv"
echo 'Default continuation matched the public 25 ms frame. Changed contrast was rejected.'
