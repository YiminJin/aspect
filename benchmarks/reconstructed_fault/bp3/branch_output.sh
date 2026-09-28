#!/usr/bin/env bash
# Prepare a NEW output branch from a selected ordinary BP3 checkpoint.
# Never edit the parent. Physics/history come from the checkpoint, not CSVs.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "Usage: $0 PARENT CHECKPOINT_ID NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
parent=$(cd "$1" && pwd)
printf -v id '%02d' "$2"
branch=$3
checkpoint="$parent/restart/$id"
metadata="$checkpoint/bp3_output_metadata"
[[ ! -e "$branch" && -f "$checkpoint/bp3_accepted_state.txt" && -d "$metadata" ]]
read -r step time < "$checkpoint/bp3_accepted_state.txt"
# Validate every referenced payload before creating the branch.
if [[ -f "$metadata/profiles.csv" ]]; then
  while IFS=, read -r saved clock file rest; do
    [[ "$saved" == step ]] && continue
    [[ "$saved" =~ ^[0-9]+$ && "$saved" -le "$step" ]]
    [[ "$file" == profiles/fault_*.csv && "$file" != *..* && -f "$parent/$file" ]]
  done < "$metadata/profiles.csv"
fi
mkdir -p "$branch/restart"
cp -a "$checkpoint" "$branch/restart/$id"
printf '%s\n' "$((10#$id))" > "$branch/restart/last_good_checkpoint.txt"
cp -a "$metadata/." "$branch/"
if [[ -f "$metadata/profiles.csv" ]]; then
  mkdir "$branch/profiles"
  while IFS=, read -r saved clock file rest; do
    [[ "$saved" == step ]] && continue
    cp -p "$parent/$file" "$branch/$file"
  done < "$metadata/profiles.csv"
fi
# Preserve native payload references once at branch creation. Later unindexed
# native files may also be copied; native writers reuse their checkpoint clock.
# Ordinary checkpoints themselves copy only small metadata, never these trees.
for directory in solution particles reconstructed_faults; do
  if [[ -d "$parent/$directory" ]]; then cp -a "$parent/$directory" "$branch/"; fi
done
echo "Prepared $branch from accepted step $step at $time s; parent preserved."
