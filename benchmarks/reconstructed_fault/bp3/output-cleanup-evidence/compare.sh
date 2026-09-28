#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for run in after thin diagnostics production-cadence slip-trigger; do
  for file in accepted_steps.csv stations.csv first_event.csv first_update_maxwell.csv restored_growth.csv; do
    cmp "before-uniform/$file" "$run/$file"
  done
  for file in "$run"/profiles/*.csv; do cmp "before-uniform/profiles/${file##*/}" "$file"; done
  echo "$run: summaries, events, audits and every saved profile exactly match baseline"
done
for run in restart-v5 restart-thin; do
  for file in "$run"/profiles/*.csv; do cmp "before-uniform/profiles/${file##*/}" "$file"; done
  for file in accepted_steps.csv stations.csv first_event.csv; do cmp "before-uniform/$file" "$run/$file"; done
  echo "$run: full history summaries and saved profiles exactly match uninterrupted baseline"
done
