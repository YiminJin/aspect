#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
printf 'run,category,files,bytes\n'
for run in before-uniform after thin diagnostics production-cadence; do
  find "$run" -type f -printf '%P\t%s\n' | awk -F '\t' -v run="$run" '
    {p=$1; c="metadata_logs_inputs";
     if(p ~ /^restart\//) c="checkpoints";
     else if(p ~ /^profiles\// || p=="profiles.csv") c="profiles";
     else if(p=="cumulative_slip.csv") c="legacy_slip";
     else if(p ~ /^(solution|particles|reconstructed_faults)(\/|\.)/) c="native_visualization";
     else if(p ~ /^(accepted_steps|stations|restored_growth|first_event|heavy_outputs|first_update_maxwell|velocity_constraints|ih_bottom_completion_rank[0-9]+)\.csv$/) c="summaries_required_audits";
     else if(p=="audit_map_size.csv") c="test_instrumentation";
     else if(p ~ /\.csv$/) c="diagnostics";
     n[c]++;size[c]+=$2}
    END {for(c in n) printf "%s,%s,%d,%.0f\n",run,c,n[c],size[c]}' | sort
 done
