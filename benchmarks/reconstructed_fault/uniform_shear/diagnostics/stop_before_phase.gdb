# Used after old_dedup.gdb, before any phase-field residual is evaluated.
break /home/ein/repository/aspect/source/simulator/phase_field.cc:1032
commands
silent
printf "OLD_DEDUP_BEFORE_PHASE_COMPLETE\n"
quit
end
run
