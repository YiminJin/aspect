# x86-64 System V ABI, installed Release executable with its symbol table.
# Save the five scalar arguments on entry. At the first subsequent exception
# print the last call and its throwing stack. No change to the trial state.
set pagination off
set confirm off
set print thread-events off
maint set worker-threads 1
set $seen=0
# SectionRoot uses a hidden result pointer. The history object is in rsi
# and its unsigned material ID is in edx. Scalar arguments use xmm0..4.
set $strip=-1
break *_ZNK6aspect13MaterialModel8Rheology16RateStateHistory11solve_stripEjdddd
commands
  silent
  set $strip=(unsigned int)$edx
  continue
end
break *_ZNK6aspect13MaterialModel8Rheology20RateAndStateFriction11solve_stripEddddd
commands
  silent
  set $seen=1
  set $v=$xmm0.v2_double[0]
  set $m=$xmm1.v2_double[0]
  set $a=$xmm2.v2_double[0]
  set $h=$xmm3.v2_double[0]
  set $p=$xmm4.v2_double[0]
  continue
end
catch throw
commands
  silent
  if $seen
    printf "Last root free=%.17g M=%.17g age=%.17g h=%.17g pressure=%.17g\n",$v,$m,$a,$h,$p
    printf "Last strip %u\n",$strip
    python print(next(line.strip() for line in open("/proc/%d/status" % gdb.selected_inferior().pid) if line.startswith("VmHWM:")), flush=True)
    bt 6
  end
  continue
end
run
