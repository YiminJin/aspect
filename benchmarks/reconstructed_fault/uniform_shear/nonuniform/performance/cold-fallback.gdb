set pagination off
set confirm off
set debuginfod enabled off
set breakpoint pending on
break _ZN6dealii9GridTools29find_active_cell_around_pointILi2ENS_13TriangulationELi2EEESt4pairINT0_IXT_EXT1_EE20active_cell_iteratorENS_5PointIXT_EdEEERKNS_7MappingIXT_EXT1_EEERKS5_RKNS7_IXT1_EdEERKSt6vectorIbSaIbEEd
run --test '[.fault_ih_performance]'
printf "Captured unaccelerated lookup at entry (SysV AMD64: hidden result, mapping, mesh, point):\n"
x/4i $pc
printf "point=(%.17g, %.17g), tolerance=%.17g\n", *(double*)$rcx, *((double*)$rcx+1), $xmm0.v2_double[0]
bt 5
kill
quit
