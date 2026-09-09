set pagination off
set confirm off
set print elements 24
set print pretty on
set breakpoint pending on
if $_isvoid($target_particle)
set $target_particle = 3836
end
break /home/ein/repository/aspect/source/particle/particle_domain.cc:1060 if voronoi_cell.particle_index == $target_particle
commands 1
silent
printf "TARGET_POLYGON\n"
print voronoi_cell
enable 2
enable 3
enable 4
disable 1
continue
end
break /home/ein/repository/aspect/source/particle/particle_domain.cc:1103
disable 2
commands 2
silent
printf "VERTEX_SAMPLE %u\n", v
print vertex_unit
continue
end
break /home/ein/repository/aspect/source/particle/particle_domain.cc:1114
disable 3
commands 3
silent
printf "CENTROID_SAMPLE\n"
print center_unit
continue
end
break /home/ein/repository/aspect/source/particle/particle_domain.cc:1183
disable 4
commands 4
silent
printf "PRODUCTION_STENCIL\n"
print weighting_function_data
printf "CPDI_CAUSE_PROBE_COMPLETE: stop during particle generation, before phase solve\n"
quit
end
run
