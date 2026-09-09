# Diagnostic only: f4032b182/current pre-ownership source line numbers.
# Override the debuggee's local tolerance, never the source or polygon data.
set pagination off
set confirm off
break /home/ein/repository/aspect/source/particle/particle_domain.cc:624
commands
silent
set var tol = $old_tol
continue
end
