#!/bin/sh
# Scientific checks are explicit C++ assertions; matching these lines is secondary.
awk '
/^P1 scratch residual\/Jacobian\/roundoff isolation verified$/ || /^P1 normal restoration verified$/ || /^P1 exceptional restoration verified$/ {print}
/P1 controlled phase history:/ && /qualified for prescribed H only/ {print "P1 controlled phase history: qualified for prescribed H only"}
'
