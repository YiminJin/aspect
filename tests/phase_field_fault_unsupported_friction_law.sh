#!/bin/bash

# Apply the standard exception filtering and remove insignificant trailing
# blanks so the new failure reference remains whitespace-clean.
"${0%/*}/cmake/default" "$@" | sed 's/[[:blank:]]*$//'
