#!/bin/bash

"${0%/*}/cmake/default" "$@" | sed 's/[[:blank:]]*$//'
