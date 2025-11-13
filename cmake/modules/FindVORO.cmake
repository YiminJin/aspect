# Copyright (C) 2020 - 2024 by the authors of the ASPECT code.
#
# This file is part of ASPECT.
#
# ASPECT is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# ASPECT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASPECT; see the file LICENSE.  If not see
# <http://www.gnu.org/licenses/>.

#
# Try to find the voro++ library
#
# This module exports
#
#   VORO_LIBRARIES
#   VORO_INCLUDE_DIRS

set(VORO_DIR "" CACHE PATH "An optional hint to a voro++ installation")
set_if_empty(VORO_DIR "$ENV{VORO_DIR}")

find_path(VORO_INCLUDE_DIR 
  NAMES voro++.hh 
  HINTS ${VORO_DIR}
  PATH_SUFFIXES include/voro++
  )

find_library(VORO_LIBRARY 
  NAMES voro++
  HINTS ${VORO_DIR}
  PATH_SUFFIXES lib
  )

if(VORO_INCLUDE_DIR AND VORO_LIBRARY)
  set(VORO_FOUND TRUE)
  set(VORO_LIBRARIES ${VORO_LIBRARY})
  set(VORO_INCLUDE_DIRS ${VORO_INCLUDE_DIR})
else()
  set(VORO_FOUND FALSE)
endif()
