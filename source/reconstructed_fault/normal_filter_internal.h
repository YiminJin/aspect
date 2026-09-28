/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef _aspect_normal_filter_internal_h
#define _aspect_normal_filter_internal_h

#include "surface_direct_internal.h"
#include <memory>

namespace aspect::internal
{
  /** One full supported open fault. No slip constraints, recentering, or
   * artificial diagonal shift. The caller assembles physical work moments. */
  class FaultNormalFilter
  {
    public:
      FaultNormalFilter(const std::vector<double> &mass,
                        const std::vector<double> &mass_edge,
                        const std::vector<double> &stiffness,
                        const std::vector<double> &stiffness_edge,
                        const double length, const unsigned int fault)
        : diagonal(mass), edge(mass_edge)
      {
        AssertDimension(mass.size(),stiffness.size());
        AssertDimension(mass_edge.size(),stiffness_edge.size());
        for (unsigned int i=0;i<mass.size();++i)
          {
            AssertThrow(mass[i]>0.,dealii::ExcMessage("Normal filter has an unsupported fault row: fault "
                        +std::to_string(fault)+", vertex "+std::to_string(i)));
            diagonal[i]+=length*length*stiffness[i];
          }
        for (unsigned int i=0;i<edge.size();++i) edge[i]+=length*length*stiffness_edge[i];
        factor=std::make_unique<FaultSurfaceDirect>(diagonal,edge,std::vector<bool>(mass.size(),false),fault);
      }

      bool matches(const std::vector<double> &mass,const std::vector<double> &mass_edge,
                   const std::vector<double> &stiffness,const std::vector<double> &stiffness_edge,
                   const double length) const
      {
        if (mass.size()!=diagonal.size() || mass_edge.size()!=edge.size()) return false;
        for (unsigned int i=0;i<mass.size();++i)
          if (mass[i]+length*length*stiffness[i]!=diagonal[i]) return false;
        for (unsigned int i=0;i<edge.size();++i)
          if (mass_edge[i]+length*length*stiffness_edge[i]!=edge[i]) return false;
        return true;
      }

      std::vector<double> solve(const std::vector<double> &rhs) const
      {
        std::vector<double> result;factor->solve(rhs,result);return result;
      }

    private:
      std::vector<double> diagonal,edge;
      std::unique_ptr<FaultSurfaceDirect> factor;
  };
}
#endif
