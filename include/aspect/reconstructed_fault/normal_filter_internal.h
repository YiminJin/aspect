/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef _aspect_normal_filter_internal_h
#define _aspect_normal_filter_internal_h

#include <aspect/reconstructed_fault/surface_direct_internal.h>
#include <aspect/reconstructed_fault/fault.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace aspect::internal
{
  /** Add one bulk-work sample to the straight fault's filter stiffness.
   * At a Q1 derivative jump, use the arithmetic mean of the two one-sided
   * gradient outer products (not the outer product of averaged gradients).
   * The exterior endpoint trace is zero. Source values/associations are untouched.
   */
  template <int dim>
  void add_normal_filter_stiffness(const ReconstructedFault<dim> &fault,
                                   const Point<dim> &position,
                                   const unsigned int segment,
                                   const Tensor<1,dim> &tangent,
                                   const double weight,
                                   std::vector<double> &diagonal,
                                   std::vector<double> &edge)
  {
    const auto add_segment = [&](const unsigned int j, const double trace_weight)
    {
      const double length=fault.vertex(j).distance(fault.vertex(j+1));
      const double k=trace_weight/(length*length);
      diagonal[j]+=k;diagonal[j+1]+=k;edge[j]-=k;
    };
    const auto on_vertex_plane = [&](const unsigned int vertex)
    {
      const auto offset=position-fault.vertex(vertex);
      // Include subtraction/tangent conditioning from either adjacent segment,
      // so the bound does not depend on which valid source association won.
      double conditioning=0.;
      for (unsigned int j=(vertex==0 ? 0 : vertex-1);
           j<std::min(vertex+1,fault.n_cells());++j)
        conditioning=std::max(conditioning,
          (fault.vertex(j).norm()+fault.vertex(j+1).norm())
          /fault.vertex(j).distance(fault.vertex(j+1)));
      const double roundoff=32.*std::numeric_limits<double>::epsilon()
        *(position.norm()+fault.vertex(vertex).norm()+offset.norm()*(1.+conditioning));
      return std::abs(offset*tangent)<=roundoff;
    };
    for (const unsigned int vertex : {segment,segment+1})
      if (on_vertex_plane(vertex))
        {
          if (vertex>0) add_segment(vertex-1,.5*weight);
          if (vertex<fault.n_cells()) add_segment(vertex,.5*weight);
          return;
        }
    // Away from the roundoff band retain the ordinary one-sided derivative,
    // including zero in a constant endpoint continuation.
    if ((segment==0 && (position-fault.vertex(0))*tangent<0.)
        || (segment+1==fault.n_cells()
            && (position-fault.vertex(segment+1))*tangent>0.))
      return;
    add_segment(segment,weight);
  }

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
