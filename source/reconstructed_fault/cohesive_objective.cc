/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/reconstructed_fault/cohesive_system_internal.h>
#include <deal.II/fe/fe_values.h>

namespace aspect { namespace CohesiveInterface
{
  System::Implementation::State
  System::Implementation::objective_anchor(const State &parent) const
  {
    State anchor=parent;
    if(!jaumann_stress)return anchor;
    const double dt=system.get_timestep();
    const auto gradient=[&](const BulkHistory<2>::Cell &cell,const Quadrature<2> &rule)
      {
        FEValues<2> fe(system.get_mapping(),system.get_fe(),rule,
                       update_gradients|update_quadrature_points);
        fe.reinit(cell);
        std::vector<dealii::Tensor<2,2>> L(rule.size());
        fe[system.introspection().extractors.velocities].get_function_gradients(system.get_old_solution(),L);
        for(unsigned int q=0;q<rule.size();++q)
          for(const auto &m:parent.geometry.lifting(fe.quadrature_point(q)))
            L[q]+=parent.rate.at(m.index)*m.gradient;
        return L;
      };
    const auto physical=[&](const BulkHistory<2>::Cell &cell,const Quadrature<2> &rule)
      {return physical_stress(parent,cell,rule);};
    const Geometry formed_geometry(std::vector<Geometry::Piece>(parent.geometry.pieces().begin(),
      parent.geometry.pieces().begin()+parent.formed),parent.geometry.width());
    anchor.bulk=parent.bulk->cut_rotated_anchor(physical,gradient,dt,8,formed_geometry);
    // Measure projection energy independently before the constitutive solve.
    // Exact point rotation has zero power. This is the actual numerical
    // projection defect, never viscous/frictional dissipation or a closing term.
    long double change=0.;
    double maximum_angle=0.,maximum_change=0.;
    for(const auto &cell:system.get_dof_handler().active_cell_iterators())
      if(cell->is_locally_owned())
        {
          const auto rule=parent.geometry.quadrature(cell->vertex(0),cell->vertex(3),bulk_order);
          FEValues<2> fe(system.get_mapping(),system.get_fe(),rule,update_JxW_values);
          fe.reinit(cell);
          const auto before=physical_stress(parent,cell,rule),after=physical_stress(anchor,cell,rule);
          const auto L=gradient(cell,rule);
          for(unsigned int q=0;q<rule.size();++q)
            {
              change+=static_cast<long double>(fe.JxW(q))*((after[q]+before[q])*(after[q]-before[q]))/(4*law.G);
              maximum_angle=std::max(maximum_angle,.5*dt*std::abs(L[q][1][0]-L[q][0][1]));
              maximum_change=std::max(maximum_change,(after[q]-before[q]).norm());
            }
        }
    const auto comm=system.get_mpi_communicator();
    const double defect=Utilities::MPI::sum(static_cast<double>(change),comm);
    const auto previous=anchor.diagnostics.find("bulk_J_m");
    if(previous!=anchor.diagnostics.end())previous->second+=defect;
    anchor.diagnostics["objective_projection_J_m"]=defect;
    anchor.diagnostics["objective_max_angle_rad"]=Utilities::MPI::max(maximum_angle,comm);
    anchor.diagnostics["objective_max_change_Pa"]=Utilities::MPI::max(maximum_change,comm);
    return anchor;
  }
}}
