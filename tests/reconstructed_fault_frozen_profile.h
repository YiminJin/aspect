/* Test-only exact prolongation of a captured Cartesian Q1 phase field. */
#ifndef aspect_test_reconstructed_fault_frozen_profile_h
#define aspect_test_reconstructed_fault_frozen_profile_h

#include "phase_field_fault_test_access.h"
#include <deal.II/base/quadrature_lib.h>
#include <map>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace aspect
{
  namespace FrozenMechanicalProfile
  {
    struct Cell
    {
      double x, y, h;
      std::array<double,4> phi;
      double value(const Point<2> &p) const
      {
        const double a=(p[0]-x)/h,b=(p[1]-y)/h;
        AssertThrow(a>=-1e-10 && a<=1.+1e-10 && b>=-1e-10 && b<=1.+1e-10,
                    ExcMessage("Frozen profile lookup left its coarse parent."));
        return (1-a)*(1-b)*phi[0]+a*(1-b)*phi[1]+(1-a)*b*phi[2]+a*b*phi[3];
      }
    };
    inline std::map<std::string,Cell> cells;
    inline std::vector<double> normalization;
    inline std::vector<Point<2>> vertices;

    inline void load()
    {
      if (!cells.empty()) return;
      const char *path=std::getenv("ASPECT_MECHANICAL_FROZEN_PROFILE");
      AssertThrow(path,ExcMessage("Missing captured profile path."));
      std::ifstream in(std::string(path)+"/phase_cells.csv");
      AssertThrow(in,ExcMessage("Missing double-precision coarse phase snapshot."));
      std::string line;std::getline(in,line);
      while (std::getline(in,line))
        {
          std::replace(line.begin(),line.end(),',',' ');
          std::istringstream row(line);std::string id;Cell c;
          AssertThrow(row>>id>>c.x>>c.y>>c.h>>c.phi[0]>>c.phi[1]>>c.phi[2]>>c.phi[3],
                      ExcMessage("Malformed coarse phase snapshot."));
          AssertThrow(cells.emplace(id,c).second,ExcMessage("Duplicate coarse cell."));
        }
      std::ifstream surface(std::string(path)+"/surface.csv");
      AssertThrow(surface,ExcMessage("Missing frozen normalization snapshot."));
      std::getline(surface,line);
      while (std::getline(surface,line))
        {
          std::replace(line.begin(),line.end(),',',' ');
          std::istringstream row(line);unsigned int node;Point<2> p;double ih;
          AssertThrow(row>>node>>p[0]>>p[1]>>ih,ExcMessage("Malformed frozen surface."));
          AssertThrow(node==vertices.size(),ExcMessage("Frozen surface ordering changed."));
          vertices.push_back(p);normalization.push_back(ih);
        }
    }

    inline const Cell &parent(std::string id)
    {
      while (!cells.count(id))
        {
          const auto colon=id.find(':');const auto digits=id.substr(colon+1);
          AssertThrow(!digits.empty(),ExcMessage("Fine mesh is not nested in captured coarse mesh."));
          id=id.substr(0,id.find('_'))+"_"+std::to_string(digits.size()-1)+":"+digits.substr(0,digits.size()-1);
        }
      return cells.at(id);
    }

    template <int dim>
    void constraints(const SimulatorAccess<dim> &sim,AffineConstraints<double> &constraints)
    {
      if (!std::getenv("ASPECT_MECHANICAL_FROZEN_PROFILE")
          || sim.get_pre_refinement_step()<sim.get_parameters().initial_adaptive_refinement) return;
      if constexpr (dim==2)
        {
          load();
          const auto &fe=sim.get_fe();
          const unsigned int phi=sim.introspection().variable("phase_field").first_component_index;
          std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());
          for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
            if (!cell->is_artificial())
              {
                const auto coarse=parent(cell->id().to_string());cell->get_dof_indices(dofs);
                for (unsigned int j=0;j<dofs.size();++j)
                  if (fe.system_to_component_index(j).first==phi && constraints.can_store_line(dofs[j]))
                    {
                      AssertThrow(constraints.is_constrained(dofs[j]),ExcMessage("BP3 phase constraint must precede prolongation."));
                      // Retain hanging-node interpolation; replace only the
                      // analytical Dirichlet values by the captured Q1 field.
                      if (constraints.get_constraint_entries(dofs[j])->empty())
                        constraints.set_inhomogeneity(dofs[j],coarse.value(sim.get_mapping().transform_unit_to_real_cell(cell,fe.get_unit_support_points()[j])));
                    }
              }
        }
    }

    template <int dim>
    void retain_normalization(const SimulatorAccess<dim> &sim,bool temperature,unsigned int,const SolverControl &)
    {
      if (!temperature || !std::getenv("ASPECT_MECHANICAL_FROZEN_PROFILE")) return;
      if constexpr (dim==2)
        {
          load();auto &manager=sim.get_reconstructed_fault_manager();
          auto &model=const_cast<MaterialModel::PhaseFieldFault<dim>&>(
            Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model()));
          const auto &fault=manager.get_fault(0);
          AssertThrow(fault.n_vertices()==vertices.size(),ExcMessage("Fault grid changed."));
          using Access=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
          const auto computed=Access::current_normalization_integrals(model);
          AssertThrow(computed.size()==1 && computed[0].size()==normalization.size(),ExcMessage("BP3 preparation must precede frozen normalization."));
          const unsigned int previous=manager.get_property_information()[manager.get_property_index("phase field fault previous I h")].position;
          double discrepancy=0.;
          for (unsigned int v=0;v<vertices.size();++v)
            {
              AssertThrow(fault.vertex(v)==vertices[v],ExcMessage("Fault coordinates changed."));
              discrepancy=std::max(discrepancy,std::abs(computed[0][v]/normalization[v]-1.));
              manager.get_fault(0).get_properties(v)[previous]=normalization[v];
            }
          // The valid fine-mesh cache key remains intact. The experimental
          // coefficient is the saved coarse I_h, not a regenerated projection.
          Access::restore_diagnostic_normalization(model,{normalization});
          sim.get_pcout()<<std::setprecision(17)<<"Frozen I_h retained exactly; unused recomputation relative difference="<<discrepancy<<std::endl;
        }
    }

    template <int dim>
    void snapshot_or_verify(const SimulatorAccess<dim> &sim)
    {
      const bool width_probe=std::getenv("ASPECT_MECHANICAL_WIDTH_PROBE");
      const bool save=std::getenv("ASPECT_MECHANICAL_EXPORT_PROFILE") || width_probe;
      const bool verify=std::getenv("ASPECT_MECHANICAL_FROZEN_PROFILE");
      if (!save && !verify) return;
      if constexpr (dim==2)
        {
          const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model());
          using Access=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
          const auto &ih=Access::current_normalization_integrals(model);
          const auto &fault=sim.get_reconstructed_fault_manager().get_fault(0);
          const unsigned int phi=sim.introspection().variable("phase_field").first_component_index;
          const auto comm=sim.get_mpi_communicator();const auto rank=Utilities::MPI::this_mpi_process(comm);
          std::ofstream out;
          if (save)
            {out.open(sim.get_output_directory()+"phase_cells_rank"+std::to_string(rank)+".csv");out<<std::setprecision(17)<<"cell,x,y,h,phi0,phi1,phi2,phi3\n";}
          QTrapezoid<dim> corners;
          FEValues<dim> corner_values(sim.get_mapping(),sim.get_fe(),corners,update_values);
          FEValues<dim> fe(sim.get_mapping(),sim.get_fe(),sim.introspection().quadratures.velocities,update_values|update_quadrature_points|update_JxW_values);
          std::vector<double> values(corners.size()),qp(fe.n_quadrature_points);
          double error=0.,integral=0.,square=0.,reference_integral=0.,hmin=1e100,hmax=0.;
          unsigned long long count=0;
          if (verify) {load();AssertThrow(ih==std::vector<std::vector<double>>{normalization},ExcMessage("Frozen I_h changed before mechanics."));}
          for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                corner_values.reinit(cell);corner_values[FEValuesExtractors::Scalar(phi)].get_function_values(sim.get_current_linearization_point(),values);
                if (save)
                  {out<<cell->id()<<','<<cell->vertex(0)[0]<<','<<cell->vertex(0)[1]<<','<<cell->diameter()/std::sqrt(2.);for (double x:values) out<<','<<x;out<<'\n';}
                fe.reinit(cell);fe[FEValuesExtractors::Scalar(phi)].get_function_values(sim.get_current_linearization_point(),qp);
                for (unsigned int q=0;q<qp.size();++q)
                  {
                    const auto p=fe.quadrature_point(q);
                    integral+=fe.JxW(q)*qp[q];square+=fe.JxW(q)*qp[q]*qp[q];
                    if (verify)
                      {
                        const double expected=parent(cell->id().to_string()).value(p);
                        error=std::max(error,std::abs(expected-qp[q]));reference_integral+=fe.JxW(q)*expected;
                      }
                    const double xd=(100000.-p[1])/std::sin(numbers::PI/3.);
                    if (xd>15000. && xd<18000. && qp[q]>0.)
                      {const double h=cell->diameter()/std::sqrt(2.);hmin=std::min(hmin,h);hmax=std::max(hmax,h);++count;}
                  }
              }
          error=Utilities::MPI::max(error,comm);integral=Utilities::MPI::sum(integral,comm);square=Utilities::MPI::sum(square,comm);
          reference_integral=Utilities::MPI::sum(reference_integral,comm);hmin=Utilities::MPI::min(hmin,comm);hmax=Utilities::MPI::max(hmax,comm);count=Utilities::MPI::sum(count,comm);
          AssertThrow(!verify || (error<2e-12 && std::abs(hmin-48.828125)<1e-10 && std::abs(hmax-48.828125)<1e-10),ExcMessage("Fine profile/spacing preservation failed."));
          if (!rank)
            {
              std::ofstream check(sim.get_output_directory()+"frozen_profile_check.csv");
              check<<std::setprecision(17)<<"cells,phi_max_error,phi_integral,phi_squared_integral,reference_phi_integral,patch_hmin,patch_hmax,patch_nonzero_qps\n"
                   <<sim.get_triangulation().n_global_active_cells()<<','<<error<<','<<integral<<','<<square<<','<<reference_integral<<','<<hmin<<','<<hmax<<','<<count<<'\n';
              if (save)
                {
                  std::ofstream surface(sim.get_output_directory()+"surface.csv");surface<<std::setprecision(17)<<"node,x,y,Ih\n";
                  for (unsigned int v=0;v<fault.n_vertices();++v) surface<<v<<','<<fault.vertex(v)[0]<<','<<fault.vertex(v)[1]<<','<<ih[0][v]<<'\n';
                  if (width_probe)
                    {
                      const auto profiles=sim.get_phase_field_handler().get_phase_field_profiles(.6);
                      const auto &r=profiles[0]->get_coordinate_values();
                      const auto &p=profiles[0]->get_phase_field_values();
                      std::ofstream stationary(sim.get_output_directory()+"stationary_profile.csv");
                      stationary<<std::setprecision(17)<<"r,phi\n";
                      for (unsigned int j=0;j<r.size();++j) stationary<<r[j]<<','<<p[j]<<'\n';
                    }
                }
            }
          out.close();
          if (save && !width_probe)
            {sim.get_pcout()<<"FROZEN PROFILE SNAPSHOT VERIFIED; intentional stop before publication."<<std::endl;MPI_Barrier(comm);AssertThrow(false,ExcMessage("Intentional frozen-profile snapshot stop."));}
          if (verify) sim.get_pcout()<<"FROZEN PROFILE PROLONGATION VERIFIED; maximum phi error="<<error<<std::endl;
        }
    }
  }
}
#endif
