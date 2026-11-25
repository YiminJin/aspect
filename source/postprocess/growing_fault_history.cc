/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/postprocess/growing_fault_history.h>
#include <aspect/material_model/growing_fault.h>
#include <aspect/geometry_model/interface.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>

namespace aspect::Postprocess
{
  template <int dim>
  void GrowingFaultHistory<dim>::initialize()
  {
    AssertThrow(dim == 2 && dynamic_cast<const GrowingFault::Material<dim> *>(&this->get_material_model()),
                ExcMessage("The growing fault history exporter requires the two-dimensional growing fault material."));
  }

  template <int dim>
  std::pair<std::string,std::string> GrowingFaultHistory<dim>::execute(TableHandler &)
  {
    if constexpr(dim==2)
      {
        const auto &m=dynamic_cast<const GrowingFault::Material<2>&>(this->get_material_model());
        const auto &history=m.history();
        const auto comm=this->get_mpi_communicator();
        const auto rank=Utilities::MPI::this_mpi_process(comm);
        const auto path=this->get_output_directory();
        const auto tag=Utilities::int_to_string(history.step,4);
        const auto &fe=this->get_fe();
        const auto &solution=this->get_solution();
        const auto c=this->introspection().component_indices.compositional_fields[m.phase_field()];
        std::ofstream points(path+"points-"+tag+"-r"+std::to_string(rank)+".csv");
        std::ofstream coefficients(path+"fields-"+tag+"-r"+std::to_string(rank)+".csv");
        points<<std::setprecision(17)<<"cell,q,x,y,weight,section,phi,tauxx,tauyy,tauzz,tauxy,Axx,Ayy,Azz,Axy,signed_shear,absolute_shear\n";
        coefficients<<std::setprecision(17)<<"cell,component,local,value\n";
        for (const auto &cell:this->get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const auto &rule=m.sections().cells.at(cell->id());
              FEValues<2> values(this->get_mapping(),fe,rule.quadrature,update_values|update_JxW_values|update_quadrature_points);
              values.reinit(cell);
              std::vector<double> phase(rule.quadrature.size());
              values[FEValuesExtractors::Scalar(c)].get_function_values(solution,phase);
              std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
              cell->get_dof_indices(ids);
              for (unsigned int i=0; i<ids.size(); ++i)
                {
                  const auto [component,j]=fe.system_to_component_index(i);
                  coefficients<<cell->id()<<','<<component<<','<<j<<','<<solution[ids[i]]<<'\n';
                }
              for (unsigned int q=0; q<phase.size(); ++q)
                {
                  const auto &v=history.cells.at(cell->id())[q];
                  const auto &p=values.quadrature_point(q);
                  points<<cell->id()<<','<<q<<','<<p[0]<<','<<p[1]<<','<<values.JxW(q)<<','<<rule.section[q]<<','<<phase[q];
                  for (const auto *tensor:
                  {
                    &v.stress,&v.cohesive_strain
                  })
                  points<<','<<(*tensor)[0][0]<<','<<(*tensor)[1][1]<<','<<(*tensor)[2][2]<<','<<(*tensor)[0][1];
                  points<<','<<v.signed_shear<<','<<v.absolute_shear<<'\n';
                }
            }
        dealii::Utilities::System::MemoryStats memory;
        dealii::Utilities::System::get_memory_stats(memory);
        std::ofstream usage(path+"memory-"+tag+"-r"+std::to_string(rank)+".txt");
        usage<<memory.VmHWM<<" KiB peak RSS, "<<sizeof(GrowingFault::PointHistory)<<" bytes/native point history\n";
        if (rank==0)
          {
            std::ofstream strips(path+"sections-"+tag+".csv");
            const auto &ages=m.friction().ages_in_seconds();
            strips<<std::setprecision(17)<<"section,xi,ds,beta,formed,"<<(ages.empty()?"":"age,")<<"slip,V,force\n";
            for (unsigned int k=0; k<history.sections.size(); ++k)
              {
                const auto &s=history.sections[k];
                strips<<k<<','<<.5*(m.sections().edges[k]+m.sections().edges[k+1])<<','<<m.sections().length(k,s.angle)<<','<<s.angle
                      <<','<<s.formed<<',';
                if (!ages.empty()) strips<<ages[k]<<',';
                strips<<s.slip<<','<<s.rate<<','<<s.resistance<<'\n';
              }
            if (!history.friction_pressures.stations.empty())
              {
                std::ofstream pressures(path+"friction-pressure-"+tag+".csv");
                pressures << std::setprecision(17) << "section,x,y,depth,compression,V\n";
                for (unsigned int k=0; k<history.sections.size(); ++k)
                  {
                    const auto &point = history.friction_pressures.stations[k];
                    pressures << k << ',' << point[0] << ',' << point[1] << ',' << this->get_geometry_model().depth(point)
                              << ',' << history.friction_pressures.values[k] << ',' << history.sections[k].rate << '\n';
                  }
              }
            // The initial seed has no completed step-work ledger.
            if (history.step>0)
              {
                std::ofstream book(path+"balance-"+tag+".csv");
                const auto &b=m.balance();
                book<<std::setprecision(17)<<"step,time,bulk,cohesive,fracture,Wext,Db,Dc,Df,phase_remainder,phase_work,energy_change,mechanical_residual,phase_KKT,adjoint_error,iterations\n"
                    <<history.step<<','<<history.time<<','<<b.bulk<<','<<b.cohesive<<','<<b.fracture<<','<<b.prescribed_velocity_work<<','<<b.bulk_loss<<','<<b.cohesive_loss
                    <<','<<b.friction_work<<','<<b.compliance_remainder<<','<<b.phase_work<<','<<b.storage_change<<','<<b.mechanical<<','<<b.kkt<<','<<b.adjoint_defect<<','<<b.iterations<<'\n';
              }
          }

      }
    return {"Growing fault","native section and point history"};
  }
  ASPECT_REGISTER_POSTPROCESSOR(GrowingFaultHistory,"growing fault history",
                                "Write accepted strip records, native point history and the physical work ledger. "
                                "The native stress columns tauxx, tauyy, tauzz and tauxy are tension-positive "
                                "deviatoric components in Pa at the accepted endpoint.")

}
