/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/postprocess/growing_fault_history.h>
#include <aspect/material_model/growing_fault.h>
#include <aspect/geometry_model/interface.h>
#include <deal.II/fe/fe_values.h>
#include <filesystem>
#include <fstream>
#include <regex>

namespace aspect::Postprocess
{
  namespace
  {
    // Existing readers glob the rank-local files for a given step. When a
    // frame is rewritten, retire only this exporter's obsolete files for it.
    void remove_obsolete_files(const std::string &directory, const std::string &step,
                               const unsigned int n_ranks, const bool has_pressures, const bool has_balance)
    {
      const std::regex shard("(points|fields|memory)-"+step+"-r([1-9][0-9]*)\\.(csv|txt)");
      for (const auto &entry : std::filesystem::directory_iterator(directory))
        {
          const std::string name = entry.path().filename().string();
          std::smatch match;
          unsigned int owner;
          const bool obsolete_shard = (std::regex_match(name, match, shard)
                                       && ((match[1]=="memory") == (match[3]=="txt"))
                                       && (std::istringstream(match[2].str()) >> owner)
                                       && owner >= n_ranks);
          if (obsolete_shard || (!has_pressures && name=="friction-pressure-"+step+".csv")
              || (!has_balance && name=="balance-"+step+".csv"))
            {
              AssertThrow(!entry.is_directory(), ExcMessage("Cannot remove directory "+entry.path().string()));
              std::filesystem::remove(entry.path());
            }
        }
    }
  }

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
        const auto n_ranks = Utilities::MPI::n_mpi_processes(comm);
        const std::string suffix = tag+"-r"+std::to_string(rank);
        const std::string point_file = path+"points-"+suffix+".csv";
        const std::string field_file = path+"fields-"+suffix+".csv";
        const std::string section_file = path+"sections-"+tag+".csv";
        bool rewriting = false;
        std::string error;
        if (rank==0)
          try
            {
              // A previous writer may have stopped before writing its strip
              // table. Check for its point file and the first surplus rank.
              const std::string surplus = tag+"-r"+std::to_string(n_ranks);
              rewriting = std::filesystem::exists(section_file) || std::filesystem::exists(point_file)
                          || std::filesystem::exists(path+"points-"+surplus+".csv")
                          || std::filesystem::exists(path+"fields-"+surplus+".csv")
                          || std::filesystem::exists(path+"memory-"+surplus+".txt");
            }
          catch (const std::filesystem::filesystem_error &exc)
            {
              error = exc.what();
            }
        const auto finish_file = [&error](std::ofstream &out, const std::string &filename)
        {
          // Opening, a buffered write, and the final flush can each fail.
          out.close();
          if (!out && error.empty())
            error = "Could not write or close "+filename;
        };
        std::ofstream points(point_file);
        std::ofstream coefficients(field_file);
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
        finish_file(points, point_file);
        finish_file(coefficients, field_file);
        dealii::Utilities::System::MemoryStats memory;
        dealii::Utilities::System::get_memory_stats(memory);
        const std::string memory_file = path+"memory-"+suffix+".txt";
        std::ofstream usage(memory_file);
        usage<<memory.VmHWM<<" KiB peak RSS, "<<sizeof(GrowingFault::PointHistory)<<" bytes/native point history\n";
        finish_file(usage, memory_file);
        if (rank==0)
          {
            std::ofstream strips(section_file);
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
            finish_file(strips, section_file);
            if (!history.friction_pressures.stations.empty())
              {
                const std::string pressure_file = path+"friction-pressure-"+tag+".csv";
                std::ofstream pressures(pressure_file);
                pressures << std::setprecision(17) << "section,x,y,depth,compression,V\n";
                for (unsigned int k=0; k<history.sections.size(); ++k)
                  {
                    const auto &point = history.friction_pressures.stations[k];
                    pressures << k << ',' << point[0] << ',' << point[1] << ',' << this->get_geometry_model().depth(point)
                              << ',' << history.friction_pressures.values[k] << ',' << history.sections[k].rate << '\n';
                  }
                finish_file(pressures, pressure_file);
              }
            // The seed and some older native checkpoints have no work ledger.
            if (m.has_accepted_balance())
              {
                const std::string balance_file = path+"balance-"+tag+".csv";
                std::ofstream book(balance_file);
                const auto &b=m.balance();
                book<<std::setprecision(17)<<"step,time,bulk,cohesive,fracture,Wext,Db,Dc,Df,phase_remainder,phase_work,energy_change,mechanical_residual,phase_KKT,adjoint_error,iterations\n"
                    <<history.step<<','<<history.time<<','<<b.bulk<<','<<b.cohesive<<','<<b.fracture<<','<<b.prescribed_velocity_work<<','<<b.bulk_loss<<','<<b.cohesive_loss
                    <<','<<b.friction_work<<','<<b.compliance_remainder<<','<<b.phase_work<<','<<b.storage_change<<','<<b.mechanical<<','<<b.kkt<<','<<b.adjoint_defect<<','<<b.iterations<<'\n';
                finish_file(book, balance_file);
              }
          }

        if (rewriting && error.empty())
          try
            {
              remove_obsolete_files(path, tag, n_ranks, !history.friction_pressures.stations.empty(),
                                    m.has_accepted_balance());
            }
          catch (const std::exception &exc)
            {
              error = exc.what();
            }
        const auto failed_rank = Utilities::MPI::min(error.empty() ? n_ranks : rank, comm);
        if (failed_rank < n_ranks)
          {
            error = Utilities::MPI::broadcast(comm, error, failed_rank);
            AssertThrow(false, ExcMessage("Growing fault history writer rank "+std::to_string(failed_rank)+": "+error));
          }
      }
    return {"Growing fault","native section and point history"};
  }
  ASPECT_REGISTER_POSTPROCESSOR(GrowingFaultHistory,"growing fault history",
                                "Write accepted strip records, native point history and the physical work ledger. "
                                "The native stress columns tauxx, tauyy, tauzz and tauxy are tension-positive "
                                "deviatoric components in Pa at the accepted endpoint.")

}
