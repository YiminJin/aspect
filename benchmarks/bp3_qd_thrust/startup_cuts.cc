/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/geometry_model/box.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/postprocess/interface.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>

namespace aspect::Postprocess
{
  /** Probe the actual FE phase, velocity and section return across each contact
   * and the interior. Included only in bounded verification inputs. */
  template <int dim>
  class StartupCuts : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      std::pair<std::string,std::string> execute(TableHandler &) override
      {
        if (this->get_timestep_number()!=0) return {};
        const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
        const auto &geometry=Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());
        const double top=geometry.get_extents()[1], sine=std::sqrt(.75);
        const unsigned int steps=512;
        const double radius=model.labels()[0].half_width;
        std::vector<double> rates;
        for (const auto &label:model.labels()) rates.push_back(label.rate);
        const auto phase_field=this->introspection().extractors.compositional_fields[
                                 this->introspection().compositional_index_for_name("phase_field")];
        std::vector<double> records(3*(steps+1)*6);
        for (const auto &cell:this->get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const auto lo=cell->vertex(0), hi=cell->vertex(GeometryInfo<dim>::vertices_per_cell-1);
              for (unsigned int cut=0; cut<3; ++cut)
                {
                  const double y=.5*cut*top;
                  if (y<lo[1] || y>hi[1] || (y==hi[1] && y!=top)) continue;
                  std::vector<Point<dim>> unit,physical;
                  std::vector<unsigned int> indices;
                  for (unsigned int i=0; i<=steps; ++i)
                    {
                      const double r=radius*(2.*i/steps-1.);
                      Point<dim> point;
                      point[0]=(r-.5*(y-top))/sine;
                      point[1]=y;
                      if (point[0]<lo[0] || point[0]>=hi[0]) continue;
                      unit.push_back(this->get_mapping().transform_real_to_unit_cell(cell,point));
                      physical.push_back(point);
                      indices.push_back(6*(cut*(steps+1)+i));
                    }
                  if (unit.empty()) continue;
                  FEValues<dim> fe(this->get_mapping(),this->get_fe(),Quadrature<dim>(unit),update_values);
                  fe.reinit(cell);
                  std::vector<double> phase(unit.size());
                  std::vector<Tensor<1,dim>> velocity(unit.size());
                  fe[phase_field].get_function_values(this->get_solution(),phase);
                  fe[this->introspection().extractors.velocities].get_function_values(this->get_solution(),velocity);
                  for (unsigned int q=0; q<unit.size(); ++q)
                    {
                      const auto sample=model.phase_profile().sample(physical[q],phase[q]);
                      const auto k=indices[q];
                      records[k]=1.;
                      records[k+1]=phase[q];
                      records[k+2]=sample.chi;
                      records[k+3]=sample.chi*sample.rate(rates);
                      records[k+4]=velocity[q][0];
                      records[k+5]=velocity[q][1];
                    }
                }
            }
        Utilities::MPI::sum(records,this->get_mpi_communicator(),records);
        bool ok=true;
        if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==0)
          {
            std::ofstream out(this->get_output_directory()+"startup-cuts.txt");
            out << std::setprecision(17) << "y_m r_m phase chi_per_m returned_rate_per_s ux_m_s uy_m_s\n";
            for (unsigned int cut=0; cut<3; ++cut)
              for (unsigned int i=0; i<=steps; ++i)
                {
                  const auto k=6*(cut*(steps+1)+i);
                  ok &= records[k]==1.;
                  out << .5*cut *top << ' ' << radius*(2.*i/steps-1.);
                  for (unsigned int j=1; j<6; ++j) out << ' ' << records[k+j];
                  out << '\n';
                }
            out.close();
            ok &= out.good();
          }
        AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(ok),this->get_mpi_communicator()),
                    ExcMessage("BP3 cut missing a unique cell trace or failed to write."));
        return {"BP3 startup cuts","startup-cuts.txt"};
      }
  };

  ASPECT_REGISTER_POSTPROCESSOR(StartupCuts,"BP3 startup cuts","Physical phase, velocities and normalized slip transfer across horizontal cuts.")
}
