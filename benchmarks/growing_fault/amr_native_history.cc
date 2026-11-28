/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/phase_field_fault/strip_material_history.h>
#include <aspect/phase_field/midpoint_fracture.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_signals.h>

#include <deal.II/fe/fe_values.h>
#include <fstream>

namespace aspect::Postprocess
{
  /** Observe exactly the same accepted endpoint on the two sides of an
   * actual refinement, including the old FE vector on a rejected attempt.
   * Register after the initial postprocessing so post-refinement sampling
   * follows the material callback that unpacks the accepted native records. */
  template <int dim>
  class RemeshHistory : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      void initialize() override
      {
        if constexpr (dim == 2)
          if (this->get_parameters().resume_computation)
            this->get_signals().pre_nonlinear_solver.connect([this](const SimulatorAccess<dim> &)
            {
              if (!read_restored_parent)
                {
                  snapshot("restored");
                  read_restored_parent = true;
                }
            });
      }

      std::pair<std::string,std::string> execute(TableHandler &) override
      {
        if constexpr (dim == 2)
          if (!connected)
            {
              this->get_signals().pre_refinement_store_user_data.connect([this](auto &)
              {
                ++pass;
                snapshot("before");
              });
              this->get_signals().post_refinement_load_user_data.connect([this](auto &)
              {
                snapshot("after");
              });
              connected = true;
            }
        return {"",""};
      }

    private:
      bool connected = false, read_restored_parent = false;
      unsigned int pass = 0;

      void snapshot(const std::string &when) const
      {
        if constexpr (dim == 2)
          {
            const auto &material = dynamic_cast<const MaterialModel::PhaseFieldFault<dim> &>
                                   (this->get_material_model()).strip_history();
            const auto &history = material.history();
            const auto comm = this->get_mpi_communicator();
            const auto rank = Utilities::MPI::this_mpi_process(comm);
            const std::string tag = this->get_output_directory()+"remesh-"+std::to_string(history.step)
                                    +"-"+std::to_string(pass)+"-"+when+"-r"+std::to_string(rank);
            const auto &fe = this->get_fe();
            const auto &intro = this->introspection();
            const auto field = material.phase_field();
            const auto c = FEValuesExtractors::Scalar(intro.component_indices.compositional_fields[field]);
            const auto &endpoint = when == "restored" ? this->get_old_solution() : this->get_solution();
            const auto &parent = when == "restored" ? this->get_old_old_solution() : this->get_old_solution();
            std::ofstream points(tag+".csv");
            points << std::setprecision(17) << "cell,level,x0,y0,hx,hy,strip,x,y,area,phase,old_phase,"
                   << "xx,yy,zz,xy,Axx,Ayy,Azz,Axy,signed,absolute\n";
            double bulk = 0., cohesive = 0.;
            for (const auto &cell : this->get_dof_handler().active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  const auto &rule = material.sections().cells.at(cell->id());
                  FEValues<2> values(this->get_mapping(),fe,rule.quadrature,update_values|update_quadrature_points|update_JxW_values);
                  values.reinit(cell);
                  std::vector<double> phi(rule.quadrature.size()), phi0(phi.size());
                  values[c].get_function_values(endpoint,phi);
                  values[c].get_function_values(parent,phi0);
                  for (unsigned int q=0; q<phi.size(); ++q)
                    {
                      const auto &point = history.cells.at(cell->id())[q];
                      const auto pos = values.quadrature_point(q);
                      const double area = values.JxW(q);
                      bulk += area*(point.stress*point.stress)/(4.*material.maxwell().G);
                      cohesive += area*material.maxwell().cohesive_energy(point.cohesive_strain,phi[q]);
                      points << cell->id() << ',' << cell->level() << ',' << cell->vertex(0)[0] << ',' << cell->vertex(0)[1]
                             << ',' << cell->extent_in_direction(0) << ',' << cell->extent_in_direction(1)
                             << ',' << rule.section[q] << ',' << pos[0] << ',' << pos[1] << ',' << area << ',' << phi[q] << ',' << phi0[q];
                      for (const auto *tensor : {&point.stress,&point.cohesive_strain})
                        points << ',' << (*tensor)[0][0] << ',' << (*tensor)[1][1] << ',' << (*tensor)[2][2] << ',' << (*tensor)[0][1];
                      points << ',' << point.signed_shear << ',' << point.absolute_shear << '\n';
                    }
                }
            const auto &model = this->get_parameters().surface_energy_model;
            const double Kf = model.fracture_energy/(model.geometry.normalization()*model.length_scale);
            const auto &phase = endpoint.block(intro.block_indices.compositional_fields[field]);
            const PhaseField::FractureRows rows(*this,field);
            double fracture = 0.;
            for (const auto &[i,mass] : rows.mass)
              fracture += Kf*mass*phase[i];
            for (const auto &[ij,K] : rows.stiffness)
              fracture += .5*Kf*phase[ij.first]*K*phase[ij.second];
            std::ofstream summary(tag+".txt");
            summary << std::setprecision(17) << "energy " << bulk << ' ' << cohesive << ' ' << fracture << '\n';
            summary << "operator " << model.length_scale << ' ' << Kf << ' ' << this->get_parameters().phase_discontinuous_penalty << '\n';
            summary << "clock " << history.step << ' ' << history.time << '\n';
            for (const double edge : material.sections().edges)
              summary << "edge " << edge << '\n';
            for (const auto &strip : history.sections)
              summary << "strip " << strip.formed << ' ' << strip.angle << ' ' << strip.slip << ' ' << strip.rate << ' ' << strip.resistance << '\n';
            for (double value : material.friction().state())
              summary << "state " << value << '\n';
            for (unsigned int k=0; k<history.friction_pressures.stations.size(); ++k)
              summary << "pressure " << history.friction_pressures.stations[k] << ' ' << history.friction_pressures.values[k] << '\n';
            bool ledger_written = true;
            if (material.has_accepted_balance())
              {
                std::ostringstream balance;
                {
                  aspect::oarchive ar(balance);
                  ar << material.balance();
                }
                std::ofstream ledger(tag+".ledger",std::ios::binary);
                ledger << balance.str();
                ledger.close();
                ledger_written = !ledger.fail();
              }
            summary.close();
            points.close();
            AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(summary && points && ledger_written),comm),
                        ExcMessage("Could not write the growing remesh observation."));
          }
      }
  };

  ASPECT_REGISTER_POSTPROCESSOR(RemeshHistory,"growing remesh history",
                                "Sample accepted states on both sides of mesh transfer for a zero-duration comparison.")
}
