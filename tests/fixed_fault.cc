/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/simulator/assemblers/fault_stress.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/time_stepping/interface.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace Postprocess
  {
    /** Compare the whole native field with the homogeneous reduced donor run.
     * Export the same samples at full precision for clean-run and retry comparisons. */
    template <int dim>
    class FixedFaultProbe : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("Fixed fault probe");
          prm.declare_entry("Compare homogeneous reference","true",Patterns::Bool(),
                            "Check both equivalent labels and every bulk sample against the homogeneous donor.");
          prm.leave_subsection();
          prm.leave_subsection();
        }

        void parse_parameters(ParameterHandler &prm) override
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("Fixed fault probe");
          homogeneous = prm.get_bool("Compare homogeneous reference");
          prm.leave_subsection();
          prm.leave_subsection();
        }

        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          const auto &labels = Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model()).labels();
          const unsigned int step = this->get_timestep_number();
          const auto comm = this->get_mpi_communicator();
          const unsigned int rank = Utilities::MPI::this_mpi_process(comm);
          const auto directory = this->get_output_directory();
          std::ofstream snapshot(directory+"fault-"+std::to_string(step)+"-"+std::to_string(rank)+".txt");
          snapshot << std::setprecision(17) << this->get_time() << ' ' << this->get_timestep();
          for (const auto &label : labels)
            snapshot << ' ' << label.position << ' ' << label.age << ' ' << label.slip << ' ' << label.rate;
          snapshot << '\n';

          // The archived one-rank solver sampled QGauss(3) in every cell.
          // For the shortened trajectory the lifecycle checker compares
          // these same fields against the independent clean continuation.
          if (homogeneous)
            AssertThrow(labels.size() == 2,ExcMessage("The homogeneous regression requires its two original endpoints."));
          const bool reference = homogeneous && (step == 0 || (step <= 2 && this->get_timestep() == .0125 && this->get_time() == step*.0125));
          const double rate[] = {0.17074374817067778,0.16359216623205453,0.16724399270585846};
          const double age[] = {10.,7.7554483265394705,5.9829702628709551};
          const double shear_gradient[] = {-0.14972708428112763,0.098036219175796038,-0.063366308920228251};
          const double stress_xy[] = {0.,.0325,.045};
          const double stress_yy[] = {0.,0.,.00010496603196585};
          const auto check = [](const double actual, const double expected)
          {
            AssertThrow(std::abs(actual-expected) <= 1.e-10+1.e-8*std::abs(expected),
                        ExcMessage("Fixed-fault donor comparison failed: measured "+std::to_string(actual)
                                   +", expected "+std::to_string(expected)));
          };
          if (reference)
            for (const auto &section : labels)
              {
                check(section.rate,rate[step]);
                check(section.age,age[step]);
                check(section.slip,.0125*((step>=1 ? rate[1] : 0.)+(step>=2 ? rate[2] : 0.)));
              }

          const QGauss<dim> rule(3);
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_values | update_quadrature_points);
          std::vector<Vector<double>> values(rule.size(),Vector<double>(this->get_fe().n_components()));
          unsigned int points = 0;
          const auto &intro = this->introspection().component_indices;
          const auto &fields = this->introspection();
          const auto xx = fields.compositional_index_for_name("fault_stress_xx");
          const auto yy = fields.compositional_index_for_name("fault_stress_yy");
          const auto xy = fields.compositional_index_for_name("fault_stress_xy");
          const auto phase = fields.compositional_index_for_name("phase_field");
          if (step == 0)
            check_density(xx);
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                if (step == 0 && points == 0)
                  check_work(cell);
                fe.reinit(cell);
                fe.get_function_values(this->get_solution(),values);
                for (unsigned int q=0; q<rule.size(); ++q)
                  {
                    ++points;
                    snapshot << cell->id() << ' ' << q << ' ' << fe.quadrature_point(q);
                    const auto &v = values[q];
                    for (const unsigned int component :
                    {
                      intro.velocities[0],intro.velocities[1],intro.pressure,intro.temperature,
                                         intro.compositional_fields[xx],intro.compositional_fields[yy],intro.compositional_fields[xy],intro.compositional_fields[phase]
                      })
                    snapshot << ' ' << v[component];
                    snapshot << '\n';
                    if (reference)
                      {
                        check(v[intro.velocities[0]],0.);
                        check(v[intro.velocities[1]],shear_gradient[step]*fe.quadrature_point(q)[0]);
                        check(v[intro.pressure],stress_yy[step]);
                        check(v[intro.temperature],293.);
                        check(v[intro.compositional_fields[xx]],-stress_yy[step]);
                        check(v[intro.compositional_fields[yy]],stress_yy[step]);
                        check(v[intro.compositional_fields[xy]],stress_xy[step]);
                        check(v[intro.compositional_fields[phase]],.35);
                      }
                  }
              }
          const auto total = Utilities::MPI::sum(points,comm);
          if (rank == 0)
            std::ofstream(directory+"fault-checks",step==0 ? std::ios::out : std::ios::app) << step << ' ' << total << '\n';
          return {"Fixed fault probe", "sampled accepted fields"};
        }

      private:
        // Use the production assembler on a nonconstant prescribed stress. A
        // Q2 velocity w=(x^2+y, xy+y^2) is represented exactly by its nodes.
        // Compare r.w against the analytic gradient, including shear and sign.
        void check_work(const typename DoFHandler<dim>::active_cell_iterator &cell) const
        {
          const auto &intro = this->introspection();
          const auto &system = this->get_fe();
          const unsigned int n = intro.stokes_dof_info.size();
          const QGauss<dim> rule(4);
          Assemblers::FaultStressLoad<dim> assembler;
          assembler.initialize_simulator(this->get_simulator());
          for (const bool rebuild :
          {
            true,false
          })
          {
            internal::Assembly::Scratch::StokesSystem<dim> scratch(system,this->get_mapping(),rule,QGauss<dim-1>(3),
                                                                   update_gradients | update_quadrature_points | update_JxW_values,update_default,
                                                                   intro.n_compositional_fields,n,false,false,rebuild,false,false);
            internal::Assembly::CopyData::StokesSystem<dim> data(n,false);
            scratch.reinit(cell);
            data.local_rhs = 0.;
            assembler.create_additional_material_model_outputs(scratch.material_model_outputs);
            auto offset = scratch.material_model_outputs.template get_additional_output_object<MaterialModel::FaultStressOffset<dim>>();
            const auto &fe = scratch.finite_element_values;
            double expected = 0.;
            for (unsigned int q=0; q<rule.size(); ++q)
              {
                const auto &point = fe.quadrature_point(q);
                const double x = point[0], y = point[1];
                auto &stress = offset->stress[q];
                stress[0][0] = 1.+x*y;
                stress[1][1] = 2.-y;
                stress[0][1] = .5+x*y*y;
                expected -= fe.JxW(q)*(stress[0][0]*2.*x + stress[1][1]*(x+2.*y) + stress[0][1]*(1.+y));
              }
            assembler.execute(scratch,data);
            double work = 0.;
            for (unsigned int i=0; i<n; ++i)
              {
                const auto index = intro.stokes_dof_info[i].local_dof_index;
                const auto component = system.system_to_component_index(index).first;
                if (component == intro.component_indices.pressure)
                  {
                    AssertThrow(data.local_rhs[i] == 0.,ExcMessage("Stress load reached a pressure test."));
                    continue;
                  }
                const auto p = this->get_mapping().transform_unit_to_real_cell(cell,system.unit_support_point(index));
                work += data.local_rhs[i]*(component == intro.component_indices.velocities[0] ?
                                           p[0]*p[0]+p[1] : p[0]*p[1]+p[1]*p[1]);
              }
            AssertThrow(std::abs(expected) > 1.e-6 && std::abs(work-expected) < 1.e-12,
                        ExcMessage("Incorrect stress-offset virtual work."));
          }
        }

        void check_density(const unsigned int stress_field) const
        {
          const auto n = this->n_compositional_fields();
          MaterialModel::MaterialModelInputs<dim> in(2,n);
          MaterialModel::MaterialModelOutputs<dim> out(2,n);
          for (unsigned int q=0; q<2; ++q)
            {
              in.temperature[q] = 293.;
              in.pressure[q] = 0.;
              std::fill(in.composition[q].begin(),in.composition[q].end(),0.);
              in.composition[q][stress_field] = q == 0 ? 1. : 1.e6;
            }
          this->get_material_model().evaluate(in,out);
          AssertThrow(out.densities[0] == 3300. && out.densities[1] == out.densities[0],
                      ExcMessage("Positive fault stress changed the background density."));
        }

        bool homogeneous;
    };

    ASPECT_REGISTER_POSTPROCESSOR(FixedFaultProbe,"fixed fault probe","Read accepted fields for the fixed fault regression.")
  }
}


namespace aspect
{
  namespace TimeStepping
  {
    /** Discard one fully solved first interval and retry it at half length. */
    template <int dim>
    class RejectLongFirstStep : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        double execute() override
        {
          return std::numeric_limits<double>::max();
        }

        std::pair<Reaction,double> determine_reaction(const TimeStepInfo &) override
        {
          if (this->get_timestep_number() == 1 && this->get_timestep() > .02)
            return {Reaction::repeat_step,.5*this->get_timestep()};
          return {Reaction::advance,std::numeric_limits<double>::max()};
        }
    };

    ASPECT_REGISTER_TIME_STEPPING_MODEL(RejectLongFirstStep, "reject long first fault step",
                                        "Fixed fault regression fixture that requests a retry after Stokes convergence.")
  }
}
