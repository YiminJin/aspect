/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include <aspect/material_model/fixed_fault.h>
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
     * Export the same samples at full precision for restart and retry. */
    template <int dim>
    class FixedFaultProbe : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          const auto &section = Plugins::get_plugin_as_type<const MaterialModel::FixedFault<dim>>(this->get_material_model()).accepted_section();
          const unsigned int step = this->get_timestep_number();
          const auto comm = this->get_mpi_communicator();
          const unsigned int rank = Utilities::MPI::this_mpi_process(comm);
          const auto directory = this->get_output_directory();
          std::ofstream snapshot(directory+"fault-"+std::to_string(step)+"-"+std::to_string(rank)+".txt");
          snapshot << std::setprecision(17) << this->get_time() << ' ' << this->get_timestep() << ' '
                   << section.age << ' ' << section.slip << ' ' << section.rate << '\n';

          // The archived one-rank solver sampled QGauss(3) in every cell.
          // For the shortened trajectory the lifecycle checker compares
          // these same fields against the independent clean continuation.
          const bool reference = (step == 0 || (step <= 2 && this->get_timestep() == .0125 && this->get_time() == step*.0125));
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
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe.reinit(cell);
                fe.get_function_values(this->get_solution(),values);
                for (unsigned int q=0; q<rule.size(); ++q)
                  {
                    ++points;
                    snapshot << cell->id() << ' ' << q << ' ' << fe.quadrature_point(q) << ' ' << values[q] << '\n';
                    if (reference)
                      {
                        const auto &v = values[q];
                        check(v[intro.velocities[0]],0.);
                        check(v[intro.velocities[1]],shear_gradient[step]*fe.quadrature_point(q)[0]);
                        check(v[intro.pressure],stress_yy[step]);
                        check(v[intro.temperature],293.);
                        check(v[intro.compositional_fields[0]],-stress_yy[step]);
                        check(v[intro.compositional_fields[1]],stress_yy[step]);
                        check(v[intro.compositional_fields[2]],stress_xy[step]);
                        check(v[intro.compositional_fields[3]],.35);
                      }
                  }
              }
          const auto total = Utilities::MPI::sum(points,comm);
          if (rank == 0)
            std::ofstream(directory+"fault-checks",step==0 ? std::ios::out : std::ios::app) << step << ' ' << total << '\n';
          return {"Fixed fault probe", "sampled accepted fields"};
        }
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
