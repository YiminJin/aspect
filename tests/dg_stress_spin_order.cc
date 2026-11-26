/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator/assemblers/interface.h>
#include <aspect/simulator/solver/dg_transport.h>
#include <aspect/simulator_signals.h>

#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace
  {
    constexpr double interval = .0125;

    // At initialization -div(2 kappa D(u)) = (0,-2 kappa) for u=(8,x^2).
    // Constant pressure balances normal traction and gives uniform section rates.
    template <int dim>
    class InitialShearForce : public Assemblers::Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        void execute(internal::Assembly::Scratch::ScratchBase<dim> &base,
                     internal::Assembly::CopyData::CopyDataBase<dim> &copy) const override
        {
          if (this->get_timestep_number()!=0)
            return;
          const auto &scratch = dynamic_cast<const internal::Assembly::Scratch::StokesSystem<dim>&>(base);
          auto &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&>(copy);
          const auto &fe = scratch.finite_element_values;
          const auto &intro = this->introspection();
          const double force = 20.*std::expm1(-.1);
          for (unsigned int i=0; i<data.local_rhs.size(); ++i)
            {
              const unsigned int index = intro.stokes_dof_info[i].local_dof_index;
              for (unsigned int q=0; q<fe.n_quadrature_points; ++q)
                data.local_rhs(i) += force*fe[intro.extractors.velocities].value(index,q)[1]*fe.JxW(q);
            }
        }
    };

    // Exact exterior of the rotated-then-translated negative control.
    // At time zero this also supplies the rotated initial DG interpolant.
    class RotatedParent : public Function<2>
    {
      public:
        explicit RotatedParent(const double rate) : Function<2>(3), rate(rate) {}
        double value(const Point<2> &p,const unsigned int c) const override
        {
          const double departure = p[0]-8.*get_time();
          const double twice_angle = interval*(2.*departure+rate);
          const double shear = .3*std::sin(twice_angle)+.05*std::cos(twice_angle);
          const double diagonal = .3*std::cos(twice_angle)-.05*std::sin(twice_angle);
          if (c==2)
            return shear;
          return c==0 ? .1+diagonal : .1-diagonal;
        }
      private:
        const double rate;
    };

    // Advance a separately rotated parent on the actual mesh, including its
    // consistent time-dependent inflow. Return xy at the receiving cell center.
    double receiver_shear_after_upstream_rotation(const SimulatorAccess<2> &sim, const double rate)
    {
      RotatedParent exterior(rate);
      const auto &intro = sim.introspection();
      const auto &sets = intro.index_sets;
      const auto comm = sim.get_mpi_communicator();
      const std::array<unsigned int,3> fields = {{
          intro.compositional_index_for_name("fault_stress_xx"),
          intro.compositional_index_for_name("fault_stress_yy"),intro.compositional_index_for_name("fault_stress_xy")
        }
      };
      const auto &system = sim.get_fe();
      const auto &dg = system.base_element(intro.base_elements.compositional_fields[fields[0]]);
      LinearAlgebra::BlockVector owned,ghost;
      owned.reinit(sets.system_partitioning,comm);
      owned = sim.get_solution();
      std::vector<types::global_dof_index> indices(system.n_dofs_per_cell());
      for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(indices);
            for (unsigned int i=0; i<dg.n_dofs_per_cell(); ++i)
              {
                const auto x=sim.get_mapping().transform_unit_to_real_cell(cell,dg.unit_support_point(i));
                for (unsigned int c=0; c<3; ++c)
                  owned[indices[system.component_to_system_index(intro.component_indices.compositional_fields[fields[c]],i)]]
                    = exterior.value(x,c);
              }
          }
      owned.compress(VectorOperation::insert);
      ghost.reinit(sets.system_partitioning,sets.system_relevant_partitioning,comm);
      ghost=owned;
      const auto transported=Transport::advect_stress(sim,fields,ghost,exterior,0.,interval);
      double center_xy=0.;
      for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned() && cell->center()[0]>.5)
          for (unsigned int i=0; i<dg.n_dofs_per_cell(); ++i)
            center_xy += dg.shape_value(i,Point<2>(.5,.5))*transported.at(cell->id())[i][0][1];
      return Utilities::MPI::sum(center_xy,comm);
    }
  }

  namespace Postprocess
  {
    template <int dim>
    class SpinOrderProbe : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        void initialize() override
        {
          this->get_signals().set_assemblers.connect([](const SimulatorAccess<dim> &,Assemblers::Manager<dim> &assemblers)
          {
            assemblers.stokes_system.push_back(std::make_unique<InitialShearForce<dim>>());
          });
        }

        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          const auto &material=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
          const auto &labels=material.labels();
          const auto &intro=this->introspection();
          const auto velocity=intro.extractors.velocities;
          const auto comm=this->get_mpi_communicator();
          if (this->get_timestep_number()==0)
            {
              AssertThrow(labels.size()==2,ExcMessage("Expected two section labels."));
              rate=labels.front().rate;
              AssertThrow(rate>0. && std::abs(labels.back().rate-rate)<1.e-10,
                          ExcMessage("Expected uniform initial section rate."));
              FEValues<dim> fe(this->get_mapping(),this->get_fe(),QGauss<dim>(3),update_values|update_quadrature_points);
              std::vector<Tensor<1,dim>> values(fe.n_quadrature_points);
              double error=0.;
              for (const auto &cell : this->get_dof_handler().active_cell_iterators())
                if (cell->is_locally_owned())
                  {
                    fe.reinit(cell);
                    fe[velocity].get_function_values(this->get_solution(),values);
                    for (unsigned int q=0; q<values.size(); ++q)
                      error=std::max({error,std::abs(values[q][0]-8.),std::abs(values[q][1]-std::pow(fe.quadrature_point(q)[0],2))});
                  }
              AssertThrow(Utilities::MPI::max(error,comm)<1.e-10,ExcMessage("Incorrect accepted quadratic shear velocity."));
              if constexpr (dim==2)
                upstream_rotated_shear = receiver_shear_after_upstream_rotation(*this,rate);
            }
          else
            {
              AssertThrow(this->get_time()==interval && this->get_timestep()==interval,ExcMessage("Unexpected spin-test clock."));
              FEValues<dim> fe(this->get_mapping(),this->get_fe(),QMidpoint<dim>(),update_values|update_gradients);
              std::vector<double> xy(1);
              std::vector<Tensor<2,dim>> gradient(1);
              double observed=0.;
              for (const auto &cell : this->get_dof_handler().active_cell_iterators())
                if (cell->is_locally_owned() && cell->center()[0]>.5)
                  {
                    fe.reinit(cell);
                    const auto xy_field = intro.compositional_index_for_name("fault_stress_xy");
                    fe[intro.extractors.compositional_fields[xy_field]].get_function_values(this->get_solution(),xy);
                    fe[velocity].get_function_gradients(this->get_solution(),gradient);
                    // Remove the exactly DGQ2-representable endpoint strain and
                    // Q1 slip terms, and undo Maxwell relaxation, in Pa.
                    // Here t=e_y and n=-e_x, so subtracting slip adds V to 2D_xy.
                    const double slip_rate = .5*(labels.front().rate+labels.back().rate);
                    const double kappa=-10.*std::expm1(-interval);
                    observed=std::exp(interval)*(xy[0]-kappa*(gradient[0][0][1]+gradient[0][1][0]+slip_rate));
                  }
              observed=Utilities::MPI::sum(observed,comm);
              RotatedParent analytic(rate);
              // Independent P2 projection evaluated at the center of [0.5,1]:
              // its evaluation kernel in [-1,1] is 1 - (5/2) P2.
              const QGauss<1> rule(5);
              double expected=0.;
              for (unsigned int q=0; q<rule.size(); ++q)
                {
                  const double s=2.*rule.point(q)[0]-1.;
                  expected+=rule.weight(q)*(1.-1.25*(3.*s*s-1.))*analytic.value(Point<2>(.75+.25*s,.5),2);
                }
              analytic.set_time(interval);
              if (Utilities::MPI::this_mpi_process(comm)==0)
                std::ofstream(this->get_output_directory()+"spin-values") << std::setprecision(17)
                                                                          << observed << ' ' << expected << ' ' << upstream_rotated_shear << ' ' << rate << '\n';
              AssertThrow(std::abs(upstream_rotated_shear-analytic.value(Point<2>(.75,.5),2))<1.e-7,
                          ExcMessage("Reversed control missed the upstream shear."));
              AssertThrow(std::abs(upstream_rotated_shear-expected)>7.e-4 && std::abs(observed-expected)<5.e-10,
                          ExcMessage("The accepted shear requires rotation at the receiving point after transport."));
              if (Utilities::MPI::this_mpi_process(comm)==0)
                std::ofstream(this->get_output_directory()+"spin-check") << "Transported parent rotated at the receiving cell.\n";
            }
          return {"DG spin order","checked"};
        }
      private:
        double rate = 0.;
        double upstream_rotated_shear = 0.;
    };
    ASPECT_REGISTER_POSTPROCESSOR(SpinOrderProbe,"DG spin order check","Observe the order of bulk stress advection and rotation.")
  }
}
