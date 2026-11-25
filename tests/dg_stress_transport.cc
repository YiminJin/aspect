/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_signals.h>
#include <aspect/simulator/assemblers/fault_stress.h>
#include <aspect/simulator/solver/dg_transport.h>

#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace
  {
    constexpr double shear_rate = .4;

    // Unit-width vertical section in translation and simple shear. The free normal
    // traction on the horizontal walls sets p=tau_yy. The Q1 balance is
    // uniform, with t=e_y, n=-e_x and chi=1.
    double section_rate(const double xx_minus_yy, const double xy, const double dt)
    {
      const double beta = std::exp(-dt);
      double lo = 0., hi = 2.;
      for (unsigned int k=0; k<60; ++k)
        {
          const double rate = .5*(lo+hi);
          const double strength = (5.-beta*xx_minus_yy)*.025*
                                  std::asinh(rate/.002*std::exp((.6+.015*std::log(10.*.001/.008))/.025));
          if (3.85-beta*xy-strength-10.*(1.-beta)*shear_rate > (1.+10.*(1.-beta))*rate)
            lo = rate;
          else
            hi = rate;
        }
      return .5*(lo+hi);
    }

    // Incoming stress after the first physical translation and accepted-spin
    // rotation, before multiplication by beta. Both diagonal slopes are 0.3.
    SymmetricTensor<2,2> first_parent(const double x, const double dt)
    {
      const double angle = .5*dt*(shear_rate+section_rate(.6,.05,.1));
      const double mean = .1+.3*(x-8.*dt);
      const double half_difference = .3*std::cos(2.*angle)-.05*std::sin(2.*angle);
      SymmetricTensor<2,2> result;
      result[0][0] = mean+half_difference;
      result[1][1] = mean-half_difference;
      result[0][1] = .3*std::sin(2.*angle)+.05*std::cos(2.*angle);
      return result;
    }

    template <int dim>
    class MeasuredTransportLoad : public Assemblers::FaultStressLoad<dim>
    {
      public:
        void execute(internal::Assembly::Scratch::ScratchBase<dim> &scratch_base,
                     internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const override
        {
          auto &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>&>(scratch_base);
          auto &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&>(data_base);
          if (this->get_timestep_number()!=1 || scratch.cell->center()[0]<.5)
            return Assemblers::FaultStressLoad<dim>::execute(scratch,data);

          // Contract the very local RHS passed to ASPECT's scatter with w=(x,0).
          // Vertical fault slip contributes only to xy and drops out of this action.
          internal::Assembly::CopyData::StokesSystem<dim> isolated(data.local_rhs.size(),false);
          Assemblers::FaultStressLoad<dim>::execute(scratch,isolated);
          data.local_rhs += isolated.local_rhs;
          double work = 0.;
          for (unsigned int i=0; i<data.local_rhs.size(); ++i)
            {
              const auto index = this->introspection().stokes_dof_info[i].local_dof_index;
              if (this->get_fe().system_to_component_index(index).first == this->introspection().component_indices.velocities[0])
                work += isolated.local_rhs[i]*this->get_mapping().transform_unit_to_real_cell(scratch.cell,this->get_fe().unit_support_point(index))[0];
            }
          const double dt = this->get_timestep();
          const double exact = -.5*std::exp(-dt)*first_parent(.75,dt)[0][0];
          std::ofstream out(this->get_output_directory()+"load-"+std::to_string(dt));
          out << std::setprecision(17) << work << ' ' << exact << ' ' << exact-.5*std::exp(-dt)*.3*8.*dt << '\n';
          AssertThrow(std::abs(work-exact)<2.e-11,ExcMessage("Actual Stokes stress load missed the transported receiving-cell parent."));
        }
    };

    class Exterior : public Function<2>
    {
      public:
        Exterior(const double speed,const bool jump,const bool expanding)
          : Function<2>(3), speed(speed), jump(jump), expanding(expanding) {}
        double value(const Point<2> &p,const unsigned int c) const override
        {
          if (expanding)
            return c+1.;
          if (jump)
            return (c+1.)*(1.+p[1]+p[1]*p[1]);
          return (.3+c)*(p[0]-speed*this->get_time())+(.7-.2*c)*p[1]-.4*c;
        }
      private:
        const double speed;
        const bool jump,expanding;
    };

    // Exercise the same collective operator using exact nodal input, separately
    // from Maxwell mechanics. In the jump run the sole interior face crosses
    // ranks when the two cells are partitioned. Export every owned DG node.
    void spatial_check(const SimulatorAccess<2> &sim)
    {
      const auto &system=sim.get_fe();
      const auto &intro=sim.introspection();
      const auto &sets=intro.index_sets;
      const std::array<unsigned int,3> fields= {{
          intro.compositional_index_for_name("fault_stress_xx"),
          intro.compositional_index_for_name("fault_stress_yy"),intro.compositional_index_for_name("fault_stress_xy")
        }
      };
      const auto &dg=system.base_element(intro.base_elements.compositional_fields[fields[0]]);
      const auto comm=sim.get_mpi_communicator();
      const unsigned int rank=Utilities::MPI::this_mpi_process(comm);
      std::ofstream output(sim.get_output_directory()+"spatial-"+std::to_string(rank));
      output << std::setprecision(17);
      for (unsigned int test=0; test<5; ++test)
        {
          const bool jump=test==3, expanding=test==4;
          const double speed=test==1 ? -8. : (test==2 ? 0. : 8.);
          Exterior exterior(speed,jump,expanding);
          LinearAlgebra::BlockVector owned, ghost;
          owned.reinit(sets.system_partitioning,comm);
          ghost.reinit(sets.system_partitioning,sets.system_relevant_partitioning,comm);
          std::vector<types::global_dof_index> indices(system.n_dofs_per_cell());
          // Set DG values on the appropriate one-sided trace at the face.
          for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                cell->get_dof_indices(indices);
                for (unsigned int i=0; i<indices.size(); ++i)
                  {
                    if (!sim.get_dof_handler().locally_owned_dofs().is_element(indices[i]))
                      continue;
                    const auto comp=system.system_to_component_index(i).first;
                    if (comp==intro.component_indices.velocities[0])
                      {
                        const auto x=sim.get_mapping().transform_unit_to_real_cell(cell,system.unit_support_point(i));
                        owned[indices[i]]=expanding ? x[0] : speed;
                      }
                    for (unsigned int c=0; c<3; ++c)
                      if (comp==intro.component_indices.compositional_fields[fields[c]])
                        {
                          const auto x=sim.get_mapping().transform_unit_to_real_cell(cell,system.unit_support_point(i));
                          owned[indices[i]]=jump && cell->center()[0]>.5 ? 0. : exterior.value(x,c);
                        }
                  }
              }
          owned.compress(VectorOperation::insert);
          ghost=owned;
          const std::vector<double> durations=jump ? std::vector<double> {.001,.002,.003,.024} :
                                              std::vector<double> {.025};
          for (const double dt : durations)
            {
              const auto result=Transport::advect_stress(sim,fields,ghost,exterior,0.,dt);
              exterior.set_time(dt);
              for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
                if (cell->is_locally_owned())
                  for (unsigned int i=0; i<dg.n_dofs_per_cell(); ++i)
                    {
                      const auto x=sim.get_mapping().transform_unit_to_real_cell(cell,dg.unit_support_point(i));
                      for (unsigned int c=0; c<3; ++c)
                        {
                          const double actual=result.at(cell->id())[i].access_raw_entry(c);
                          output << test << ' ' << dt << ' ' << cell->id() << ' ' << i << ' ' << c << ' ' << x << ' ' << actual << '\n';
                          if (!jump)
                            AssertThrow(std::abs(actual-exterior.value(x,c))<2.e-12,
                                        ExcMessage("Affine translation or material divergence check failed."));
                        }
                    }
            }
        }
    }

  }

  namespace Postprocess
  {
    template <int dim>
    class TransportProbe : public Postprocess::Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        void initialize() override
        {
          this->get_signals().set_assemblers.connect([](const SimulatorAccess<dim> &,Assemblers::Manager<dim> &assemblers)
          {
            for (auto &a : assemblers.stokes_system)
              if (dynamic_cast<Assemblers::FaultStressLoad<dim>*>(a.get()))
                {
                  a=std::make_unique<MeasuredTransportLoad<dim>>();
                  return;
                }
            AssertThrow(false,ExcMessage("Expected the production fault stress load."));
          });
          const std::string publications=this->get_output_directory()+"published-"
                                         +std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()));
          std::ofstream(publications,std::ios::trunc).close();
          this->get_signals().post_accept_timestep.connect([this,publications](LinearAlgebra::BlockVector &)
          {
            std::ofstream log(publications,std::ios::app);
            log << std::setprecision(17) << this->get_time() << ' ' << this->get_timestep() << '\n';
          });
        }

        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          const auto step=this->get_timestep_number();
          if constexpr (dim==2)
            if (step==0)
              spatial_check(*this);
          const auto &material=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
          const auto &intro=this->introspection();
          const auto comm=this->get_mpi_communicator();
          const auto rank=Utilities::MPI::this_mpi_process(comm);
          const auto dir=this->get_output_directory();
          std::ofstream snapshot(dir+"accepted-"+std::to_string(step)+"-"+std::to_string(rank));
          snapshot << std::setprecision(17) << this->get_time() << ' ' << this->get_timestep() << '\n';
          const auto &labels=material.labels();
          for (unsigned int i=0; i<labels.size(); ++i)
            snapshot << labels[i].position << ' ' << material.label_ages().at(i) << ' ' << labels[i].slip << ' ' << labels[i].rate << '\n';
          const double dt=this->get_timestep();
          SymmetricTensor<2,2> endpoint_at_origin;
          if (step==1)
            {
              const auto parent=first_parent(0.,dt);
              const double rate=section_rate(parent[0][0]-parent[1][1],parent[0][1],dt);
              endpoint_at_origin = std::exp(-dt)*parent;
              endpoint_at_origin[0][1] += 10.*(1.-std::exp(-dt))*(shear_rate+rate);
              const double decay=std::exp(-rate*dt/.008);
              for (unsigned int i=0; i<labels.size(); ++i)
                {
                  const auto &label=labels[i];
                  AssertThrow(std::abs(label.rate-rate)<1.e-10,ExcMessage("Incorrect solved section rate."));
                  AssertThrow(std::abs(label.slip-dt*rate)<1.e-11 &&
                              std::abs(material.label_ages().at(i)-(10.*decay+.008/rate*(1.-decay)))<2.e-10,
                              ExcMessage("Incorrect first accepted label history."));
                }
            }
          const auto &system=this->get_fe();
          std::vector<types::global_dof_index> indices(system.n_dofs_per_cell());
          double error=0.;
          unsigned int cells=0;
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                ++cells;
                cell->get_dof_indices(indices);
                for (unsigned int c=0; c<3; ++c)
                  for (unsigned int i=0; i<9; ++i)
                    {
                      const unsigned int index=system.component_to_system_index(intro.component_indices.compositional_fields[c],i);
                      const auto x=this->get_mapping().transform_unit_to_real_cell(cell,system.unit_support_point(index));
                      const double actual=this->get_solution()[indices[index]];
                      snapshot << cell->id() << ' ' << c << ' ' << i << ' ' << actual << '\n';
                      if (step==1)
                        {
                          const double expected=endpoint_at_origin.access_raw_entry(c)+(c<2 ? .3*x[0]*std::exp(-dt) : 0.);
                          error=std::max(error,std::abs(actual-expected));
                          AssertThrow(std::abs(actual-expected)<2.e-10,
                                      ExcMessage("Accepted DG stress differs from translated, rotated and reacted parent."));
                        }
                    }
              }
          const unsigned int count=Utilities::MPI::sum(cells,comm);
          error=Utilities::MPI::max(error,comm);
          if (rank==0)
            {
              std::ofstream(dir+"transport-checks",step==0 ? std::ios::out : std::ios::app) << step << ' ' << count << '\n';
              if (step==1)
                {
                  AssertThrow(std::ifstream(dir+"load-"+std::to_string(this->get_timestep())).good(),ExcMessage("Missing measured Stokes action."));
                  std::ofstream(dir+"endpoint-error") << std::setprecision(17) << error << '\n';
                }
            }
          return {"DG stress transport","checked"};
        }
    };

    ASPECT_REGISTER_POSTPROCESSOR(TransportProbe,"DG stress transport check","Check transported stress through ordinary assembly and publication.")
  }
}
