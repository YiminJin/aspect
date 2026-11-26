/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator/assemblers/fault_stress.h>
#include <aspect/simulator_signals.h>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace MaterialModel
  {
    namespace
    {
      // A Q2-representable virtual velocity in m/s with engineering shear
      // rate 1/s in the first segment's geometric frame. For a bend this
      // single affine test resolves a different shear on the next segment.
      template <int dim>
      Tensor<1,dim> virtual_velocity(const MaterialFaultTrace<dim> &trace, const Point<dim> &x)
      {
        const auto &frame = trace.segment(0);
        return frame.tangent * (frame.normal * x);
      }

      template <int dim>
      class MeasuredLoad : public Assemblers::FaultStressLoad<dim>
      {
        public:
          void execute(internal::Assembly::Scratch::ScratchBase<dim> &scratch_base,
                       internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const override
          {
            const auto &scratch = dynamic_cast<const internal::Assembly::Scratch::StokesSystem<dim>&>(scratch_base);
            auto &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&>(data_base);
            if (this->get_timestep_number() != 0 || this->get_nonlinear_iteration() != 0)
              {
                Assemblers::FaultStressLoad<dim>::execute(scratch_base,data_base);
                return;
              }

            const auto &material = Plugins::get_plugin_as_type<const PhaseFieldFault<dim>>(this->get_material_model());
            const auto &trace = material.phase_profile().geometry;
            // Isolate just this assembler's RHS, before scatter or constraints.
            internal::Assembly::CopyData::StokesSystem<dim> isolated(data.local_dof_indices.size(),false);
            Assemblers::FaultStressLoad<dim>::execute(scratch_base,isolated);
            data.local_rhs += isolated.local_rhs;

            const auto &intro = this->introspection();
            double work = 0.;
            for (unsigned int i=0; i<isolated.local_rhs.size(); ++i)
              {
                const auto index = intro.stokes_dof_info[i].local_dof_index;
                const auto component = this->get_fe().system_to_component_index(index).first;
                for (unsigned int d=0; d<dim; ++d)
                  if (component == intro.component_indices.velocities[d])
                    {
                      const auto x = this->get_mapping().transform_unit_to_real_cell(scratch.cell,this->get_fe().unit_support_point(index));
                      work += isolated.local_rhs(i)*virtual_velocity(trace,x)[d];
                    }
              }
            const auto rank = Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
            std::ofstream file(this->get_output_directory()+"stokes-r"+std::to_string(rank)+".csv", std::ios::app);
            file << std::setprecision(17) << scratch.cell->id() << ',' << work << ','
                 << scratch.material_model_outputs.viscosities.front() << '\n';
            file.close();
            AssertThrow(file,ExcIO());
          }
      };
    }

    template <int dim>
    class TransferWork : public PhaseFieldFault<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Material model");
          prm.enter_subsection("Transfer work probe");
          prm.declare_entry("Middle rate increment", "0", Patterns::Double(),
                            "Middle-vertex perturbation at initialization, in m/s.");
          prm.leave_subsection();
          prm.leave_subsection();
        }

        void parse_parameters(ParameterHandler &prm) override
        {
          PhaseFieldFault<dim>::parse_parameters(prm);
          AssertThrow(!this->uses_growing_strips(), ExcMessage("This benchmark requires mature trace mechanics."));
          prm.enter_subsection("Material model");
          prm.enter_subsection("Transfer work probe");
          increment = prm.get_double("Middle rate increment");
          prm.leave_subsection();
          prm.leave_subsection();
        }

        void initialize() override
        {
          this->coupling->initialize_simulator(this->get_simulator());
          this->coupling->initialize();
          // Prescribe these three trial rates during initialization. As in the
          // ridge diagnostic, the free RSF root and convergence veto stay off.
          this->get_signals().set_assemblers.connect([](const SimulatorAccess<dim> &, Assemblers::Manager<dim> &manager)
          {
            for (auto &assembler : manager.stokes_system)
              if (dynamic_cast<Assemblers::FaultStressLoad<dim> *>(assembler.get()))
                {
                  assembler = std::make_unique<MeasuredLoad<dim>>();
                  return;
                }
            AssertThrow(false,ExcMessage("Missing fault stress load."));
          });
        }

        void update() override
        {
          AssertThrow(this->get_timestep_number() == 0, ExcMessage("The work experiment runs only at initialization."));
          // The diagnostic accessor borrows the live trace of this mutable
          // material. Seed trial_rates through prepare_parent(), then restore
          // the accepted rates before either action. At zero physical duration
          // no spin, displacement, age or stress depends on the perturbation.
          auto &trace = const_cast<MaterialFaultTrace<dim>&>(this->phase_profile().geometry);
          AssertThrow(trace.labels().size() == 3, ExcMessage("The transfer test requires three ordered labels."));
          std::vector<double> rates;
          for (const auto &label : trace.labels())
            rates.push_back(label.rate);
          auto perturbed_rates = rates;
          perturbed_rates[1] += increment;
          AssertThrow(perturbed_rates[1] > 0., ExcMessage("The rate perturbation must preserve positive speed."));
          trace.accept_section_rates(perturbed_rates,0.);
          PhaseFieldFault<dim>::update();
          trace.accept_section_rates(rates,0.);
          const auto &intro = this->introspection();
          const auto comm = this->get_mpi_communicator();
          LinearAlgebra::BlockVector owned(intro.index_sets.system_partitioning,comm);
          owned = this->get_old_solution();
          const auto block = intro.block_indices.velocities;
          owned.block(block) = 0.;
          LinearAlgebra::BlockVector sample(intro.index_sets.system_partitioning,intro.index_sets.system_relevant_partitioning,comm);
          sample = owned;
          const auto at_rest = this->coupling->sample_bulk(sample);

          std::vector<types::global_dof_index> ids(this->get_fe().n_dofs_per_cell());
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                cell->get_dof_indices(ids);
                for (unsigned int d=0; d<dim; ++d)
                  for (unsigned int j=0; j<this->get_fe().base_element(intro.base_elements.velocities).n_dofs_per_cell(); ++j)
                    {
                      const auto i = this->get_fe().component_to_system_index(intro.component_indices.velocities[d],j);
                      if (!this->get_dof_handler().locally_owned_dofs().is_element(ids[i]))
                        continue;
                      const auto x = this->get_mapping().transform_unit_to_real_cell(cell,this->get_fe().unit_support_point(i));
                      owned(ids[i]) = virtual_velocity(trace,x)[d];
                    }
              }
          owned.compress(VectorOperation::insert);
          sample = owned;
          const auto with_shear = this->coupling->sample_bulk(sample);
          AssertThrow(at_rest.size() == with_shear.size(),ExcInternalError());

          // The section residual has the positive driving shear. Difference
          // the actual production rows on the identical phase, parent and
          // speeds. Slip stiffness cancels; friction is disabled in both.
          const auto middle_row = [&](const auto &points)
          {
            std::vector<double> rhs(3), diagonal(3), upper(2), scale(3);
            for (const auto &p : points)
              p.add_balance(p.transfer.rate(rates), {0.,0.},rhs,diagonal,upper,scale);
            return rhs[1];
          };
          const auto rank = Utilities::MPI::this_mpi_process(comm);
          std::ofstream file(this->get_output_directory()+"section-r"+std::to_string(rank)+".txt");
          file << std::setprecision(17) << middle_row(with_shear)-middle_row(at_rest) << ' ' << at_rest.size();
          for (const auto value : this->phase_profile().normal_integrals())
            file << ' ' << value;
          file << '\n';
          file.close();
          AssertThrow(file,ExcIO());

          // On the one-cell signed-work test capture the production normal
          // traction at every active Q8 point in the same frozen bulk field.
          if (this->get_triangulation().n_global_active_cells() == 1)
            {
              std::ofstream samples(this->get_output_directory()+"compression.csv");
              samples << std::setprecision(17) << "segment,xi,rest_Pa,shear_Pa\n";
              for (unsigned int i=0; i<at_rest.size(); ++i)
                samples << at_rest[i].transfer.foot.first_vertex << ',' << at_rest[i].transfer.foot.xi << ','
                        << at_rest[i].compression << ',' << with_shear[i].compression << '\n';
              samples.close();
              AssertThrow(samples,ExcIO());
            }
        }
      private:
        double increment;
    };

    ASPECT_REGISTER_MATERIAL_MODEL(TransferWork,"fault transfer work probe",
                                   "Prescribed-rate virtual-work diagnostic for a three-vertex material trace.")
  }

  namespace Postprocess
  {
    // Apply a unit speed to the accepted trace. The area integral of chi
    // distributed to the two Q1 stations has units of m in two dimensions.
    // This measures the scalar transfer independently of the solved rates.
    template <int dim>
    class IntegratedSlipTransfer : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          const auto &material = Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
          const auto &profile = material.phase_profile();
          const unsigned int n_labels = material.labels().size();
          const auto phase = this->introspection().compositional_index_for_name(this->get_parameters().surface_energy_model.composition);
          const QGauss<dim> section_rule(8);
          const UpdateFlags flags = update_values | update_quadrature_points | update_JxW_values;
          FEValues<dim> section(this->get_mapping(), this->get_fe(), section_rule, flags);
          std::vector<double> native(n_labels), cut(n_labels);
          const auto integrate = [&](const FEValues<dim> &fe, std::vector<double> &rows)
          {
            std::vector<double> values(fe.n_quadrature_points);
            fe[this->introspection().extractors.compositional_fields[phase]].get_function_values(this->get_solution(), values);
            for (unsigned int q=0; q<values.size(); ++q)
              profile.sample(fe.quadrature_point(q), values[q]).add_load(fe.JxW(q), rows);
          };

          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                section.reinit(cell);
                integrate(section, native);
                // Integrate the same weight again, splitting cells on support
                // edges and changes in the winning segment inside overlaps.
                const auto rule = profile.geometry.cut_quadrature(cell->vertex(0), cell->vertex(GeometryInfo<dim>::vertices_per_cell-1));
                if (rule.size() == 0)
                  integrate(section, cut);
                else
                  {
                    FEValues<dim> split(this->get_mapping(), this->get_fe(), rule, flags);
                    split.reinit(cell);
                    integrate(split, cut);
                  }
              }
          const auto comm = this->get_mpi_communicator();
          Utilities::MPI::sum(native, comm, native);
          Utilities::MPI::sum(cut, comm, cut);
          unsigned int wrote_file = 1;
          if (Utilities::MPI::this_mpi_process(comm) == 0)
            {
              std::ofstream file(this->get_output_directory()+"slip-transfer-"+std::to_string(this->get_timestep_number())+".csv");
              file << std::setprecision(17) << "id,x_m,y_m,normal_integral_m,Q8_m,cut_m\n";
              for (unsigned int i=0; i<n_labels; ++i)
                file << i << ',' << material.labels()[i].position[0] << ',' << material.labels()[i].position[1] << ','
                     << profile.normal_integrals()[i] << ',' << native[i] << ',' << cut[i] << '\n';
              file.close();
              wrote_file = file.good();
            }
          AssertThrow(Utilities::MPI::min(wrote_file, comm), ExcMessage("Could not write integrated slip transfer."));
          return {"Slip transfer", "integrated accepted scalar weights"};
        }
    };

    ASPECT_REGISTER_POSTPROCESSOR(IntegratedSlipTransfer, "integrated slip transfer",
                                  "For a constant test speed, integrate the accepted scalar transfer weight with each "
                                  "Q1 station shape, using native section and support-resolved bulk quadrature.")
  }
}
