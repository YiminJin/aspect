#include "phase_field_fault_stage_i.cc"
#include "phase_field_test_access.h"
#include <aspect/particle/particle_domain.h>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class VerifyFrozenPhaseField : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          const auto &handler = this->get_phase_field_handler();
          const unsigned int b = this->introspection().variable("phase_field").block_index;
          auto &matrix = const_cast<LinearAlgebra::BlockSparseMatrix &>(this->get_system_matrix());
          auto &rhs = const_cast<LinearAlgebra::BlockVector &>(this->get_system_rhs());
          LinearAlgebra::BlockVector probe(this->get_solution());
          probe.block(b) = 0.1;
          AssertThrow(internal::PhaseFieldTestAccess<dim>::assemble(handler, matrix, rhs, probe),
                      ExcMessage("The fixed phase-field probe is inadmissible."));

          // Independent AT1 weak residual at constant phi: the gradient term
          // vanishes, leaving -(H*g'(phi)+Gc/(c0*l))*w_i*Vp. Coefficients here
          // are the fixture's constants, not the production derivative routine.
          constexpr double phi = 0.1, m = 480000.0;
          const double denominator = (1-phi)*(1-phi)+m*phi*(1+phi);
          const double dg = -m*(1-phi)*(1+3*phi)/(denominator*denominator);
          LinearAlgebra::BlockVector reference(
            this->introspection().index_sets.system_partitioning, this->get_mpi_communicator());
          const auto &pm = handler.get_associated_particle_manager();
          const auto H_position = pm.get_property_manager().get_data_info()
                                  .get_position_by_field_name("crack_driving_force");
          const auto &vertex_dofs = internal::PhaseFieldTestAccess<dim>::vertex_dofs(handler);
          for (const auto &particle : pm.get_particle_handler())
            {
              const auto domain = pm.get_particle_domain_handler()
                                  .get_particle_domain(particle.get_local_index());
              const unsigned int n = domain.n_relevant_vertices();
              std::vector<types::global_dof_index> dofs(n);
              Vector<double> local(n);
              const double density = particle.get_properties()[H_position]*dg+240000.0;
              for (unsigned int i=0; i<n; ++i)
                {
                  dofs[i] = vertex_dofs[domain.relevant_vertex_index(i)];
                  local[i] = -density*domain.weighting_function_value(i)*domain.volume();
                }
              this->get_current_constraints().distribute_local_to_global(local, dofs, reference);
            }
          reference.compress(VectorOperation::add);
          const double norm = rhs.block(b).l2_norm();
          reference.block(b) -= rhs.block(b);
          AssertThrow(reference.block(b).l2_norm() < 1.e-11*norm,
                      ExcMessage("The production phase-field residual differs from the AT1 weak form."));

          // Frozen H, composition, geometry, and particle domains define the
          // same discrete problem at timestep zero and at each real timestep.
          std::vector<double> entries;
          for (const auto row : rhs.block(b).locally_owned_elements())
            {
              entries.push_back(rhs.block(b)[row]);
              for (auto p=matrix.block(b,b).begin(row); p!=matrix.block(b,b).end(row); ++p)
                entries.push_back(p->value());
            }
          if (this->get_timestep_number() == 0)
            initial_entries = entries;
          AssertDimension(entries.size(), initial_entries.size());
          double error = 0, scale = 0;
          for (unsigned int i=0; i<entries.size(); ++i)
            {
              error = std::max(error, std::abs(entries[i]-initial_entries[i]));
              scale = std::max(scale, std::abs(initial_entries[i]));
            }
          AssertThrow(Utilities::MPI::max(error, this->get_mpi_communicator())
                      < 1.e-12*Utilities::MPI::max(scale, this->get_mpi_communicator()),
                      ExcMessage("Frozen-input phase-field residual/Jacobian changed with timestep."));
          return {"Frozen-input phase-field residual and Jacobian:", "verified"};
        }
      private:
        std::vector<double> initial_entries;
    };
    ASPECT_REGISTER_POSTPROCESSOR(VerifyFrozenPhaseField, "verify frozen phase field",
      "Verify the AT1 weak residual and timestep-independent frozen-input operator.")
  }
}
