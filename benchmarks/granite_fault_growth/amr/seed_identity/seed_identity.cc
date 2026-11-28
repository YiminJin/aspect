/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_signals.h>

namespace aspect::Postprocess
{
  /** Manufactured DG endpoints for testing the AMR criterion in isolation.
   * Only the returned phase is replaced, before the timestep manager decides
   * whether to refine or accept. This fixture does not measure fracture work. */
  template <int dim>
  class SeedIdentityFixture : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      void initialize() override
      {
        this->get_signals().post_set_initial_state.connect([this](const auto &)
        {
          set_manufactured_phase(0);
        });
        this->get_signals().post_nonlinear_solver.connect([this](const auto &)
        {
          set_manufactured_phase(this->get_timestep_number());
        });
      }

      std::pair<std::string,std::string> execute(TableHandler &) override
      {
        return {"Seed identity fixture", "manufactured endpoint accepted"};
      }

    private:
      void set_manufactured_phase(const unsigned int step)
      {
        const auto &fe=this->get_fe();
        const auto component=this->introspection().component_indices.compositional_fields[0];
        const auto &comm=this->get_mpi_communicator();
        LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,comm);
        owned=this->get_solution();
        std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());
        double source_edge=0., receiver_edge=0.;
        for (const auto &cell : this->get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const auto center=cell->center();
              const bool column=center[0]>2.75 && center[0]<3.;
              const bool island=step>=1 && center[0]>2.25 && center[0]<2.5 && center[1]<.5;
              const bool damage=island || (column && (center[1]<1.75 || step>=2));
              cell->get_dof_indices(dofs);
              for (unsigned int j=0; j<9; ++j)
                owned[dofs[fe.component_to_system_index(component,j)]]=damage ? .02 : .004;
              if (column && center[1]>=1.75)
                receiver_edge=std::max(receiver_edge,cell->extent_in_direction(0));
              if (column && center[1]<.5)
                source_edge=std::max(source_edge,cell->extent_in_direction(0));
            }
        owned.compress(VectorOperation::insert);
        // The test deliberately supplies a different trial to the existing
        // pre-accept inspector. No simulator hook is needed for this fixture.
        const_cast<LinearAlgebra::BlockVector &>(this->get_solution())=owned;
        source_edge=Utilities::MPI::max(source_edge,comm);
        receiver_edge=Utilities::MPI::max(receiver_edge,comm);
        this->get_pcout()<<"Manufactured seed step "<<step<<", source edge "<<source_edge
                         <<" m, receiver edge "<<receiver_edge<<" m"<<std::endl;
      }
  };

  ASPECT_REGISTER_POSTPROCESSOR(SeedIdentityFixture,"manufactured seed identity",
                                "Supply prescribed DG endpoints to exercise the growing fault refinement criterion.")
}
