/* Focused value-cache lifecycle checks on the existing distributed Q1 fixture. */
#include "phase_field_fault_test_access.h"
#include <aspect/phase_field.h>
#include <aspect/plugins.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <fstream>
#include <sstream>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class VerifyFaultIhCache : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          using Access = MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
          auto &model = const_cast<MaterialModel::PhaseFieldFault<dim> &>(
            Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model()));
          auto &manager = this->get_reconstructed_fault_manager();
          auto &solution = const_cast<LinearAlgebra::BlockVector &>(this->get_solution());
          const LinearAlgebra::BlockVector saved_solution(solution);
          const auto saved_fault = manager.get_fault(0);
          const unsigned int block = this->introspection().variable("phase_field").block_index;
          const auto calculate = [&]() { return Access::compute_normalization_integrals(model); };
          const auto status = [&]() { return Access::normalization_cache_status(model); };
          const auto restore = [&]()
          {
            solution = saved_solution;
            manager.get_fault(0) = saved_fault;
            manager.invalidate_particle_projection_cache();
            Access::invalidate_normalization_cache(model);
          };
          if (const char *saved_phase = std::getenv("ASPECT_IH_SAVED_PHASE"))
            {
              // Frozen-data check only: use the saved FE nodes and actual
              // fault coordinates on an identical bulk/fault discretization.
              // No mechanics/history update consumes the substituted state.
              try
                {
                  AssertThrow(dim==2,ExcNotImplemented());
                  std::ifstream input(saved_phase);
                  AssertThrow(input.is_open(),ExcMessage("Cannot read saved I_h phase fixture."));
                  std::string line;
                  std::getline(input,line);
                  std::map<std::array<double,2>,double> values;
                  while (std::getline(input,line))
                    {
                      std::replace(line.begin(),line.end(),',',' ');
                      std::istringstream row(line);
                      unsigned long long old_dof;
                      std::array<double,2> point;
                      double phi;
                      AssertThrow(bool(row>>old_dof>>point[0]>>point[1]>>phi),ExcMessage("Malformed saved phase row."));
                      values[point]=phi;
                    }
                  LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,
                                                   this->get_mpi_communicator());
                  owned=saved_solution;
                  const unsigned int component=this->introspection().variable("phase_field").first_component_index;
                  std::vector<types::global_dof_index> dofs(this->get_fe().n_dofs_per_cell());
                  for (const auto &cell : this->get_dof_handler().active_cell_iterators())
                    if (cell->is_locally_owned())
                      {
                        cell->get_dof_indices(dofs);
                        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                          {
                            const auto id=dofs[this->get_fe().component_to_system_index(component,v)];
                            if (!this->get_dof_handler().locally_owned_dofs().is_element(id)) continue;
                            const auto entry=values.find({cell->vertex(v)[0],cell->vertex(v)[1]});
                            AssertThrow(entry!=values.end(),ExcMessage("Saved phase and test mesh do not match exactly."));
                            owned[id]=entry->second;
                          }
                      }
                  owned.compress(VectorOperation::insert);
                  solution=owned;
                  const char *saved_surface=std::getenv("ASPECT_IH_SAVED_SURFACE");
                  AssertThrow(saved_surface,ExcMessage("Saved surface coordinates are required."));
                  std::ifstream surface(saved_surface);
                  AssertThrow(surface.is_open(),ExcMessage("Cannot read saved surface fixture."));
                  std::getline(surface,line);
                  unsigned int vertices=0;
                  while (std::getline(surface,line))
                    {
                      std::replace(line.begin(),line.end(),',',' ');
                      std::istringstream row(line);
                      unsigned int fault,vertex;
                      Point<dim> point;
                      AssertThrow(bool(row>>fault>>vertex>>point[0]>>point[1]),ExcMessage("Malformed saved surface row."));
                      AssertThrow(fault==0 && vertex<manager.get_fault(0).n_vertices(),
                                  ExcMessage("Saved/test fault topology mismatch."));
                      const_cast<Point<dim> &>(manager.get_fault(0).vertex(vertex))=point;
                      ++vertices;
                    }
                  AssertThrow(vertices==manager.get_fault(0).n_vertices(),ExcMessage("Incomplete saved surface fixture."));
                  manager.invalidate_particle_projection_cache();
                  calculate();
                }
              catch (...) { restore(); throw; }
              restore();
              return {"Saved FE I_h:","saved phase/geometry comparison verified; working state restored"};
            }
          try
            {
              const auto original = calculate();
              auto before = status();
              AssertThrow(calculate() == original, ExcMessage("Cache hit changed I_h."));
              AssertThrow(std::get<1>(status()) == std::get<1>(before)+1
                          && std::get<2>(status()) == std::get<2>(before)
                          && std::get<3>(status()) == 0,
                          ExcMessage("Unchanged preparation integrated or requested FE data."));

              // Modify owned data on rank zero only: one rank's miss must make
              // every rank follow the same collective integration path.
              LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,
                                               this->get_mpi_communicator());
              owned = saved_solution;
              if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
                owned.block(block) *= 0.999;
              solution = owned;
              const auto changed = calculate();
              AssertThrow(changed != original && std::get<2>(status()) == std::get<2>(before)+1,
                          ExcMessage("Owned phase mutation did not invalidate I_h."));
              solution = saved_solution;
              AssertThrow(calculate() == original, ExcMessage("Restored phase changed recomputed I_h."));

              // Test-only relocation keeps the topology/property layout intact.
              // It deliberately bypasses append-version changes, so coordinate
              // equality must also protect against restored/replaced geometry.
              before = status();
              for (unsigned int v=0; v<manager.get_fault(0).n_vertices(); ++v)
                const_cast<Point<dim> &>(manager.get_fault(0).vertex(v))[1] += 1e-5;
              manager.invalidate_particle_projection_cache();
              calculate();
              AssertThrow(std::get<2>(status()) == std::get<2>(before)+1,
                          ExcMessage("Changed geometry reused I_h."));
              manager.get_fault(0) = saved_fault;
              manager.invalidate_particle_projection_cache();
              AssertThrow(calculate() == original, ExcMessage("Restored geometry changed I_h."));

              // A singular profile must fail collectively without publishing a
              // partially computed value cache; restoring the state must recover.
              owned = saved_solution;
              owned.block(block) = 1.0;
              solution = owned;
              bool failed = false;
              try { calculate(); }
              catch (const std::exception &) { failed = true; }
              AssertThrow(failed && !std::get<0>(status()),
                          ExcMessage("Failed integration published a valid cache."));
              solution = saved_solution;
              AssertThrow(calculate() == original, ExcMessage("Failure recovery changed I_h."));

              // Exercise the same invalidator installed on checkpoint loading.
              // No cache is serialized; this is not a full checkpoint I/O test.
              before = status();
              Access::invalidate_normalization_cache(model);
              AssertThrow(!std::get<0>(status()), ExcMessage("Restart invalidator left cache valid."));
              AssertThrow(calculate() == original && std::get<2>(status()) == std::get<2>(before)+1,
                          ExcMessage("Restart invalidation reused stale values."));

              const auto response = Access::compute_cohesive_response(.7, 1e10, .03, .03,
                                                                     1e6, 2e-6, .2, .2);
              AssertThrow(response.history_correction == 0.0,
                          ExcMessage("Identical localization has a history correction."));
              const double original_history = .7*1e6/1e10*(.2*.03/.03-.2);
              const double original_rate = .2/.03*2e-6 + original_history;
              AssertThrow(std::abs(response.crack_strain_rate-original_rate)
                          <= 8*std::numeric_limits<double>::epsilon()*original_rate,
                          ExcMessage("Frozen localization changed the uncached formula."));
            }
          catch (...) { restore(); throw; }
          restore();
          return {"I_h value cache:", "unchanged/phase/geometry/failure/restart invalidation verified"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(VerifyFaultIhCache,
                                 "verify fault I h cache", "Exact distributed I_h value-cache lifecycle checks.")
  }
}
