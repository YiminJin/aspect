#include "phase_field_fault_stage_i.cc"
#include "phase_field_test_access.h"
#include <aspect/particle/particle_domain.h>

#include <aspect/simulator_signals.h>


namespace aspect
{
  namespace
  {
    struct ParticleHistoryMoments
    {
      unsigned int particle_count = 0;
      std::vector<double> values;
      std::vector<std::vector<double>> theta;
      std::map<types::particle_index, double> H;
    };


    template <int dim>
    ParticleHistoryMoments
    collect_particle_history_moments(const SimulatorAccess<dim> &simulator)
    {
      const auto &particle_manager = simulator.get_phase_field_handler()
                                     .get_associated_particle_manager();
      const auto &particle_data =
        particle_manager.get_property_manager().get_data_info();
      const unsigned int stress_position =
        particle_data.get_position_by_plugin_index(
          particle_manager.get_property_manager()
          .get_plugin_index_by_name("maxwell stress"));
      const unsigned int H_position =
        particle_data.get_position_by_field_name("crack_driving_force");
      const unsigned int n_values =
        SymmetricTensor<2,dim>::n_independent_components+1;

      unsigned int local_count = 0;
      std::vector<double> local_moments(2*n_values, 0.0);
      for (const auto &particle : particle_manager.get_particle_handler())
        {
          for (unsigned int component = 0; component < n_values; ++component)
            {
              const double value =
                component+1 == n_values
                ? particle.get_properties()[H_position]
                : particle.get_properties()[stress_position+component];
              local_moments[2*component] += value;
              local_moments[2*component+1] += value*value;
            }
          ++local_count;
        }

      ParticleHistoryMoments result;
      result.particle_count = Utilities::MPI::sum(
        local_count, simulator.get_mpi_communicator());
      result.values.resize(local_moments.size());
      for (unsigned int i = 0; i < local_moments.size(); ++i)
        result.values[i] = Utilities::MPI::sum(
          local_moments[i], simulator.get_mpi_communicator());
      return result;
    }


    template <int dim>
    ParticleHistoryMoments &
    supplied_initial_history()
    {
      static ParticleHistoryMoments history;
      return history;
    }


    template <int dim>
    void
    capture_supplied_initial_history(const SimulatorAccess<dim> &simulator)
    {
      auto &snapshot = supplied_initial_history<dim>();
      snapshot = collect_particle_history_moments(simulator);
      const auto &manager = simulator.get_reconstructed_fault_manager();
      if (simulator.get_timestep_number() > 0)
        {
          const auto &pm = simulator.get_phase_field_handler().get_associated_particle_manager();
          const auto H_position = pm.get_property_manager().get_data_info()
                                  .get_position_by_field_name("crack_driving_force");
          std::map<types::particle_index, double> local_H;
          for (const auto &particle : pm.get_particle_handler())
            local_H.emplace(particle.get_id(), particle.get_properties()[H_position]);
          // Advection can move particles between owners before the next check.
          for (const auto &rank_H : Utilities::MPI::all_gather(
                 simulator.get_mpi_communicator(), local_H))
            snapshot.H.insert(rank_H.begin(), rank_H.end());
          const auto position = manager.get_property_information()[
            manager.get_property_index("phase field fault state")].position;
          for (const auto &fault : manager.get_faults())
            {
              snapshot.theta.emplace_back();
              for (unsigned int vertex = 0; vertex < fault.n_vertices(); ++vertex)
                snapshot.theta.back().push_back(fault.get_properties(vertex)[position]);
            }
        }
    }
  }


  namespace Postprocess
  {
    template <int dim>
    class VerifyPhaseFieldFaultStageJFeedback : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string>
        execute(TableHandler &) override
        {
          if (this->get_timestep_number() == 0)
            {
              const ParticleHistoryMoments current =
                collect_particle_history_moments(*this);
              const ParticleHistoryMoments &supplied =
                supplied_initial_history<dim>();
              AssertThrow(supplied.particle_count > 0
                          && current.particle_count == supplied.particle_count,
                          ExcMessage("The Stage-J timestep-zero history fixture "
                                     "was not initialized consistently."));
              for (unsigned int i = 0; i < current.values.size(); ++i)
                AssertThrow(std::abs(current.values[i]-supplied.values[i])
                            <= 1.e-12*std::max(1.0,
                                              std::abs(supplied.values[i])),
                            ExcMessage("Timestep zero evolved supplied Maxwell/H "
                                       "particle history."));
            }

          if (this->get_timestep_number() > 0)
            {
              const auto &snapshot = supplied_initial_history<dim>();
              const auto &manager = this->get_reconstructed_fault_manager();
              const auto position = manager.get_property_information()[
                manager.get_property_index("phase field fault state")].position;
              bool observable_increment = false;
              double maximum_increment_ratio = 0.0;
              for (unsigned int f = 0; f < manager.get_faults().size(); ++f)
                for (unsigned int vertex = 0;
                     vertex < manager.get_fault(f).n_vertices(); ++vertex)
                  {
                    // Independent exact solution of dTheta/dt=1-V*Theta/Dc.
                    // Long-double integral evaluation does not call update_state().
                    const long double old_theta = snapshot.theta[f][vertex];
                    const long double rate =
                      manager.get_timestep_committed_slip_rate(f)[vertex]/0.04L;
                    const long double dt = this->get_timestep();
                    const long double increment =
                      (1.0L-rate*old_theta)*(-std::expm1(-rate*dt))/rate;
                    const double expected = old_theta+increment;
                    const double actual = manager.get_fault(f)
                                          .get_properties(vertex)[position];
                    const double tolerance = 1.e-12*std::abs(expected);
                    AssertThrow(std::abs(actual-expected) <= tolerance,
                                ExcMessage("Theta does not match the independent aging-law reference."));
                    observable_increment = observable_increment
                                           || std::abs(increment) > 100.0*tolerance;
                    maximum_increment_ratio = std::max(maximum_increment_ratio,
                      static_cast<double>(std::abs(increment)/tolerance));
                  }
              AssertThrow(observable_increment,
                          ExcMessage("The Theta increment is too small to detect a missing update."));

              const auto current = collect_particle_history_moments(*this);
              const unsigned int H_moment = 2*SymmetricTensor<2,dim>::n_independent_components;
              const double H_change = current.values[H_moment]-snapshot.values[H_moment];
              this->get_pcout() << "   Stage-J history resolution: Theta increment/tolerance="
                               << maximum_increment_ratio
                               << ", relative H increment="
                               << H_change/snapshot.values[H_moment] << std::endl;
              if (this->get_timestep_number() == 1)
                AssertThrow(H_change > 1.e-8*snapshot.values[H_moment],
                            ExcMessage("The feedback fixture did not produce observable H evolution."));
              const auto &pm = this->get_phase_field_handler().get_associated_particle_manager();
              const auto H_position = pm.get_property_manager().get_data_info()
                                      .get_position_by_field_name("crack_driving_force");
              for (const auto &particle : pm.get_particle_handler())
                {
                  const double H = particle.get_properties()[H_position];
                  AssertThrow(H >= snapshot.H.at(particle.get_id()),
                              ExcMessage("A particle's irreversible history decreased."));
                  // This fixture grows H on step one. After phi responds, the
                  // second step retains that maximum; strict growth each step
                  // would contradict the irreversible max rule.
                  if (this->get_timestep_number() == 2)
                    AssertThrow(H == snapshot.H.at(particle.get_id()),
                                ExcMessage("The second-step history plateau was not preserved."));
                }
              LinearAlgebra::BlockVector difference(
                this->introspection().index_sets.system_partitioning,
                this->get_mpi_communicator());
              difference = this->get_solution();
              LinearAlgebra::BlockVector old(difference);
              old = this->get_old_solution();
              difference -= old;
              const unsigned int phase_block = this->introspection()
                                               .variable("phase_field").block_index;
              const double phase_change = difference.block(phase_block).l2_norm();
              AssertThrow(phase_change > 1.e-8,
                          ExcMessage("The feedback fixture did not evolve the phase field."));
              this->get_pcout() << "   Stage-J feedback step " << this->get_timestep_number()
                               << ": H increment=" << H_change
                               << ", phase increment=" << phase_change << std::endl;
            }
          // Verify the return edge H_k -> next phase-field assembly against
          // the integrated AT1 weak form at a fixed constant probe. In this
          // uniform fixture the Q1 partition of unity removes gradient terms.
          const auto &handler = this->get_phase_field_handler();
          const auto &pm = handler.get_associated_particle_manager();
          const auto H_position = pm.get_property_manager().get_data_info()
                                  .get_position_by_field_name("crack_driving_force");
          double local_H_integral = 0, local_volume = 0;
          for (const auto &particle : pm.get_particle_handler())
            {
              const double volume = pm.get_particle_domain_handler()
                .get_particle_domain(particle.get_local_index()).volume();
              local_H_integral += volume*particle.get_properties()[H_position];
              local_volume += volume;
            }
          constexpr double phi = 0.1, m = 480000.0;
          const double denominator = (1-phi)*(1-phi)+m*phi*(1+phi);
          const double dg = -m*(1-phi)*(1+3*phi)/(denominator*denominator);
          const double expected = Utilities::MPI::sum(
            -dg*local_H_integral-240000.0*local_volume, this->get_mpi_communicator());
          LinearAlgebra::BlockVector probe(this->get_solution());
          const unsigned int b = this->introspection().variable("phase_field").block_index;
          probe.block(b) = phi;
          auto &rhs = const_cast<LinearAlgebra::BlockVector &>(this->get_system_rhs());
          auto &matrix = const_cast<LinearAlgebra::BlockSparseMatrix &>(this->get_system_matrix());
          AssertThrow(internal::PhaseFieldTestAccess<dim>::assemble(handler, matrix, rhs, probe),
                      ExcMessage("The Stage-J phase-field probe is inadmissible."));
          double local_sum = 0;
          for (const auto index : rhs.block(b).locally_owned_elements())
            local_sum += rhs.block(b)[index];
          const double actual = Utilities::MPI::sum(local_sum, this->get_mpi_communicator());
          AssertThrow(std::abs(actual-expected) < 1.e-10*std::abs(expected),
                      ExcMessage("Phase-field assembly does not consume the newly committed H."));
          if (this->get_timestep_number() == 2)
            verify_history_failure_preservation();
          return {"Reconstructed-fault Stage-J history feedback:",
                  "verified"};
        }

      private:
        void verify_history_failure_preservation()
        {
          auto &manager = this->get_reconstructed_fault_manager();
          auto &particles = this->get_phase_field_handler()
                            .get_associated_particle_manager().get_particle_handler();
          const auto &associations = manager.get_locally_owned_particle_fault_associations();
          const auto H_position = this->get_phase_field_handler()
            .get_associated_particle_manager().get_property_manager().get_data_info()
            .get_position_by_field_name("crack_driving_force");
          const unsigned int rank = Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
          const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(this->get_mpi_communicator());
          types::particle_index selected_id = numbers::invalid_unsigned_int;
          unsigned int index=0;
          std::map<types::particle_index, std::vector<double>> particle_values;
          for (const auto &particle : particles)
            {
              const auto values = particle.get_properties();
              particle_values.emplace(particle.get_id(), std::vector<double>(values.begin(), values.end()));
              if (associations[index++].active && selected_id == numbers::invalid_unsigned_int)
                selected_id = particle.get_id();
            }
          const unsigned int owner = Utilities::MPI::min(
            selected_id == numbers::invalid_unsigned_int ? n_ranks : rank,
            this->get_mpi_communicator());
          AssertThrow(owner < n_ranks, ExcMessage("No active particle for the failed-history fixture."));
          std::vector<std::vector<double>> surface_values, slip_rates;
          for (unsigned int f=0; f<manager.get_faults().size(); ++f)
            {
              slip_rates.push_back(manager.get_timestep_committed_slip_rate(f));
              for (unsigned int v=0; v<manager.get_fault(f).n_vertices(); ++v)
                {
                  const auto values = manager.get_fault(f).get_properties(v);
                  surface_values.emplace_back(values.begin(), values.end());
                }
            }
          LinearAlgebra::BlockVector bulk(
            this->introspection().index_sets.system_partitioning, this->get_mpi_communicator());
          bulk = this->get_solution();

          // Only one owner sees invalid H. Every rank must leave candidate
          // construction together, before either projection/commit can strand
          // another rank or publish a partial update.
          if (rank == owner)
            for (auto &particle : particles)
              if (particle.get_id() == selected_id)
                particle.get_properties()[H_position] = -1.0;
          auto &model = const_cast<MaterialModel::PhaseFieldFault<dim> &>(
            Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
              this->get_material_model()));
          std::string error;
          try
            {
              model.commit_reconstructed_fault_mechanical_history(this->get_solution());
            }
          catch (const std::exception &exception)
            {
              error = exception.what();
            }
          AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(
                        error.find("Stored crack-driving history is inadmissible") != std::string::npos),
                        this->get_mpi_communicator()) == 1,
                      ExcMessage("The one-owner history failure was not propagated to every rank."));
          for (auto &particle : particles)
            {
              const auto &saved = particle_values.at(particle.get_id());
              auto values = particle.get_properties();
              if (rank == owner && particle.get_id() == selected_id)
                {
                  AssertThrow(values[H_position] == -1.0,
                              ExcMessage("A failed history operation overwrote the failing input."));
                  values[H_position] = saved[H_position];
                }
              AssertThrow(std::equal(values.begin(), values.end(), saved.begin()),
                          ExcMessage("A failed history candidate changed particle properties."));
            }
          index=0;
          for (unsigned int f=0; f<manager.get_faults().size(); ++f)
            {
              AssertThrow(manager.get_slip_rate(f) == slip_rates[f]
                          && manager.get_timestep_committed_slip_rate(f) == slip_rates[f],
                          ExcMessage("A failed history candidate changed V."));
              for (unsigned int v=0; v<manager.get_fault(f).n_vertices(); ++v)
                {
                  const auto values = manager.get_fault(f).get_properties(v);
                  AssertThrow(std::equal(values.begin(), values.end(), surface_values[index++].begin()),
                              ExcMessage("A failed history candidate changed surface histories."));
                }
            }
          LinearAlgebra::BlockVector difference(bulk);
          difference = this->get_solution();
          difference -= bulk;
          AssertThrow(difference.l2_norm() == 0.0,
                      ExcMessage("A failed history candidate changed the production bulk state."));
          this->get_pcout() << "Stage-J one-owner failed history candidate: all state preserved" << std::endl;
        }
    };


    ASPECT_REGISTER_POSTPROCESSOR(
      VerifyPhaseFieldFaultStageJFeedback,
      "verify phase field fault stage j history feedback",
      "Verify timestep-zero preservation and observable H/phase/Theta feedback.")
  }


  template <int dim>
  void
  connect_stage_j_history_signals(SimulatorSignals<dim> &signals)
  {
    signals.start_timestep.connect(&capture_supplied_initial_history<dim>);
  }


  ASPECT_REGISTER_SIGNALS_CONNECTOR(connect_stage_j_history_signals<2>,
                                    connect_stage_j_history_signals<3>)
}
