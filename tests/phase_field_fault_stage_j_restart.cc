#include "phase_field_fault_stage_j.cc"
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace stage_j_restart
  {
    template <int dim>
    std::vector<double> fingerprint(const SimulatorAccess<dim> &sim)
    {
      auto result = collect_particle_history_moments(sim).values;
      const auto &manager = sim.get_reconstructed_fault_manager();
      for (unsigned int f=0; f<manager.get_faults().size(); ++f)
        for (unsigned int v=0; v<manager.get_fault(f).n_vertices(); ++v)
          {
            const auto values = manager.get_fault(f).get_properties(v);
            result.insert(result.end(), values.begin(), values.end());
            result.push_back(manager.get_timestep_committed_slip_rate(f)[v]);
            for (unsigned int d=0; d<dim; ++d)
              result.push_back(manager.get_fault(f).vertex(v)[d]);
          }
      LinearAlgebra::BlockVector owned(sim.introspection().index_sets.system_partitioning,
                                      sim.get_mpi_communicator());
      owned = sim.get_solution();
      for (unsigned int b=0; b<owned.n_blocks(); ++b)
        result.push_back(owned.block(b).l2_norm());
      return result;
    }

    template <int dim>
    std::vector<double> &restored_fingerprint()
    {
      static std::vector<double> values;
      return values;
    }

    void compare(const std::vector<double> &actual,
                 const std::vector<double> &expected)
    {
      AssertThrow(actual.size() == expected.size() && !expected.empty(),
                  ExcMessage("Missing or incompatible Stage-J restart reference."));
      for (unsigned int i=0; i<actual.size(); ++i)
        AssertThrow(std::abs(actual[i]-expected[i])
                    <= 1.e-10*std::max(1.0,std::abs(expected[i])),
                    ExcMessage("Stage-J restart changed physical state entry "+std::to_string(i)));
    }

    template <int dim>
    void verify_restored_state(const SimulatorAccess<dim> &sim)
    {
      auto &expected = restored_fingerprint<dim>();
      if (!expected.empty())
        {
          compare(fingerprint(sim), expected);
          expected.clear();
          sim.get_pcout() << "Stage-J checkpoint histories, V, geometry, and bulk: verified" << std::endl;
        }
    }

    template <int dim>
    void connect(SimulatorSignals<dim> &signals)
    {
      signals.start_timestep.connect(&verify_restored_state<dim>);
    }
    ASPECT_REGISTER_SIGNALS_CONNECTOR(connect<2>, connect<3>)
  }

  namespace Postprocess
  {
    template <int dim>
    class VerifyStageJRestart : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          checkpoint_values = stage_j_restart::fingerprint(*this);
          if (this->get_parameters().resume_computation)
            {
              std::vector<double> expected;
              if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
                {
                  std::ifstream input("output-phase_field_fault_stage_j_restart_create/stage-j-final-state.txt");
                  double value;
                  while (input >> value)
                    expected.push_back(value);
                }
              expected = Utilities::MPI::broadcast(this->get_mpi_communicator(), expected, 0);
              stage_j_restart::compare(checkpoint_values, expected);
              return {"Stage-J resumed feedback versus uninterrupted run:", "verified"};
            }
          if (this->get_timestep_number() == 2
              && Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
            {
              std::ofstream output(this->get_output_directory()+"stage-j-final-state.txt");
              output << std::setprecision(17);
              for (const double value : checkpoint_values)
                output << value << '\n';
            }
          return {"", ""};
        }

        void save(std::map<std::string,std::string> &strings) const override
        {
          std::ostringstream output;
          {
            aspect::oarchive archive(output);
            archive << checkpoint_values;
          }
          strings["StageJRestart"] = output.str();
        }

        void load(const std::map<std::string,std::string> &strings) override
        {
          std::istringstream input(strings.at("StageJRestart"));
          aspect::iarchive archive(input);
          archive >> stage_j_restart::restored_fingerprint<dim>();
        }
      private:
        std::vector<double> checkpoint_values;
    };
    ASPECT_REGISTER_POSTPROCESSOR(VerifyStageJRestart, "verify stage j restart",
      "Verify saved histories before resumed mechanics and compare resumed feedback with an uninterrupted run.")
  }
}
