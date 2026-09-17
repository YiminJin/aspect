// Archived K5 probe; not included by the maintained BP3 plugin.
// Accepted-rate functional history. File publication is benchmark-local and
// occurs only after the ordinary accepted-state checks, never during mechanics.
namespace aspect
{
  namespace BP3Benchmark
  {
    template <int dim>
    void initialize_theta_function(const SimulatorAccess<dim> &sim, const unsigned int state)
    {
      const char *path=std::getenv("ASPECT_FAULT_THETA_HISTORY_DIAGNOSTIC");
      if (!path) return;
      static bool initialized=false;
      AssertThrow(!initialized && sim.get_timestep_number()==0,
                  ExcMessage("Functional Theta replay requires a fresh, single initialization."));
      const auto &fault=sim.get_reconstructed_fault_manager().get_fault(0);
      bool okay=true;
      if (sim.get_pcout().is_active())
        {
          std::ofstream out(path);
          out<<std::setprecision(17)<<fault.n_vertices()<<'\n';
          for (unsigned int i=0;i<fault.n_vertices();++i)
            out<<fault.vertex(i)[0]<<' '<<fault.vertex(i)[1]<<' '<<fault.get_properties(i)[state]<<'\n';
          out.close();okay=static_cast<bool>(out);
        }
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(okay),sim.get_mpi_communicator()),
                  ExcMessage("Cannot initialize functional Theta history."));
      initialized=true;
    }

    template <int dim>
    void append_theta_function(const SimulatorAccess<dim> &sim,const std::vector<double> &V)
    {
      const char *path=std::getenv("ASPECT_FAULT_THETA_HISTORY_DIAGNOSTIC");
      if (!path || sim.get_timestep_number()==0) return;
      bool okay=true;
      if (sim.get_pcout().is_active())
        {
          std::ofstream out(path,std::ios::app);
          out<<std::setprecision(17)<<sim.get_timestep_number()<<' '<<sim.get_timestep();
          for (const double v:V) out<<' '<<v;
          out<<'\n';out.close();okay=static_cast<bool>(out);
        }
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(okay),sim.get_mpi_communicator()),
                  ExcMessage("Cannot publish accepted functional Theta history."));
    }
  }
}
