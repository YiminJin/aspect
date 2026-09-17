// BP3 initialization of fixed prestress data, not a runtime cohesive law.
namespace aspect
{
  namespace BP3Benchmark
  {
    std::string mature_prestress_file;

    template <int dim>
    void initialize_mature_prestress(const SimulatorAccess<dim> &sim,
                                    MaterialModel::PhaseFieldFault<dim> &model)
    {
      auto &manager=sim.get_reconstructed_fault_manager();
      auto &fault=manager.get_fault(0);
      const auto background=manager.get_property_index("background tractions");
      const auto correction=manager.get_property_index("BP3 fixed shear correction");
      const auto p=manager.get_property_information()[background].position;
      const auto c=manager.get_property_information()[correction].position;
      AssertThrow(!mature_prestress_file.empty(),ExcMessage("Mature BP3 requires its captured initial prestress file."));
      std::istringstream in(Utilities::read_and_distribute_file_content(
        Utilities::expand_ASPECT_SOURCE_DIR(mature_prestress_file),sim.get_mpi_communicator()));
      unsigned int n;AssertThrow(in>>n && n==fault.n_vertices(),ExcMessage("Mature prestress fault size mismatch."));
      for (unsigned int v=0;v<n;++v)
        {
          double x,y,shear,normal,a,b,d;
          AssertThrow(in>>x>>y>>shear>>normal>>a>>b>>d,ExcMessage("Incomplete mature prestress snapshot."));
          AssertThrow(x==fault.vertex(v)[0] && y==fault.vertex(v)[1],ExcMessage("Mature prestress geometry mismatch."));
          auto data=fault.get_properties(v);
          data[p]=shear;data[p+1]=normal;
          data[c]=a;data[c+1]=b;data[c+2]=d;
        }
      model.set_reconstructed_fault_background_traction_property(background,correction);
      sim.get_pcout()<<"Mature BP3: fixed effective prestress initialized; C and cohesive energy are zero."<<std::endl;
    }

    // Diagnostics use the same domain rule as mechanics. This projection is
    // output only: mechanics always evaluates the rational background directly.
    template <int dim>
    std::vector<double> projected_mature_background(const SimulatorAccess<dim> &sim,
                                                  const ReconstructedFaultSurfaceResidual &weak)
    {
      auto &manager=sim.get_reconstructed_fault_manager();
      const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model());
      std::vector<double> local(manager.get_fault(0).n_vertices(),0.),global(local.size());
      for (const auto &parent:manager.get_locally_owned_particle_fault_associations())
        if (parent.active)
          for (const auto &q:parent.quadrature)
            {
              const double b=model.reconstructed_fault_background_tractions(0,q.segment_index,q.xi).first;
              local[q.segment_index]+=q.weight*(1-q.xi)*b;
              local[q.segment_index+1]+=q.weight*q.xi*b;
            }
      Utilities::MPI::sum(local,sim.get_mpi_communicator(),global);
      return ReconstructedFaultUtilities::solve_tridiagonal_system(
        weak.mass_diagonal[0],weak.mass_off_diagonal[0],global);
    }
  }
}
