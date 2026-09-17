// BP3 initialization of fixed prestress data, not a runtime cohesive law.
namespace aspect
{
  namespace BP3Benchmark
  {
    std::string mature_prestress_file;

    template <int dim>
    void
    initialize_mature_prestress (const SimulatorAccess<dim> &sim, MaterialModel::PhaseFieldFault<dim> &model)
    {
      auto &manager = sim.get_reconstructed_fault_manager ();
      auto &fault = manager.get_fault (0);
      const auto background = manager.get_property_index ("background tractions");
      const auto correction = manager.get_property_index ("BP3 fixed shear correction");
      const auto p = manager.get_property_information ()[background].position;
      const auto c = manager.get_property_information ()[correction].position;
      AssertThrow (!mature_prestress_file.empty (),
                   ExcMessage ("Mature BP3 requires its captured initial prestress file."));
      std::istringstream in (Utilities::read_and_distribute_file_content (
          Utilities::expand_ASPECT_SOURCE_DIR (mature_prestress_file), sim.get_mpi_communicator ()));
      unsigned int n;
      AssertThrow (in >> n && n == fault.n_vertices (), ExcMessage ("Mature prestress fault size mismatch."));
      for (unsigned int v = 0; v < n; ++v)
        {
          double x, y, shear, normal, a, b, d;
          AssertThrow (in >> x >> y >> shear >> normal >> a >> b >> d,
                       ExcMessage ("Incomplete mature prestress snapshot."));
          AssertThrow (x == fault.vertex (v)[0] && y == fault.vertex (v)[1],
                       ExcMessage ("Mature prestress geometry mismatch."));
          auto data = fault.get_properties (v);
          data[p] = shear;
          data[p + 1] = normal;
          data[c] = a;
          data[c + 1] = b;
          data[c + 2] = d;
        }
      model.set_reconstructed_fault_background_traction_property (background, correction);
    }

  }
}
