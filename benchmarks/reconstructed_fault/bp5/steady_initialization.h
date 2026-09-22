// Explicit modified-BP5 initial condition. No change to subsequent split aging.
namespace aspect
{
  namespace BP3Benchmark
  {
    std::vector<double> prepared_initial_state;

    template <int dim>
    void initialize_steady_prestress(const SimulatorAccess<dim> &sim,
                                     MaterialModel::PhaseFieldFault<dim> &model)
    {
      auto &manager = sim.get_reconstructed_fault_manager();
      auto &fault = manager.get_fault(0);
      const unsigned int n = fault.n_vertices();
      const auto background = manager.get_property_index("background tractions");
      const auto correction = manager.get_property_index("BP3 fixed shear correction");
      const auto p = manager.get_property_information()[background].position;
      const auto c = manager.get_property_information()[correction].position;
      const auto state = manager.get_property_information()[manager.get_property_index(
        "phase field fault state")].position;
      const double theta0 = model.characteristic_fault_slip_distance()/BP3::Vinit;
      const auto &names = sim.introspection().chemical_composition_field_names();
      AssertThrow(names.size() == 1 && names[0] == "strengthening",
                  ExcMessage("BP5 initial state requires the two-material strengthening mixture."));
      const auto chemical = manager.get_property_information()[manager.get_property_index(
        "phase field fault chemical composition strengthening")].position;
      std::vector<double> fractions(n);
      prepared_initial_state.resize(n);

      // Set state after production material projection. The law mixes a
      // arithmetically, so (a_eff-a_VW)/(a_VS-a_VW) is exactly its VS fraction.
      // Initial particle state is only a seed; this Q1 nodal state is authoritative.
      for (unsigned int i = 0; i < n; ++i)
        {
          auto data = fault.get_properties(i);
          const auto mixture = MaterialModel::MaterialUtilities::compute_composition_fractions({data[chemical]});
          const double f = mixture[1];
          AssertThrow(std::isfinite(f) && f >= -32.*std::numeric_limits<double>::epsilon()
                        && f <= 1.+32.*std::numeric_limits<double>::epsilon(),
                      ExcMessage("Invalid BP5 projected strengthening fraction."));
          fractions[i] = std::clamp(f, 0., 1.);
          data[state] = prepared_initial_state[i] = theta0*BP3::initial_state_ratio(fractions[i]);
          // Replace captured background and its correction together.
          data[p] = 0.; data[p+1] = BP3::sigma0;
          data[c] = 0.; data[c+1] = 0.; data[c+2] = 1.;
        }
      model.set_reconstructed_fault_background_traction_property(background, correction);

#ifdef ASPECT_BP5_NORMAL_CONTROL
      // Reuse the reference's effective traction coefficients, not a new weak
      // prestress solve. Projection/state are independently rebuilt and checked.
      const char *reference = std::getenv("ASPECT_BP5_INITIAL_REFERENCE");
      AssertThrow(reference, ExcMessage("Normal control requires saved initialization coefficients."));
      std::ifstream saved(reference);
      AssertThrow(saved, ExcMessage("Cannot open saved BP5 initialization."));
      std::string line;
      std::getline(saved, line);
      for (unsigned int i = 0; i < n; ++i)
        {
          AssertThrow(std::getline(saved, line), ExcMessage("Incomplete initial reference."));
          std::replace(line.begin(), line.end(), ',', ' ');
          std::istringstream row(line);
          unsigned int node;
          double xd, theta, shear, normal, weight, friction, damping, load, error, raw, f, ratio;
          AssertThrow(row >> node >> xd >> theta >> shear >> normal >> weight >> friction >> damping
                      >> load >> error >> raw >> f >> ratio, ExcMessage("Invalid initial reference row."));
          auto data = fault.get_properties(i);
          AssertThrow(node == i && std::abs(xd-BP3::down_dip(fault.vertex(i)[0],fault.vertex(i)[1])) < 1e-9
                      && std::abs(theta-data[state]) < 1e-7 && std::abs(raw-data[chemical]) < 1e-13
                      && std::abs(f-fractions[i]) < 1e-13 && normal == BP3::sigma0,
                      ExcMessage("Normal control initial geometry, projected material or state differs."));
          data[p] = shear;
          data[p+1] = normal;
        }
      AssertThrow(!std::getline(saved, line), ExcMessage("Unexpected extra initial reference row."));
      sim.get_pcout() << "   BP5 normal control: imported fixed background; no prestress recalibration." << std::endl;
      return;
#endif

      // Private zero-perturbation input retains the actual phase, temperature
      // and chemical fields. This is a constitutive quadrature probe, not a
      // physical velocity iterate: keep the phase lift, then remove velocity
      // and pressure from the probe (including the side-velocity lifting).
      const auto &intro = sim.introspection();
      const auto comm = sim.get_mpi_communicator();
      LinearAlgebra::BlockVector owned(intro.index_sets.system_partitioning, comm);
      owned = sim.get_solution();
      sim.get_current_constraints().distribute(owned);
      owned.block(intro.block_indices.velocities) = 0.;
      owned.block(intro.block_indices.pressure) = 0.;
      LinearAlgebra::BlockVector probe(intro.index_sets.system_partitioning,
                                       intro.index_sets.system_relevant_partitioning, comm);
      probe = owned;
      const ReconstructedFaultVector V(1, std::vector<double>(n, BP3::Vinit));
      auto &surface = sim.get_reconstructed_fault_surface_system();
      const auto before = surface.evaluate_surface_residual(probe, V);
      AssertThrow(std::abs(before.minimum_normal_traction-BP3::sigma0) < 1e-6
                    && std::abs(before.maximum_normal_traction-BP3::sigma0) < 1e-6,
                  ExcMessage("Initial prestress probe must have zero normal stress perturbation."));

      // Production assembly supplies the projected mixture, native work measure,
      // endpoint continuation, Q1 products and MPI reduction. Include resistance
      // only: the zero-strain probe's crack-induced shear is NOT prestress.
      std::vector<double> rhs(n);
      for (unsigned int i = 0; i < n; ++i)
        rhs[i] = before.friction_traction[0][i] + before.damping_traction[0][i];
      const auto shear = ReconstructedFaultUtilities::solve_tridiagonal_system(
        before.mass_diagonal[0], before.mass_off_diagonal[0], rhs);
      for (unsigned int i = 0; i < n; ++i)
        fault.get_properties(i)[p] = shear[i];

      // Re-evaluate through the same residual path, without factorizing K or
      // publishing history. Subtraction isolates the newly added background.
      const auto after = surface.evaluate_surface_residual(probe, V);
      double error = 0.;
      std::ofstream out;
      if (sim.get_pcout().is_active())
        {
          out.open(sim.get_output_directory()+"steady_initialization.csv");
          out << std::setprecision(17)
              << "node,xd,Theta0,tau_bg,sigma_bg,weight,friction_load,damping_load,background_load,weak_error_Pa,projected_chemical,strengthening_fraction,R0\n";
        }
      for (unsigned int i = 0; i < n; ++i)
        {
          double weight = before.mass_diagonal[0][i];
          if (i > 0) weight += before.mass_off_diagonal[0][i-1];
          if (i+1 < n) weight += before.mass_off_diagonal[0][i];
          const double load = after.shear_traction[0][i]-before.shear_traction[0][i];
          const double residual = (load-rhs[i])/weight;
          error = std::max(error, std::abs(residual));
          if (out)
            out << i << ',' << BP3::down_dip(fault.vertex(i)[0],fault.vertex(i)[1])
                << ',' << prepared_initial_state[i] << ',' << shear[i] << ',' << BP3::sigma0 << ',' << weight
                << ',' << before.friction_traction[0][i] << ',' << before.damping_traction[0][i]
                << ',' << load << ',' << residual << ',' << fault.get_properties(i)[chemical]
                << ',' << fractions[i] << ',' << prepared_initial_state[i]/theta0 << '\n';
        }
      AssertThrow(error < 1e-5, ExcMessage("Steady BP5 native weak prestress balance failed."));
      sim.get_pcout() << "   BP5 projected-state initialization: R_VW=" << BP3::weakening_initial_state_ratio
                     << ", steady Theta=" << theta0 << " s, native weak prestress error=" << error << " Pa." << std::endl;
    }
  }
}
