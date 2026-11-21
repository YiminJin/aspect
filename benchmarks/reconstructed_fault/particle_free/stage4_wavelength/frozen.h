// Benchmark-only reconstruction of a frozen derivative specimen, not restart
// conversion or preparation of a physically equilibrated slipping background.
void import_frozen_phase()
{
  const char *directory=std::getenv("ASPECT_WAVELENGTH_INPUT");
  AssertThrow(directory,ExcMessage("Frozen wavelength input directory missing."));
  std::ifstream file(std::string(directory)+"/phase.txt");
  unsigned int nx,ny; file>>nx>>ny;
  std::vector<std::array<double,9>> coefficients(nx*ny);
  for(auto &row:coefficients)for(auto &v:row)file>>v;
  AssertThrow(file,ExcMessage("Incomplete frozen Bernstein coefficients."));
  const auto &sets=this->introspection().index_sets;
  LinearAlgebra::BlockVector owned(sets.system_partitioning,this->get_mpi_communicator());
  owned=sim().solution;
  const auto &fe=this->get_fe();
  const auto pc=this->introspection().phase_field().first_component_index;
  std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
  for(const auto &cell:this->get_dof_handler().active_cell_iterators())if(cell->is_locally_owned())
    {
      const unsigned int ix=std::lround(cell->center()[0]*nx/8.-.5);
      const unsigned int iy=std::lround(cell->center()[1]*ny/4.-.5);
      cell->get_dof_indices(ids);
      for(unsigned int j=0;j<9;++j)
        owned[ids[fe.component_to_system_index(pc,j)]]=coefficients.at(iy*nx+ix)[j];
    }
  owned.compress(VectorOperation::insert);
  sim().solution=owned;
}

void frozen_wavelength_response()
{
  const auto directory=std::string(std::getenv("ASPECT_WAVELENGTH_INPUT"));
  auto &state=*model().bulk_cohesive_state;
  AssertThrow(!state.section_geometry->cut,ExcMessage("This prepared horizontal specimen requires its original Cartesian section quadrature."));
  std::ifstream file(directory+"/sections.txt");
  unsigned int ns;file>>ns;
  AssertDimension(ns,state.sections.size());
  for(auto &section:state.sections)
    {
      double coordinate,angle;unsigned int formed;
      file>>coordinate>>angle>>formed;
      AssertThrow(std::abs(coordinate-section.coordinate)<1e-12,ExcMessage("Frozen section coordinate mismatch."));
      section.angle=angle;section.formed=formed;
      section.weight/=std::cos(angle);
    }
  AssertThrow(file,ExcMessage("Incomplete frozen section planes."));
  for(auto &entry:state.normal_weights)for(unsigned int q=0;q<entry.second.size();++q)
    entry.second[q]*=std::cos(state.sections[state.section_ids.at(entry.first)[q]].angle);
  // Reuse exactly the native coupling and ordinary Stokes preconditioner.
  // Fixed-phase stress/strain derivatives are independent of parent tau and A.
  sim().bulk_tip_coupling=std::make_shared<internal::BulkTipCoupling<dim>>(sim(),model(),sim().current_constraints);
  auto &op=*sim().bulk_tip_coupling;
  if (this->get_timestep_number()>0) op.prepare_direction();
  op.accepted_coordinates=op.coordinates(sim().solution);
  op.accepted_evaluation=op.evaluate(op.accepted_coordinates,false);
  const AffineConstraints<double> saved(sim().current_constraints);
  const bool newton=sim().assemble_newton_stokes_system,matrix=sim().assemble_newton_stokes_matrix;
  for(const auto &line:saved.get_lines())sim().current_constraints.set_inhomogeneity(line.index,0.);
  sim().current_constraints.close();
  sim().assemble_newton_stokes_system=false;sim().assemble_newton_stokes_matrix=false;
  sim().set_assemblers();
  sim().pressure_scaling=sim().compute_pressure_scaling_factor();
  sim().rebuild_stokes_matrix=true;sim().rebuild_stokes_preconditioner=true;
  sim().assemble_stokes_system();sim().build_stokes_preconditioner();
  measure_restoring_response();
  sim().current_constraints.copy_from(saved);
  sim().assemble_newton_stokes_system=newton;sim().assemble_newton_stokes_matrix=matrix;
  sim().bulk_tip_coupling.reset();sim().set_assemblers();
}
