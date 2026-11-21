// Independent evidence for the common prepared physical state. These routines
// neither publish history nor alter the simulator field, including on restart.
LinearAlgebra::BlockVector prepared_phase_reference() const
{
  const auto &sets=this->introspection().index_sets;
  LinearAlgebra::BlockVector owned(sets.system_partitioning,this->get_mpi_communicator());
  owned=this->get_solution();
  const auto &fe=this->get_fe();const auto pc=this->introspection().phase_field().first_component_index;
  std::ifstream file(std::string(std::getenv("ASPECT_WAVELENGTH_INPUT"))+"/phase.txt");
  unsigned int nx,ny;file>>nx>>ny;
  std::vector<std::array<double,9>> coefficients(nx*ny);
  for(auto &row:coefficients)for(auto &v:row)file>>v;
  AssertThrow(file,ExcMessage("Prepared reference phase is incomplete."));
  std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
  for(const auto &cell:this->get_dof_handler().active_cell_iterators())if(cell->is_locally_owned())
    {
      const unsigned int ix=std::lround(cell->center()[0]*nx/8.-.5),iy=std::lround(cell->center()[1]*ny/4.-.5);
      cell->get_dof_indices(ids);
      for(unsigned int j=0;j<9;++j)owned[ids[fe.component_to_system_index(pc,j)]]=coefficients.at(iy*nx+ix)[j];
    }
  owned.compress(VectorOperation::insert);
  LinearAlgebra::BlockVector readable(sets.system_partitioning,sets.system_relevant_partitioning,this->get_mpi_communicator());
  readable=owned;return readable;
}

void export_prepared_history() const
{
  const auto &state=*model().bulk_cohesive_state;
  AssertThrow(!state.section_geometry->cut,ExcMessage("This prepared horizontal specimen requires its original Cartesian section quadrature."));
  const auto suffix="-r"+std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()));
  std::ofstream map(this->get_output_directory()+"prepared-history"+suffix+".csv");
  std::ofstream binary(this->get_output_directory()+"prepared-history"+suffix+".bin",std::ios::binary);
  map<<"cell,nq\n";
  for(const auto &entry:state.history)
    {
      map<<entry.first<<','<<entry.second.size()<<'\n';
      for(const auto &h:entry.second)
        {
          std::array<double,12> row;unsigned int k=0;
          for(const auto *tensor:{&h.stress,&h.A})
            for(unsigned int i=0;i<3;++i)for(unsigned int j=i;j<3;++j)row[k++]=(*tensor)[i][j];
          binary.write(reinterpret_cast<const char *>(row.data()),sizeof(row));
        }
    }
  AssertThrow(map && binary,ExcMessage("Prepared history evidence could not be written."));
}
