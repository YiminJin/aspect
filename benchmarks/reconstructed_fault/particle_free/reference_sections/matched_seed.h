// Benchmark hook before the native limiter and zero-clock initialization.
// Rigidly rotating a compact seed analytically can perturb a support-node
// zero. Transport the same scalar FE coefficients by cell/local shape to
// compare the same discrete specimen. Both runs use the same partition.
void match_reference_seed() const
{
  const char *source=std::getenv("ASPECT_SECTION_SEED_FROM");
  if (!source && !std::getenv("ASPECT_SECTION_SAVE_SEED")) return;
  const auto suffix="/reference-seed-r"+std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".txt";
  const auto &fe=this->get_fe();
  const auto pc=this->introspection().phase_field().first_component_index;
  const auto pb=this->introspection().phase_field().block_index;
  const auto offset=sim().solution.get_block_indices().block_start(pb);
  const auto count=fe.base_element(fe.component_to_base_index(pc).first).n_dofs_per_cell();
  std::ifstream input;
  std::ofstream output;
  if (source) input.open(std::string(source)+suffix);
  else output.open(this->get_output_directory()+suffix);
  output<<std::setprecision(17);
  LinearAlgebra::Vector owned(this->introspection().index_sets.system_partitioning[pb],this->get_mpi_communicator());
  std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
  for (const auto &cell:this->get_dof_handler().active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell->get_dof_indices(indices);
        if (source)
          {
            std::string id;
            input>>id;
            AssertThrow(id==cell->id().to_string(),ExcMessage("Reference seed cell identity mismatch."));
          }
        else output<<cell->id();
        for (unsigned int j=0;j<count;++j)
          {
            const auto i=indices[fe.component_to_system_index(pc,j)]-offset;
            if (source)
              {
                double value;
                AssertThrow(input>>value,ExcMessage("Incomplete matched phase seed."));
                owned[i]=value;
              }
            else output<<' '<<sim().solution.block(pb)[i];
          }
        if (!source) output<<'\n';
      }
  if (source)
    {
      owned.compress(VectorOperation::insert);
      for (auto *v:{&sim().solution,&sim().old_solution,&sim().old_old_solution}) v->block(pb)=owned;
    }
}
