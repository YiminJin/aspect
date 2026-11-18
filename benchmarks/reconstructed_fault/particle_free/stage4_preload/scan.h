// Read-only first variations for every nonnegative native Bernstein direction.
// The artificial fixed-phase constraint is not a physical upper obstacle.
void observe_preload_scan()
{
  const auto &s=*model().bulk_cohesive_state;
  const auto comm=this->get_mpi_communicator();
  const auto rank=Utilities::MPI::this_mpi_process(comm);
  const auto &fe=this->get_fe();
  const auto pc=this->introspection().phase_field().first_component_index;
  const auto pb=this->introspection().phase_field().block_index;
  const auto offset=this->get_solution().get_block_indices().block_start(pb);
  LinearAlgebra::Vector gradient;
  this->get_phase_field_handler().assemble_dg_fracture(this->get_solution(),gradient);
  auto *op=sim().bulk_tip_coupling.get();
  LinearAlgebra::BlockVector field=this->get_solution();
  if(s.step)
    {
      AssertThrow(op && op->accepted_evaluation,ExcMessage("Scan requires the accepted stage."));
      field=op->ghost(op->accepted_coordinates);
      op->fracture_matrix.vmult(gradient,op->accepted_coordinates.block(pb));
      gradient+=op->fracture_linear;
    }
  const double G=s.parameters.G,m=s.parameters.m;
  FEValues<dim> values(this->get_mapping(),fe,model().mechanical_quadrature(),
                       update_values|update_JxW_values);
  std::vector<double> phase(values.n_quadrature_points);
  std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
  // At t=0, completely virgin sections have phi=A=V=f=0. The native
  // orientation objective reduces exactly to the principal eigenvector of
  // integral (u sin(2a)+v cos(2a))^2. This reads, but does not publish, its
  // maximizer. Formed sections retain their stored planes. Later stages use
  // the actual native selected plane, not this diagnostic construction.
  std::vector<double> moments(3*s.sections.size(),0.),angles(s.sections.size());
  if(!s.step)
    {
      for(const auto &entry:s.native_cells)
        {
          values.reinit(entry.second);
          values[FEValuesExtractors::Scalar(pc)].get_function_values(field,phase);
          for(unsigned int q=0;q<values.n_quadrature_points;++q)
            {
              const auto k=s.section_ids.at(entry.first)[q];
              if(s.sections[k].formed)continue;
              const auto &h=s.history.at(entry.first)[q];
              AssertThrow(phase[q]==0. && h.A.norm()==0. && s.sections[k].force==0.,
                          ExcMessage("Analytic virgin diagnostic requires exactly intact, zero-resistance sections."));
              const double u=-.5*(h.stress[0][0]-h.stress[1][1]),v=h.stress[0][1],w=values.JxW(q);
              moments[3*k]+=w*u*u;moments[3*k+1]+=w*v*v;moments[3*k+2]+=w*u*v;
            }
        }
      Utilities::MPI::sum(moments,comm,moments);
      for(unsigned int k=0;k<angles.size();++k)
        {
          angles[k]=s.sections[k].angle;
          if(!s.sections[k].formed && moments[3*k]+moments[3*k+1]>0.)
            angles[k]=.25*std::atan2(2.*moments[3*k+2],moments[3*k+1]-moments[3*k]);
        }
    }
  std::ofstream out(this->get_output_directory()+"preload-scan-"+Utilities::int_to_string(s.step,4)+"-r"+std::to_string(rank)+".csv");
  out<<"x,y,j,B,mass,fracture,drive,selected_drive,envelope_drive\n"<<std::setprecision(17);
  for(const auto &entry:s.native_cells)
    {
      const auto cell=entry.second;values.reinit(cell);cell->get_dof_indices(ids);
      values[FEValuesExtractors::Scalar(pc)].get_function_values(field,phase);
      std::array<double,9> mass{},drive{},selected{},envelope{};
      for(unsigned int q=0;q<values.n_quadrature_points;++q)
        {
          double Y=0.,Ys=0.,Ye=0.;
          if(s.step)
            Y=Ys=Ye=op->accepted_evaluation->points.at(entry.first)[q].driving;
          else
            {
              const auto k=s.section_ids.at(entry.first)[q];
              const auto &h=s.history.at(entry.first)[q];
              const auto T=h.stress+2.*G*h.A;
              const double u=-.5*(T[0][0]-T[1][1]),v=T[0][1],f=s.sections[k].force;
              const double F=u*std::sin(2.*s.sections[k].angle)+v*std::cos(2.*s.sections[k].angle)-f;
              const double Fs=u*std::sin(2.*angles[k])+v*std::cos(2.*angles[k])-f;
              const double phi=phase[q],c=m*phi*(1.+phi),den=c+(1.-phi)*(1.-phi);
              const double derivative=m*(1.+3.*phi)*(1.-phi)/(den*den)/(2.*G);
              Y=derivative*F*F;Ys=derivative*Fs*Fs;
              Ye=s.sections[k].formed ? Y : derivative*(u*u+v*v);
            }
          for(unsigned int j=0;j<9;++j)
            {
              const double w=values.JxW(q)*values.shape_value_component(fe.component_to_system_index(pc,j),q,pc);
              mass[j]+=w;drive[j]+=w*Y;selected[j]+=w*Ys;envelope[j]+=w*Ye;
            }
        }
      for(unsigned int j=0;j<9;++j)
        {
          const auto i=ids[fe.component_to_system_index(pc,j)]-offset;
          out<<cell->center()[0]<<','<<cell->center()[1]<<','<<j<<','<<field.block(pb)[i]<<','<<mass[j]<<','<<gradient[i]<<','<<drive[j]<<','<<selected[j]<<','<<envelope[j]<<'\n';
        }
    }
}
