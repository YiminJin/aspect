// Read-only accepted-state observations for the bounded phase-release pair.
// No probe is published and no extra physical interval is evaluated.
std::unique_ptr<LinearAlgebra::BlockVector> release_initial_field;
LinearAlgebra::Vector release_initial_fracture_gradient;
double release_initial_fracture=0.;

static unsigned int release_region(const Point<dim> &p)
{
  if (std::abs(p[0]-4.)>3. || std::abs(p[1]-2.)>1.)return 3; // remote
  if (std::abs(p[0]-4.)<1.5 && std::abs(p[1]-2.)<.25)return 0; // central
  if (std::abs(p[0]-4.)<2.3125 && std::abs(p[1]-2.)<.25)return 1; // old taper/tail
  return 2; // initially intact surroundings, including ahead of both tips
}

static double release_probe(const Point<dim> &p,const unsigned int probe)
{
  const double left=probe==0 ? 6.03125 : 6.3125;
  const double tx=std::max(0.,1.-std::abs(p[0]-(left+.0625))/.0625);
  const double ty=std::max(0.,1.-std::abs(p[1]-2.)/.125);
  return tx*ty;
}

void observe_phase_release()
{
  using Field=LinearAlgebra::BlockVector;
  const auto &s=*model().bulk_cohesive_state;
  const auto comm=this->get_mpi_communicator();
  const auto rank=Utilities::MPI::this_mpi_process(comm);
  const auto &fe=this->get_fe();
  const auto pc=this->introspection().phase_field().first_component_index;
  const auto pb=this->introspection().phase_field().block_index;
  const auto offset=this->get_solution().get_block_indices().block_start(pb);
  LinearAlgebra::Vector endpoint_gradient;
  const double fracture=this->get_phase_field_handler().assemble_dg_fracture(this->get_solution(),endpoint_gradient);
  if(s.step==0)
    {
      release_initial_field=std::make_unique<Field>(this->get_solution());
      release_initial_fracture_gradient=endpoint_gradient;
      release_initial_fracture=fracture;
    }
  AssertThrow(release_initial_field,ExcMessage("Release observer needs its fresh initial reference; restart observation is not enabled."));
  auto *op=sim().bulk_tip_coupling.get();
  Field stage=this->get_solution();
  LinearAlgebra::Vector stage_gradient(endpoint_gradient);
  if(s.step)
    {
      AssertThrow(op && op->accepted_evaluation,ExcMessage("Missing accepted stage for release observation."));
      stage=op->ghost(op->accepted_coordinates); // phase is b_m, not extrapolated b_1
      op->fracture_matrix.vmult(stage_gradient,op->accepted_coordinates.block(pb));
      stage_gradient+=op->fracture_linear;
    }
  std::array<LinearAlgebra::Vector,2> direction;
  for(auto &v:direction){v.reinit(endpoint_gradient);v=0.;}
  std::array<double,4> max_change{},phase_integral{},regional_work{};
  std::array<double,2> drive{},measure{},max_drive{},epsilon_bound{{1.,1.}};
  std::array<std::array<double,3>,2> reduced_difference{};
  const std::array<double,3> eps{{1e-4,1e-5,1e-6}};
  double minimum_change=0.;
  const double G=s.parameters.G,m=s.parameters.m;
  auto H=[&](double phi){const double c=m*phi*(1.+phi);return c/(c+(1.-phi)*(1.-phi));};
  FEValues<dim> values(this->get_mapping(),fe,model().mechanical_quadrature(),
                       update_values|update_quadrature_points|update_JxW_values);
  std::vector<double> initial(values.n_quadrature_points),end(values.n_quadrature_points),mid(values.n_quadrature_points);
  std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
  std::ofstream coefficients(this->get_output_directory()+"release-phase-"+Utilities::int_to_string(s.step,4)+"-r"+std::to_string(rank)+".csv");
  coefficients<<"cell,x,y,B0,B1,B2,B3,B4,B5,B6,B7,B8\n"<<std::setprecision(17);
  for(const auto &entry:s.native_cells)
    {
      const auto cell=entry.second;cell->get_dof_indices(ids);values.reinit(cell);
      values[FEValuesExtractors::Scalar(pc)].get_function_values(*release_initial_field,initial);
      values[FEValuesExtractors::Scalar(pc)].get_function_values(this->get_solution(),end);
      values[FEValuesExtractors::Scalar(pc)].get_function_values(stage,mid);
      std::array<std::array<double,9>,2> chi{};
      coefficients<<entry.first<<','<<cell->center()[0]<<','<<cell->center()[1];
      for(unsigned int j=0;j<9;++j)
        {
          const auto i=ids[fe.component_to_system_index(pc,j)]-offset;
          const double B=this->get_solution().block(pb)[i],old=release_initial_field->block(pb)[i];
          coefficients<<','<<B;
          regional_work[release_region(cell->center())]+=.5*(B-old)*(endpoint_gradient[i]+release_initial_fracture_gradient[i]);
          Point<dim> unit;unit[0]=.5*(j%3);unit[1]=.5*(j/3);
          const auto p=this->get_mapping().transform_unit_to_real_cell(cell,unit);
          for(unsigned int probe=0;probe<2;++probe)
            {
              chi[probe][j]=release_probe(p,probe);direction[probe][i]=chi[probe][j];
              if(chi[probe][j]>0.)
                {
                  const double upper=s.step ? op->upper_bound()[i] : 1.;
                  epsilon_bound[probe]=std::min(epsilon_bound[probe],(upper-stage.block(pb)[i])/chi[probe][j]);
                }
            }
        }
      coefficients<<'\n';
      for(unsigned int q=0;q<values.n_quadrature_points;++q)
        {
          const auto p=values.quadrature_point(q);const auto region=release_region(p);
          const double change=end[q]-initial[q];
          max_change[region]=std::max(max_change[region],change);
          phase_integral[region]+=values.JxW(q)*change;
          minimum_change=std::min(minimum_change,change);
          double Y=0.,F=0.;
          if(s.step)
            {const auto &r=op->accepted_evaluation->points.at(entry.first)[q];Y=r.driving;F=r.force;}
          else
            {
              const auto &h=s.history.at(entry.first)[q];
              const auto &section=s.sections[s.section_ids.at(entry.first)[q]];
              MaterialModel::BulkCohesive::Tensor S;
              S[0][0]=-.5*std::sin(2.*section.angle);S[1][1]=-S[0][0];S[0][1]=.5*std::cos(2.*section.angle);
              F=(h.stress+2.*G*h.A)*S-section.force;
              const double phi=mid[q],c=m*phi*(1.+phi),den=c+(1.-phi)*(1.-phi);
              Y=m*(1.+3.*phi)*(1.-phi)/(den*den)*F*F/(2.*G);
            }
          for(unsigned int probe=0;probe<2;++probe)
            {
              double shape=0.;
              for(unsigned int j=0;j<9;++j)shape+=chi[probe][j]*values.shape_value_component(fe.component_to_system_index(pc,j),q,pc);
              drive[probe]+=values.JxW(q)*shape*Y;measure[probe]+=values.JxW(q)*shape;
              if(shape>0.)max_drive[probe]=std::max(max_drive[probe],Y);
              if(s.step==0)for(unsigned int k=0;k<eps.size();++k)
                reduced_difference[probe][k]-=values.JxW(q)*(H(mid[q]+eps[k]*shape)-H(mid[q]))*F*F/(2.*G);
            }
        }
    }
  for(auto &v:direction)v.compress(VectorOperation::insert);
  for(unsigned int probe=0;probe<2;++probe)
    {
      drive[probe]=Utilities::MPI::sum(drive[probe],comm);
      measure[probe]=Utilities::MPI::sum(measure[probe],comm);
      max_drive[probe]=Utilities::MPI::max(max_drive[probe],comm);
      epsilon_bound[probe]=Utilities::MPI::min(epsilon_bound[probe],comm);
      const double fracture_derivative=stage_gradient*direction[probe];
      gate("positive ahead-tip probe measure",measure[probe]>0.);
      if(s.step==0)
        {
          gate("ahead-tip finite differences admissible",epsilon_bound[probe]>=eps[0]);
          for(unsigned int k=0;k<eps.size();++k)
            {
              Field trial(this->introspection().index_sets.system_partitioning,comm);trial=this->get_solution();
              trial.block(pb).add(eps[k],direction[probe]);
              Field readable=this->get_solution();readable=trial;
              LinearAlgebra::Vector scratch;
              const double energy=this->get_phase_field_handler().assemble_dg_fracture(readable,scratch);
              const double local_change=Utilities::MPI::sum(reduced_difference[probe][k],comm);
              const double fd=(energy-fracture+local_change)/eps[k];
              const double expected=fracture_derivative-drive[probe];
              const double error=std::abs(fd-expected)/std::max(measure[probe],std::abs(expected));
              if(k==eps.size()-1)gate("ahead-tip total directional derivative",error<2e-4);
              if(rank==0)
                {
                  std::ofstream f(this->get_output_directory()+"release-probe-fd.csv",std::ios::app);
                  if(f.tellp()==0)f<<"probe,epsilon,finite_difference,analytic_derivative,scaled_error\n";
                  f<<std::setprecision(17)<<probe<<','<<eps[k]<<','<<fd<<','<<expected<<','<<error<<'\n';
                }
            }
        }
      if(rank==0)
        {
          std::ofstream f(this->get_output_directory()+"release-driving.csv",std::ios::app);
          if(f.tellp()==0)f<<"step,endpoint_time,stage_time,probe,measure,mechanical_driving,fracture_derivative,net_driving,mean_net_driving,max_local_Y,admissible_stage_epsilon\n";
          f<<std::setprecision(17)<<s.step<<','<<s.time<<','<<(s.step?s.time-.5*this->get_timestep():0.)<<','<<probe<<','<<measure[probe]<<','<<drive[probe]<<','<<fracture_derivative<<','<<drive[probe]-fracture_derivative<<','<<(drive[probe]-fracture_derivative)/measure[probe]<<','<<max_drive[probe]<<','<<epsilon_bound[probe]<<'\n';
        }
    }
  double partition=0.;
  for(unsigned int region=0;region<4;++region)
    {
      max_change[region]=Utilities::MPI::max(max_change[region],comm);
      phase_integral[region]=Utilities::MPI::sum(phase_integral[region],comm);
      regional_work[region]=Utilities::MPI::sum(regional_work[region],comm);partition+=regional_work[region];
      if(rank==0)
        {
          std::ofstream f(this->get_output_directory()+"release-regions.csv",std::ios::app);
          if(f.tellp()==0)f<<"step,time,region,max_phase_increment,integrated_phase_increment,added_fracture_row_partition\n";
          f<<std::setprecision(17)<<s.step<<','<<s.time<<','<<region<<','<<max_change[region]<<','<<phase_integral[region]<<','<<regional_work[region]<<'\n';
        }
    }
  gate("release exact SIPG fracture partition",std::abs(partition-(fracture-release_initial_fracture))<1e-10*std::max(1.,std::abs(fracture)));
  gate("release material phase irreversibility",Utilities::MPI::min(minimum_change,comm)>=-1e-10);
}
