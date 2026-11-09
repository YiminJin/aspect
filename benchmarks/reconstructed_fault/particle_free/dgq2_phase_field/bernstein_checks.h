/* Basis-equivalence checks using the production spatial operators. */
void bernstein_equivalence(dealii::parallel::distributed::Triangulation<2> &mesh,
                          const dealii::Mapping<2> &mapping)
{
  using namespace aspect;
  FE_DGQ<2> nodal(2);FE_DGQBernstein<2> bernstein(2);
  DoFHandler<2> dn(mesh),db(mesh);dn.distribute_dofs(nodal);db.distribute_dofs(bernstein);
  const auto on=dn.locally_owned_dofs(),ob=db.locally_owned_dofs();
  const auto rn=DoFTools::extract_locally_relevant_dofs(dn),rb=DoFTools::extract_locally_relevant_dofs(db);
  PhaseField::DGOperator<2> n(mapping,dn,0,0,on,rn,MPI_COMM_WORLD),b(mapping,db,0,0,ob,rb,MPI_COMM_WORLD);
  using V=aspect::LinearAlgebra::Vector;
  FullMatrix<double> E(9,9),inverse(9,9);nodal.get_interpolation_matrix(bernstein,E);bernstein.get_interpolation_matrix(nodal,inverse);
  auto transform=[&](const V &input,V &output,const bool to_native,const bool dual)
  {
    auto cn=dn.begin_active();auto cb=db.begin_active();std::vector<types::global_dof_index> ni(9),bi(9);
    for(;cn!=dn.end();++cn,++cb)if(cn->is_locally_owned())
      {
        cn->get_dof_indices(ni);cb->get_dof_indices(bi);Vector<double> in(9),out(9);
        for(unsigned int j=0;j<9;++j)in[j]=input[(to_native||dual)?ni[j]:bi[j]];
        if(dual)E.Tvmult(out,in);else (to_native?inverse:E).vmult(out,in);
        for(unsigned int j=0;j<9;++j)output[(to_native||dual)?bi[j]:ni[j]]=out[j];
      }
    output.compress(VectorOperation::insert);
  };
  V bn(ob,MPI_COMM_WORLD),an(on,MPI_COMM_WORLD),ab(ob,MPI_COMM_WORLD),qb(ob,MPI_COMM_WORLD),qn(on,MPI_COMM_WORLD),expected(ob,MPI_COMM_WORLD);
  for(const auto &cell:db.active_cell_iterators())if(cell->is_locally_owned())
    {std::vector<types::global_dof_index> ids(9);cell->get_dof_indices(ids);
     for(unsigned int j=0;j<9;++j)bn[ids[j]]=.3+.2*std::sin(cell->center()[0]+.7*cell->center()[1]+j);}
  bn.compress(VectorOperation::insert);transform(bn,an,false,false);
  double max_action=0.,max_energy=0.,max_transport=0.,max_balance=0.,max_mean=0.;
  for(bool boundary:{false,true})
    {
      aspect::LinearAlgebra::SparseMatrix kn,kb;n.reinit_matrix(kn);b.reinit_matrix(kb);
      V fn(on,MPI_COMM_WORLD),fb(ob,MPI_COMM_WORLD);
      const PhaseField::DGOperator<2>::Coefficient value=boundary ? PhaseField::DGOperator<2>::Coefficient([](const Point<2> &p){return .4+.01*p[0];}):PhaseField::DGOperator<2>::Coefficient();
      auto coefficient=[](const Point<2> &p){return 1.+.05*p[0];};
      const double cn=n.assemble_sipg(kn,fn,coefficient,10.,value),cb=b.assemble_sipg(kb,fb,coefficient,10.,value);
      kn.vmult(qn,an);kb.vmult(qb,bn);
      max_energy=std::max(max_energy,std::abs(.5*(qn*an)-an*fn+cn-(.5*(qb*bn)-bn*fb+cb)));
      qn-=fn;qb-=fb;transform(qn,expected,true,true);expected-=qb;
      max_action=std::max(max_action,expected.linfty_norm()/std::max(1.,qb.linfty_norm()));
    }
  auto velocity=[](const auto &,const auto &points,auto &speed,auto &divergence)
  {for(unsigned int q=0;q<points.size();++q){speed[q][0]=.4+.1*points[q][0];speed[q][1]=.05*points[q][1];divergence[q]=.15;}};
  n.assemble_transport(velocity);b.assemble_transport(velocity);
  AssertThrow(std::abs(n.maximum_mean_rate()-b.maximum_mean_rate())<1e-13,ExcMessage("Basis changed transport CFL."));
  // With stationary inflow and one unlimited SSPRK3 step the update is a
  // cubic polynomial in dt. Its derivative at zero, recovered exactly by this
  // cubic formula, is the actual production semidiscrete action L*u+g. This
  // tests L_B=E^{-1}L_N*E without exposing a private operator just for tests.
  const double h = std::min(.002, .1/std::max(1., n.maximum_mean_rate()));
  V derivative_n(an), derivative_b(bn);
  derivative_n *= -11.; derivative_b *= -11.;
  const double weights[] = {18., -9., 2.};
  for (unsigned int j=0; j<3; ++j)
    {
      V next_n(on,MPI_COMM_WORLD),next_b(ob,MPI_COMM_WORLD);
      auto inflow=[](const Point<2> &p,double){return .4+.01*p[0];};
      const auto sn=n.transport(an,next_n,inflow,0.,(j+1)*h,0.,1.,false);
      const auto sb=b.transport(bn,next_b,inflow,0.,(j+1)*h,0.,1.,false);
      AssertThrow(sn.substeps==1 && sb.substeps==1,ExcMessage("Generator oracle requires one SSPRK3 step."));
      derivative_n.add(weights[j],next_n);derivative_b.add(weights[j],next_b);
    }
  derivative_n /= 6.*h; derivative_b /= 6.*h;
  transform(derivative_n,expected,true,false);expected-=derivative_b;
  const double generator_error=expected.linfty_norm()/std::max(1.,derivative_b.linfty_norm());
  AssertThrow(generator_error<2e-10,ExcMessage("Semidiscrete transport basis identity failed."));
  unsigned long long limited=0;
  for(bool limiter:{false,true})
    {
      if(limiter)
        {
          // Same interpolant with an actual interior overshoot. Its mean is
          // admissible; subdivision and scaling must act on the polynomial.
          for(const auto &cell:dn.active_cell_iterators())if(cell->is_locally_owned())
            {std::vector<types::global_dof_index> ids(9);cell->get_dof_indices(ids);
             for(unsigned int j=0;j<9;++j){const auto x=nodal.get_unit_support_points()[j];an[ids[j]]=1.02-.5*std::pow(x[0]-.5,2);}}
          an.compress(VectorOperation::insert);transform(an,bn,true,false);
        }
      V next_n(on,MPI_COMM_WORLD),next_b(ob,MPI_COMM_WORLD);
      auto inflow=[](const Point<2> &p,double t){return .4+.01*p[0]+.02*t;};
      const auto sn=n.transport(an,next_n,inflow,0.,.02,0.,1.,limiter);
      const auto sb=b.transport(bn,next_b,inflow,0.,.02,0.,1.,limiter);
      transform(next_n,expected,true,false);expected-=next_b;
      max_transport=std::max(max_transport,expected.linfty_norm());
      max_balance=std::max({max_balance,std::abs(sn.integrated_source-sb.integrated_source),std::abs(sn.integrated_boundary_flux-sb.integrated_boundary_flux),std::abs(sb.balance_error)});
      max_mean=std::max(max_mean,std::abs(sn.final_integral-sb.final_integral));
      if(limiter)limited=sb.limited_cells;
    }
  AssertThrow(max_action<2e-12 && max_energy<2e-10 && max_transport<2e-12 && max_balance<2e-12 && max_mean<2e-12 && limited>0,
              ExcMessage("Native Bernstein operator/limited transport equivalence failed."));
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    std::cout<<std::setprecision(17)<<"SIPG_action,SIPG_energy,transport_coefficients,flux_balance,integral,limited_cells,generator_relative\n"
             <<max_action<<','<<max_energy<<','<<max_transport<<','<<max_balance<<','<<max_mean<<','<<limited<<','<<generator_error<<'\n';
}
