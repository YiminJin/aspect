/* Manufactured constrained phase problems using the native sparse solver. */
#include <aspect/phase_field/obstacle.h>
#include <aspect/reconstructed_fault/endpoint_phase_constraints.h>
#include <aspect/reconstructed_fault/endpoint_mpi.h>

void obstacle_check(aspect::PhaseField::DGOperator<2> &op,
                    const dealii::Mapping<2> &mapping,const dealii::DoFHandler<2> &dofs,
                    const dealii::IndexSet &owned,const dealii::IndexSet &relevant,
                    const std::string &mode,const unsigned int level)
{
  using namespace aspect;
  using VectorType=aspect::LinearAlgebra::Vector;
  const auto comm=MPI_COMM_WORLD;
  const auto &element=dofs.get_fe();
  const double a=numbers::PI/2.;
  const auto lower_function=[&](const Point<2> &x){return .2+.03*std::cos(a*x[0]);};
  const auto growth=[](const Point<2> &x){return .05*std::pow(std::max(0.,1.-x.norm_square()),4);};
  const auto exact=[&](const Point<2>&x){return lower_function(x)+(mode=="no-growth" ? 0.:growth(x));};
  aspect::LinearAlgebra::SparseMatrix physical;op.reinit_matrix(physical);
  VectorType rhs(owned,comm),lower(owned,comm),state(owned,comm),upper(owned,comm);
  op.assemble_sipg(physical,rhs,[](const Point<2>&){return 1.;},10.,lower_function);
  const QGauss<2> rule(8);
  FEValues<2> fe(mapping,element,rule,update_values|update_JxW_values|update_quadrature_points);
  std::vector<types::global_dof_index> ids(element.n_dofs_per_cell());
  for (const auto &cell:dofs.active_cell_iterators()) if (cell->is_locally_owned())
    {
      cell->get_dof_indices(ids);fe.reinit(cell);
      for (unsigned int i=0;i<ids.size();++i)
        lower[ids[i]]=lower_function(mapping.transform_unit_to_real_cell(cell,element.get_unit_support_points()[i]));
      for (unsigned int q=0;q<rule.size();++q)
        {
          const auto x=fe.quadrature_point(q);const double r2=x.norm_square(),s=std::max(0.,1.-r2);
          const double lap=-.03*a*a*std::cos(a*x[0])+.05*(-16*s*s*s+48*r2*s*s);
          const double f=-lap+.7*exact(x)-.05*std::max(0.,r2-1.);
          for (unsigned int i=0;i<ids.size();++i)
            {
              rhs[ids[i]]+=fe.JxW(q)*f*fe.shape_value(i,q);
              for (unsigned int j=0;j<ids.size();++j)
                physical.add(ids[i],ids[j],.7*fe.JxW(q)*fe.shape_value(i,q)*fe.shape_value(j,q));
            }
        }
    }
  physical.compress(VectorOperation::add);rhs.compress(VectorOperation::insert);lower.compress(VectorOperation::insert);
  if (mode=="moving-obstacle")
    {
      op.assemble_transport([](const auto &,const auto &points,auto &speed,auto &divergence)
      { for (unsigned int q=0;q<points.size();++q){speed[q][0]=.3;speed[q][1]=0.;divergence[q]=0.;} });
      const auto stats=op.transport(lower,state,[&](const Point<2>&x,double t){auto y=x;y[0]-=.3*t;return lower_function(y);},0.,.2,0.,1.,true);
      AssertThrow(std::abs(stats.balance_error)<1e-12,ExcMessage("Moving obstacle transport balance failed."));
      lower=state;
    }
  AffineConstraints<double> constraints;constraints.close();
  aspect::internal::EndpointPhaseConstraints coordinates(constraints,0,dofs.n_dofs());
  coordinates.set_discontinuous_q2(dofs,0,0,comm);
  const auto lower_nodal=aspect::internal::endpoint_values(lower,comm);
  const auto lower_coefficients=coordinates.select(lower_nodal);
  aspect::LinearAlgebra::SparseMatrix matrix;op.reinit_matrix(matrix);
  for (const auto i:owned)
    for (auto entry=physical.begin(i);entry!=physical.end(i);++entry)
      for (const auto &row:coordinates.weights()[i])
        for (const auto &column:coordinates.weights()[entry->column()])
          matrix.add(row.first,column.first,row.second*entry->value()*column.second);
  matrix.compress(VectorOperation::add);
  const auto dual=coordinates.transpose(aspect::internal::endpoint_values(rhs,comm));
  aspect::internal::endpoint_assign(rhs,dual);aspect::internal::endpoint_assign(lower,lower_coefficients);
  upper=.8;state=lower;
  VectorType expected(owned,comm),reaction(owned,comm);
  expected=lower;
  if (mode=="no-growth" || mode=="moving-obstacle")
    {
      for (const auto &cell:dofs.active_cell_iterators()) if (cell->is_locally_owned())
        {
          cell->get_dof_indices(ids);
          // Tensor Bernstein coordinate locations use lexicographic ordering.
          for (unsigned int i=0;i<ids.size();++i)
            {
              Point<2> unit(.5*(i%3),.5*(i/3));
              const auto x=mapping.transform_unit_to_real_cell(cell,unit);
              const double increment=mode=="moving-obstacle" ? growth(x):0.;
              expected[ids[i]]+=increment;
              reaction[ids[i]]=increment==0. ? .02*cell->measure()/9.:0.;
            }
        }
      expected.compress(VectorOperation::insert);reaction.compress(VectorOperation::insert);
      matrix.vmult(rhs,expected);rhs-=reaction;
    }
  const auto result=PhaseField::solve_obstacle(matrix,rhs,lower,upper,state,relevant,comm,1e-10,100,2000);
  const auto coefficients=aspect::internal::endpoint_values(state,comm);
  const auto nodal=coordinates.expand(coefficients);
  VectorType nodal_owned(owned,comm),ghost(owned,relevant,comm);
  aspect::internal::endpoint_assign(nodal_owned,nodal);ghost=nodal_owned;
  double error=0.,minimum_gap=1.,maximum=0.;unsigned int active=0,free=0;
  const auto exact_discrete=coordinates.expand(aspect::internal::endpoint_values(expected,comm));
  if (mode=="no-growth" || mode=="moving-obstacle")
    {auto diff=nodal;diff-=exact_discrete;error=diff.linfty_norm();}
  else
    {
      std::vector<double> values(rule.size());
      for (const auto &cell:dofs.active_cell_iterators()) if (cell->is_locally_owned())
        {
          fe.reinit(cell);fe.get_function_values(ghost,values);
          for (unsigned int q=0;q<rule.size();++q) error+=fe.JxW(q)*std::pow(values[q]-exact(fe.quadrature_point(q)),2);
        }
      error=std::sqrt(Utilities::MPI::sum(error,comm));
    }
  for (const auto i:owned)
    { minimum_gap=std::min(minimum_gap,state[i]-lower[i]);maximum=std::max(maximum,static_cast<double>(state[i]));
      if (std::abs(state[i]-lower[i])<1e-9) ++active;else ++free; }
  minimum_gap=Utilities::MPI::min(minimum_gap,comm);maximum=Utilities::MPI::max(maximum,comm);
  active=Utilities::MPI::sum(active,comm);free=Utilities::MPI::sum(free,comm);
  VectorType lower_physical(owned,comm);aspect::internal::endpoint_assign(lower_physical,lower_nodal);
  const double integral_increment=op.integral(nodal_owned)-op.integral(lower_physical);
  AssertThrow(minimum_gap>-1e-10 && maximum<=.8+1e-10 && result.residual<=1e-10,
              ExcMessage("Native obstacle primal/dual constraints failed."));
  if (mode=="no-growth" || mode=="moving-obstacle") AssertThrow(error<1e-8,ExcMessage("Known discrete obstacle solution failed."));
  if (Utilities::MPI::this_mpi_process(comm)==0)
    std::cout<<std::setprecision(17)<<"{\"mode\":\""<<mode<<"\",\"level\":"<<level<<",\"error\":"<<error
             <<",\"KKT\":"<<result.residual<<",\"iterations\":"<<result.iterations<<",\"linear_iterations\":"<<result.linear_iterations
             <<",\"minimum_coefficient_gap\":"<<minimum_gap<<",\"maximum_coefficient\":"<<maximum
             <<",\"active\":"<<active<<",\"free\":"<<free<<",\"growth_integral\":"<<integral_increment<<"}\n";
}
