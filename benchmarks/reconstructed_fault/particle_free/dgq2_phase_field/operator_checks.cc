/* Native phase operator checks with independent manufactured fields. */
#include <aspect/phase_field/dg_operator.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace aspect;
#include "obstacle_checks.h"
#include "sipg_energy_checks.h"
#include "weak_form_checks.h"
int main(int argc,char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc,argv,1);
  deal_II_exceptions::disable_abort_on_exception();
  try
    {
      const unsigned int level=argc>1 ? std::stoi(argv[1]):2;
      const std::string mode=argc>2 ? argv[2]:"translate";
      const bool limiting=argc<=3 || std::stoi(argv[3]);
      const double interval=argc>4 ? std::stod(argv[4]):.4;
      const double terminal=argc>5 ? std::stod(argv[5]):.4;
      const bool refined=argc>6 && std::stoi(argv[6]);
      parallel::distributed::Triangulation<2> mesh(MPI_COMM_WORLD);
      GridGenerator::hyper_cube(mesh,-2,2);mesh.refine_global(level);
      if (refined)
        {
          for (const auto &cell:mesh.active_cell_iterators())
            if (cell->is_locally_owned() && cell->center()[0]>0.) cell->set_refine_flag();
          mesh.execute_coarsening_and_refinement();
        }
      const std::string geometry = argc>7 ? argv[7] : "cartesian";
      if (geometry != "cartesian")
        GridTools::transform([&](const Point<2> &p)
        {
          Point<2> q = p;
          if (geometry == "affine")
            {
              q[0] = .9*p[0] - .25*p[1];
              q[1] = .4*p[0] + 1.1*p[1];
            }
          else
            {
              AssertThrow(geometry == "bilinear", ExcMessage("Unknown mapped test geometry."));
              q[0] += .08*p[0]*p[1];
              q[1] += .06*p[0]*p[1];
            }
          return q;
        }, mesh);
      FE_DGQ<2> element(2);MappingQ1<2> mapping;DoFHandler<2> dofs(mesh);dofs.distribute_dofs(element);
      IndexSet relevant;DoFTools::extract_locally_relevant_dofs(dofs,relevant);
      const auto owned=dofs.locally_owned_dofs();
      PhaseField::DGOperator<2> op(mapping,dofs,0,0,owned,relevant,MPI_COMM_WORLD);
      if (mode=="weak-forms")
        { check_material_weak_forms(mapping,dofs,owned,relevant); return 0; }
      if (mode=="obstacle" || mode=="no-growth" || mode=="moving-obstacle")
        { obstacle_check(op,mapping,dofs,owned,relevant,mode,level);return 0; }
      if (mode=="elliptic")
        {
          const double a=numbers::PI/2.;
          const auto exact=[&](const Point<2> &x) {return .4+.1*std::sin(a*x[0])*std::cos(a*x[1]);};
          const auto coefficient=[](const Point<2>&x){return 1.+.1*x[0];};
          aspect::LinearAlgebra::SparseMatrix matrix;op.reinit_matrix(matrix);
          aspect::LinearAlgebra::Vector rhs(owned,MPI_COMM_WORLD),solution(owned,MPI_COMM_WORLD),x(owned,MPI_COMM_WORLD),y(owned,MPI_COMM_WORLD),ax(owned,MPI_COMM_WORLD),ay(owned,MPI_COMM_WORLD);
          const double boundary_constant=op.assemble_sipg(matrix,rhs,coefficient,10.,exact);
          const auto energy_checks=check_sipg_energy(mapping,dofs,matrix,rhs,boundary_constant,coefficient,exact);
          const QGauss<2> rule(8);
          FEValues<2> fe(mapping,element,rule,update_values|update_quadrature_points|update_JxW_values);
          std::vector<types::global_dof_index> indices(element.n_dofs_per_cell());
          for (const auto &cell:dofs.active_cell_iterators()) if (cell->is_locally_owned())
            {
              fe.reinit(cell);cell->get_dof_indices(indices);
              for (unsigned int q=0;q<rule.size();++q)
                {
                  const auto point=fe.quadrature_point(q);
                  const double f=2*a*a*coefficient(point)*(exact(point)-.4)-.01*a*std::cos(a*point[0])*std::cos(a*point[1])+.7*exact(point);
                  for (unsigned int i=0;i<indices.size();++i)
                    {
                      rhs[indices[i]]+=fe.JxW(q)*f*fe.shape_value(i,q);
                      for (unsigned int j=0;j<indices.size();++j)
                        matrix.add(indices[i],indices[j],.7*fe.JxW(q)*fe.shape_value(i,q)*fe.shape_value(j,q));
                    }
                }
            }
          matrix.compress(VectorOperation::add);rhs.compress(VectorOperation::insert);
          for (const auto i:owned){x[i]=std::sin(1.3*i);y[i]=std::cos(.7*i);}
          x.compress(VectorOperation::insert);y.compress(VectorOperation::insert);
          matrix.vmult(ax,x);matrix.vmult(ay,y);
          const double symmetry=std::abs(x*ay-y*ax)/std::max(1.,std::abs(x*ay)+std::abs(y*ax));
          const double positive=x*ax;
          double smallest_eigenvalue=0.;
          if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)==1 && level<=2)
            {
              LAPACKFullMatrix<double> dense(dofs.n_dofs());
              for (const auto i:owned)
                for (auto entry=matrix.begin(i);entry!=matrix.end(i);++entry)
                  dense(i,entry->column())=entry->value();
              dense.compute_eigenvalues();
              smallest_eigenvalue=std::numeric_limits<double>::max();
              for (unsigned int i=0;i<dofs.n_dofs();++i)
                {
                  AssertThrow(std::abs(dense.eigenvalue(i).imag())<1e-10,ExcMessage("SIPG eigenvalue is not real."));
                  smallest_eigenvalue=std::min(smallest_eigenvalue,dense.eigenvalue(i).real());
                }
              AssertThrow(smallest_eigenvalue>0.,ExcMessage("SIPG positive model is not coercive."));
            }
          SolverControl control(2000,1e-12*rhs.l2_norm());
          SolverCG<aspect::LinearAlgebra::Vector> solver(control);
          aspect::LinearAlgebra::PreconditionAMG preconditioner;
          aspect::LinearAlgebra::PreconditionAMG::AdditionalData data;data.elliptic=true;data.higher_order_elements=true;
          preconditioner.initialize(matrix,data);solver.solve(matrix,solution,rhs,preconditioner);
          aspect::LinearAlgebra::Vector ghost(owned,relevant,MPI_COMM_WORLD);ghost=solution;
          std::vector<double> values(rule.size());double error=0.;
          for (const auto &cell:dofs.active_cell_iterators()) if (cell->is_locally_owned())
            {
              fe.reinit(cell);fe.get_function_values(ghost,values);
              for (unsigned int q=0;q<rule.size();++q) error+=fe.JxW(q)*std::pow(values[q]-exact(fe.quadrature_point(q)),2);
            }
          error=std::sqrt(dealii::Utilities::MPI::sum(error,MPI_COMM_WORLD));
          AssertThrow(symmetry<2e-13 && positive>0.,ExcMessage("SIPG symmetry/coercivity check failed."));
          if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
            std::cout<<std::setprecision(17)<<"{\"mode\":\"elliptic\",\"level\":"<<level<<",\"l2\":"<<error
                     <<",\"energy_identity\":"<<energy_checks[0]<<",\"energy_residual\":"<<energy_checks[1]<<",\"energy_tangent\":"<<energy_checks[2]
                     <<",\"refined\":"<<refined<<",\"smallest_eigenvalue\":"<<smallest_eigenvalue<<",\"symmetry\":"<<symmetry<<",\"rayleigh_numerator\":"<<positive<<",\"iterations\":"<<control.last_step()<<"}\n";
          return 0;
        }
      const bool reverse=mode=="reverse";
      const auto initial=[&](const Point<2> &x)
      {
        if (mode=="constant") return .37;
        if (mode=="inflow" || mode=="quadratic-shear") return .4+.1*std::sin(numbers::PI*x[0])+.07*std::cos(numbers::PI*x[1]);
        const double a=1-std::pow((x[0]+.5)/.7,2),b=1-std::pow(x[1]/.7,2);
        return .35*std::pow(std::max(0.,a),4)*std::pow(std::max(0.,b),4);
      };
      const auto exact=[&](const Point<2> &x,double t)
      {
        auto y=x;
        if (mode=="quadratic-shear") y[0]-=.2*t*x[1]*x[1];
        else if (mode=="rotation")
          { y[0]=std::cos(.7*t)*x[0]+std::sin(.7*t)*x[1];y[1]=-std::sin(.7*t)*x[0]+std::cos(.7*t)*x[1]; }
        else if (mode=="divergence" || mode=="constant")
          { y[0]*=std::exp(-.3*t);y[1]*=std::exp(-.2*t); }
        else y[0]-=(reverse ? -1.:1.)*t;
        return initial(y);
      };
      op.assemble_transport([&](const auto &,const auto &points,auto &velocity,auto &divergence)
      {
        for (unsigned int q=0;q<points.size();++q)
          {
            velocity[q]=Tensor<1,2>();divergence[q]=0.;
            if (mode=="quadratic-shear") velocity[q][0]=.2*points[q][1]*points[q][1];
            else if (mode=="rotation") { velocity[q][0]=-.7*points[q][1];velocity[q][1]=.7*points[q][0]; }
            else if (mode=="divergence" || mode=="constant")
              { velocity[q][0]=.3*points[q][0];velocity[q][1]=.2*points[q][1];divergence[q]=.5; }
            else velocity[q][0]=reverse ? -1.:1.;
          }
      });
      aspect::LinearAlgebra::Vector state(owned,MPI_COMM_WORLD),next(owned,MPI_COMM_WORLD),ghost(owned,relevant,MPI_COMM_WORLD);
      std::vector<types::global_dof_index> ids(element.n_dofs_per_cell());
      for (const auto &cell:dofs.active_cell_iterators()) if (cell->is_locally_owned())
        { cell->get_dof_indices(ids);for (unsigned int i=0;i<ids.size();++i)
            state[ids[i]]=initial(mapping.transform_unit_to_real_cell(cell,element.get_unit_support_points()[i])); }
      state.compress(VectorOperation::insert);
      const QGauss<2> rule(8);
      FEValues<2> fe(mapping,element,rule,update_values|update_quadrature_points|update_JxW_values);
      const auto error=[&](double t)
      {
        ghost=state;double e=0.,maximum=0.,minimum=1.;
        std::vector<double> values(rule.size());
        for (const auto &cell:dofs.active_cell_iterators()) if (cell->is_locally_owned())
          { fe.reinit(cell);fe.get_function_values(ghost,values);
            for (unsigned int q=0;q<rule.size();++q)
              { e+=fe.JxW(q)*std::pow(values[q]-exact(fe.quadrature_point(q),t),2);
                maximum=std::max(maximum,values[q]);minimum=std::min(minimum,values[q]); } }
        return std::array<double,3>{std::sqrt(dealii::Utilities::MPI::sum(e,MPI_COMM_WORLD)),dealii::Utilities::MPI::min(minimum,MPI_COMM_WORLD),dealii::Utilities::MPI::max(maximum,MPI_COMM_WORLD)};
      };
      const auto moments=[&](double time,bool reference)
      {
        ghost=state;std::array<double,5> sums{};
        std::vector<double> values(rule.size());
        for (const auto &cell:dofs.active_cell_iterators()) if (cell->is_locally_owned())
          {
            fe.reinit(cell);fe.get_function_values(ghost,values);
            for (unsigned int q=0;q<rule.size();++q)
              {
                const auto point=fe.quadrature_point(q);
                const double w=fe.JxW(q)*(reference?exact(point,time):values[q]);
                sums[0]+=w;sums[1]+=w*point[0];sums[2]+=w*point[1];
                sums[3]+=w*point[0]*point[0];sums[4]+=w*point[1]*point[1];
              }
          }
        for (auto &value:sums) value=dealii::Utilities::MPI::sum(value,MPI_COMM_WORLD);
        for (unsigned int d=1;d<5;++d) sums[d]/=sums[0];
        sums[3]=std::sqrt(std::max(0.,sums[3]-sums[1]*sums[1]));
        sums[4]=std::sqrt(std::max(0.,sums[4]-sums[2]*sums[2]));
        return sums;
      };
      const auto initial_error=error(0.);
      const auto initial_moments=moments(0.,false);
      const auto started=std::chrono::steady_clock::now();
      double time=0.,balance=0.,signed_balance=0.,maximum_balance=0.,source=0.,flux=0.;unsigned int steps=0;unsigned long long limited=0;double alpha=1.;
      while (time<terminal-1e-14)
        {
          const double dt=std::min(interval,terminal-time);
          const auto stats=op.transport(state,next,exact,time,dt,0.,1.,limiting);
          state=next;time+=dt;balance+=std::abs(stats.balance_error);signed_balance+=stats.balance_error;maximum_balance=std::max(maximum_balance,std::abs(stats.balance_error));source+=stats.integrated_source;flux+=stats.integrated_boundary_flux;
          steps+=stats.substeps;limited+=stats.limited_cells;alpha=std::min(alpha,stats.minimum_alpha);
        }
      const auto final_error=error(time);
      const auto final_moments=moments(time,false),reference_moments=moments(time,true);
      const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
        std::cout<<std::setprecision(17)<<"{\"mode\":\""<<mode<<"\",\"level\":"<<level<<",\"dofs\":"<<dofs.n_dofs()
                 <<",\"time\":"<<time<<",\"interval\":"<<interval<<",\"refined\":"<<refined<<",\"limited\":"<<limiting<<",\"initial_l2\":"<<initial_error[0]<<",\"final_l2\":"<<final_error[0]
                 <<",\"minimum\":"<<final_error[1]<<",\"maximum\":"<<final_error[2]<<",\"balance\":"<<balance
                 <<",\"signed_balance\":"<<signed_balance<<",\"maximum_step_balance\":"<<maximum_balance<<",\"source\":"<<source<<",\"boundary_flux\":"<<flux<<",\"substeps\":"<<steps
                 <<",\"limited_cells\":"<<limited<<",\"minimum_alpha\":"<<alpha<<",\"seconds\":"<<seconds
                 <<",\"operator_bytes_rank0\":"<<op.memory_consumption()
                 <<",\"initial_integral\":"<<initial_moments[0]<<",\"final_integral\":"<<final_moments[0]
                 <<",\"reference_integral\":"<<reference_moments[0]<<",\"centroid_x\":"<<final_moments[1]
                 <<",\"centroid_y\":"<<final_moments[2]<<",\"width_x\":"<<final_moments[3]<<",\"width_y\":"<<final_moments[4]
                 <<",\"centroid_error\":"<<std::hypot(final_moments[1]-reference_moments[1],final_moments[2]-reference_moments[2])
                 <<",\"width_error\":"<<std::hypot(final_moments[3]-reference_moments[3],final_moments[4]-reference_moments[4])<<"}\n";
      AssertThrow(balance<2e-12,ExcMessage("Native transport material balance failed."));
      if (mode=="constant") AssertThrow(final_error[0]<2e-13,ExcMessage("Material constant-state preservation failed."));
    }
  catch(const std::exception &e) { std::cerr<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);return 1; }
}
