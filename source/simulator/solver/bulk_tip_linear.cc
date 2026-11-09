/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/simulator.h>
#include <aspect/simulator/solver/bulk_tip.h>
#include <aspect/phase_field/box_constraints.h>
#include <aspect/phase_field/obstacle.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>

namespace aspect
{
  namespace
  {
    /** Adapter for Krylov actions, not a nonlinear or timestep executor. */
    struct TipLinearAction
    {
      std::function<void(LinearAlgebra::BlockVector &,const LinearAlgebra::BlockVector &)> action;
      void vmult(LinearAlgebra::BlockVector &dst,const LinearAlgebra::BlockVector &src) const
      {
        action(dst,src);
      }
    };
  }

  template <int dim>
  void Simulator<dim>::apply_bulk_tip_stokes_inverse(LinearAlgebra::BlockVector &dst,
                                                     const LinearAlgebra::BlockVector &src)
  {
    TimerOutput::Scope timer(computing_timer,"Tip Stokes inverse");
    const auto &coupling=*bulk_tip_coupling;
    const auto saved_linearization=current_linearization_point;
    const auto saved_normalization=last_pressure_normalization_adjustment;
    const auto saved_rhs=system_rhs;
    const bool print_inner=pcout.is_active();
    pcout.set_condition(false);
    system_rhs=0.;
    system_rhs.block(0)=src.block(0);
    system_rhs.block(0)*=coupling.shear_modulus();
    system_rhs.block(1)=src.block(1);
    // Replace the omitted gauge equation by the compatible continuity row.
    // Partition of unity makes the sum of the closed-domain continuity rows
    // zero. ASPECT then solves on its usual pressure quotient space.
    double sum=0.;
    for (const auto i:src.block(1).locally_owned_elements())if (i!=0)sum+=src.block(1)[i];
    sum=Utilities::MPI::sum(sum,mpi_communicator);
    if (system_rhs.block(1).locally_owned_elements().is_element(0))system_rhs.block(1)[0]=-sum;
    system_rhs.block(1).compress(VectorOperation::insert);
    system_rhs.block(1)*=pressure_scaling/coupling.timestep();
    current_linearization_point=0.;
    last_pressure_normalization_adjustment=0.;
    // Normalize homogeneous inverse loads to avoid absolute Arnoldi breakdown
    // thresholds when the initial continuity correction is only roundoff.
    const double rhs_scale=system_rhs.l2_norm();
    if (rhs_scale>0.)system_rhs*=1./rhs_scale;
    LinearAlgebra::BlockVector answer(solution);
    answer=0.;
    try
      {
        if (system_rhs.l2_norm()>0.)
          {
            solve_stokes(answer);
            answer*=rhs_scale;
            ++bulk_tip_coupling->stokes_solves;
          }
        dst.block(0)=answer.block(0);
        dst.block(0)*=coupling.timestep();
        dst.block(1)=answer.block(1);
        dst.block(1)/=coupling.shear_modulus();
        const double gauge=Utilities::MPI::sum(dst.block(1).locally_owned_elements().is_element(0)?double(dst.block(1)[0]):0.,mpi_communicator);
        for (const auto i:dst.block(1).locally_owned_elements())dst.block(1)[i]-=gauge;
        dst.block(1).compress(VectorOperation::insert);
      }
    catch (...)
      {
        current_linearization_point=saved_linearization;
        last_pressure_normalization_adjustment=saved_normalization;
        system_rhs=saved_rhs;
        pcout.set_condition(print_inner);
        throw;
      }
    current_linearization_point=saved_linearization;
    last_pressure_normalization_adjustment=saved_normalization;
    system_rhs=saved_rhs;
    pcout.set_condition(print_inner);
  }

  template <int dim>
  bool Simulator<dim>::compute_bulk_tip_search_direction(const LinearAlgebra::BlockVector &x,
                                                  const LinearAlgebra::BlockVector &residual,
                                                  const double shift,
                                                  LinearAlgebra::BlockVector &direction)
  {
    TimerOutput::Scope timer(computing_timer,shift>0. ? "Tip shifted search" : "Tip exact search");
    using Vector=LinearAlgebra::BlockVector;
    auto &coupling=*bulk_tip_coupling;
    const auto pb=coupling.phase_block();
    const auto &owned=introspection.index_sets.system_partitioning[pb];
    const auto &relevant=introspection.index_sets.system_relevant_partitioning[pb];
    const auto &mass=coupling.phase_mass();
    const auto &lower=coupling.obstacle();
    LinearAlgebra::Vector active(owned,mpi_communicator),active_ghost(owned,relevant,mpi_communicator);
    active=0.;
    direction=0.;
    if (shift>0.)
      {
        // A convex phase search metric is a globalization device, not a
        // replacement for the physical Jacobian or final VI. Its minimizer
        // obeys the true obstacle; the exact mechanics/section derivative then
        // supplies the coupled correction for this imposed phase displacement.
        LinearAlgebra::SparseMatrix metric;
        coupling.phase_search_matrix(metric,shift);
        LinearAlgebra::Vector load(lower),upper(lower),phase(x.block(pb));
        upper=1.;
        metric.vmult(load,phase);
        load-=residual.block(pb);
        double minimum_mass=std::numeric_limits<double>::max();
        for (const auto i:owned)minimum_mass=std::min(minimum_mass,double(mass[i]));
        minimum_mass=Utilities::MPI::min(minimum_mass,mpi_communicator);
        PhaseField::solve_obstacle(metric,load,lower,upper,phase,relevant,mpi_communicator,
                                   1e-10*minimum_mass,100,1000);
        direction.block(pb)=phase;
        direction.block(pb)-=x.block(pb);
        Vector action(direction),rhs(direction),correction(direction);
        coupling.vmult(action,direction);
        rhs=residual;
        rhs*=-1.;
        rhs-=action;
        rhs.block(pb)=0.;
        const TipLinearAction matrix
        {
          [&](Vector &dst,const Vector &src)
          {
            Vector restricted(src);
            restricted.block(pb)=0.;
            coupling.vmult(dst,restricted);
            dst.block(pb)=src.block(pb);
          }
        };
        const TipLinearAction preconditioner
        {
          [&](Vector &dst,const Vector &src)
          {
            dst=src;
            apply_bulk_tip_stokes_inverse(dst,src);
          }
        };
        SolverControl control(600,std::max(1e-16,1e-10*rhs.l2_norm()));
        SolverFGMRES<Vector> solver(control,typename SolverFGMRES<Vector>::AdditionalData(80));
        correction=0.;
        try
          {
            solver.solve(matrix,correction,rhs,preconditioner);
          }
        catch (const SolverControl::NoConvergence &)
          {
            coupling.linear_iterations+=control.last_step();
            return false;
          }
        coupling.linear_iterations+=control.last_step();
        direction+=correction;
        return true;
      }
    ++coupling.exact_attempts;
    std::vector<std::vector<int>> classifications;
    Vector action(direction),gradient(direction),fixed(direction),rhs(direction),free(direction);
    // The initial multiplier classification can release on every following
    // pass. A crossed bound is never permanently added to an irreversible list.
    for (unsigned int pass=0; pass<24; ++pass)
      {
        ++coupling.active_passes;
        coupling.vmult(action,direction);
        gradient=residual;
        gradient+=action;
        for (const auto i:owned)gradient.block(pb)[i]+=shift*mass[i]*direction.block(pb)[i];
        unsigned int released=0,feasible=1;
        double violation=0.;
        fixed=0.;
        for (const auto i:owned)
          {
            const double value=PhaseField::box_trial_value(x.block(pb)[i],direction.block(pb)[i],lower[i],1.,1.);
            const auto bound=PhaseField::box_constraint(value,gradient.block(pb)[i],std::max(1.,shift)*mass[i],lower[i],1.);
            feasible &= value>=lower[i] && value<=1.;
            released+=active[i]!=0. && int(active[i])!=bound.status;
            active[i]=bound.status;
            fixed.block(pb)[i]=bound.status<0 ? lower[i]-x.block(pb)[i] : bound.status>0 ? 1.-x.block(pb)[i] : 0.;
            violation=std::max(violation,std::abs(bound.projected_residual));
          }
        coupling.active_releases+=Utilities::MPI::sum(released,mpi_communicator);
        feasible=Utilities::MPI::min(feasible,mpi_communicator);
        violation=Utilities::MPI::max(violation,mpi_communicator);
        if (pass>0 && feasible && violation<1e-9)return true;
        std::vector<int> classification;
        for (const auto i:owned)classification.push_back(int(active[i]));
        for (const auto &previous:classifications)
          if (Utilities::MPI::min(int(previous==classification),mpi_communicator))
            {
              ++coupling.active_repeats;
              pcout<<"   Linearized tip bounds repeat: shift="<<shift<<", pass="<<pass
                   <<", natural residual="<<violation<<std::endl;
              return false;
            }
        classifications.push_back(std::move(classification));
        active.compress(VectorOperation::insert);
        active_ghost=active;
        fixed.compress(VectorOperation::insert);
        coupling.vmult(action,fixed);
        rhs=residual;
        rhs*=-1.;
        rhs-=action;
        for (const auto i:owned)
          rhs.block(pb)[i]=active[i]!=0. ? 0. : rhs.block(pb)[i]-shift*mass[i]*fixed.block(pb)[i];
        rhs.compress(VectorOperation::insert);
        LinearAlgebra::SparseMatrix phase_preconditioner;
        coupling.phase_search_matrix(phase_preconditioner,std::max(1.,shift));
        for (const auto i:owned)
          if (active[i]!=0.)phase_preconditioner.clear_row(i,1.);
          else for (auto entry=phase_preconditioner.begin(i); entry!=phase_preconditioner.end(i); ++entry)
              if (active_ghost[entry->column()]!=0.)entry->value()=0.;
        phase_preconditioner.compress(VectorOperation::insert);
        LinearAlgebra::PreconditionAMG phase_inverse;
        LinearAlgebra::PreconditionAMG::AdditionalData data;
        data.elliptic=true;
        data.higher_order_elements=true;
        data.smoother_sweeps=2;
        {
          TimerOutput::Scope setup(computing_timer,"Tip phase AMG setup");
          phase_inverse.initialize(phase_preconditioner,data);
        }
        const TipLinearAction matrix
        {
          [&](Vector &dst,const Vector &src)
          {
            Vector restricted(src);
            for (const auto i:owned)if (active[i]!=0.)restricted.block(pb)[i]=0.;
            restricted.compress(VectorOperation::insert);
            coupling.vmult(dst,restricted);
            for (const auto i:owned)dst.block(pb)[i]=active[i]!=0. ? double(src.block(pb)[i]) : dst.block(pb)[i]+shift*mass[i]*src.block(pb)[i];
            dst.compress(VectorOperation::insert);
          }
        };
        const TipLinearAction preconditioner
        {
          [&](Vector &dst,const Vector &src)
          {
            dst=src;
            apply_bulk_tip_stokes_inverse(dst,src);
            phase_inverse.vmult(dst.block(pb),src.block(pb));
          }
        };
        double minimum_mass=std::numeric_limits<double>::max();
        for (const auto i:owned)minimum_mass=std::min(minimum_mass,double(mass[i]));
        minimum_mass=Utilities::MPI::min(minimum_mass,mpi_communicator);
        const double tolerance=std::max(1e-16,std::min(1e-10*rhs.l2_norm(),1e-10*minimum_mass));
        SolverControl control(600,tolerance);
        SolverFGMRES<Vector> solver(control,typename SolverFGMRES<Vector>::AdditionalData(80));
        free=0.;
        try
          {
            solver.solve(matrix,free,rhs,preconditioner);
          }
        catch (const SolverControl::NoConvergence &)
          {
            coupling.linear_iterations+=control.last_step();
            return false;
          }
        coupling.linear_iterations+=control.last_step();
        // Prescribed active rows are exact algebraic data, not approximate
        // Krylov unknowns. Recheck the complete linearized VI after enforcing them.
        for (const auto i:owned)if (active[i]!=0.)free.block(pb)[i]=0.;
        free.compress(VectorOperation::insert);
        direction=free;
        direction+=fixed;
      }
    return false;
  }
#define INSTANTIATE(dim) \
  template void Simulator<dim>::apply_bulk_tip_stokes_inverse(LinearAlgebra::BlockVector &,const LinearAlgebra::BlockVector &); \
  template bool Simulator<dim>::compute_bulk_tip_search_direction(const LinearAlgebra::BlockVector &,const LinearAlgebra::BlockVector &,double,LinearAlgebra::BlockVector &);
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
