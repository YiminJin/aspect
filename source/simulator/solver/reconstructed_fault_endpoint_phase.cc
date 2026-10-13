/* Restricted, default-off consistent-endpoint P2 minimization. */
#include <aspect/simulator.h>
#include <aspect/phase_field.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/plugins.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/endpoint_mpi.h>
#include <aspect/particle/manager.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <algorithm>

namespace aspect
{
  template<int dim> void Simulator<dim>::solve_reconstructed_fault_endpoint_phase(
    bool publish_history,const LinearAlgebra::BlockVector *initial_trial,ReconstructedFaultVector *trial_slip_rate)
  {
    TimerOutput::Scope timer(computing_timer,"Fault: endpoint phase solve");
    auto &material=Plugins::get_plugin_as_type<MaterialModel::PhaseFieldFault<dim>>(*material_model);
    auto &phase=*phase_field_handler;
    auto &manager=*reconstructed_fault_manager;
    AssertThrow(material.uses_consistent_endpoint_phase() && material.has_fe_strain_history()
                && timestep_number>0,
                ExcMessage("Endpoint minimization requires initialized accepted FE history."));
    const unsigned int b=introspection.variable("phase_field").block_index;
    const LinearAlgebra::BlockVector saved_solution(solution),saved_linearization(current_linearization_point);
    const double saved_pressure_adjustment=last_pressure_normalization_adjustment;
    const LinearAlgebra::BlockVector anchor(material.accepted_fe_strain_state());
    const auto committed_slip=manager.get_timestep_committed_slip_rate(0);
    const auto &property_info=manager.get_property_information();
    const auto q_property=property_info[manager.get_property_index("phase field fault cohesive traction")].position;
    const auto I_property=property_info[manager.get_property_index("phase field fault previous I h")].position;
    std::vector<std::pair<double,double>> surface_anchor;
    for(unsigned int i=0;i<manager.get_fault(0).n_vertices();++i)
      surface_anchor.emplace_back(manager.get_fault(0).get_properties(i)[q_property],manager.get_fault(0).get_properties(i)[I_property]);
    const auto &particles=phase.get_associated_particle_manager();
    const auto H_property=particles.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
    std::vector<double> H_anchor;
    for(const auto &particle:particles.get_particle_handler())H_anchor.push_back(particle.get_properties()[H_property]);
    const auto &settings=phase.solver_parameters;
    AssertThrow(settings.use_bound_constraints && settings.enforce_irreversibility,
                ExcMessage("Endpoint minimization requires the selected box and previous-physical-phase bounds."));
    const double scale=material.get_critical_energy_release_rates()[0]/((8./3.)*phase.get_length_scale());
    const unsigned int n=anchor.block(b).size();
    Vector<double> lower(n),x(n),mass(n),diagonal(n);
    LinearAlgebra::SparseMatrix search_matrix;
    LinearAlgebra::SparseMatrix fracture_matrix;
    Vector<double> fracture_gradient(n);
    double fracture_anchor=0.,fracture_arithmetic=0.;
    LinearAlgebra::SparseMatrix restricted_search_matrix;
    std::unique_ptr<TrilinosWrappers::SolverDirect> search_preconditioner;
    std::vector<bool> contact(n),previous_contact(n);
    bool preconditioner_ready=false;
    std::vector<Point<dim>> points(n);
    lower=internal::endpoint_values(anchor.block(b),mpi_communicator);x=lower;
    if(initial_trial)
      {
      x=internal::endpoint_values(initial_trial->block(b),mpi_communicator);
      for(unsigned int i=0;i<n;++i)
        {
          AssertThrow(x[i]>=lower[i] && x[i]<1.,ExcMessage("Diagnostic starting guess violates the physical obstacle."));
        }
      }
    const unsigned int component=introspection.variable("phase_field").first_component_index;
    std::vector<types::global_dof_index> local_indices(finite_element.n_dofs_per_cell());
    std::vector<double> coordinates(dim*n,0.),counts(n,0.);
    for(const auto &cell:dof_handler.active_cell_iterators())
      if(cell->is_locally_owned())
      {
      cell->get_dof_indices(local_indices);
      for(unsigned int v=0;v<GeometryInfo<dim>::vertices_per_cell;++v)
        {
          const auto global=local_indices[finite_element.component_to_system_index(component,v)];
          const auto i=global-anchor.get_block_indices().block_start(b);
          counts[i]+=1.;
          for(unsigned int d=0;d<dim;++d)coordinates[dim*i+d]+=cell->vertex(v)[d];
          mass[i]+=cell->measure()/GeometryInfo<dim>::vertices_per_cell;
        }
      }
    internal::endpoint_sum(coordinates,mpi_communicator);internal::endpoint_sum(counts,mpi_communicator);
    std::vector<double> local_mass(mass.begin(),mass.end());internal::endpoint_sum(local_mass,mpi_communicator);
    for(unsigned int i=0;i<n;++i)
      {mass[i]=local_mass[i];AssertThrow(mass[i]>0 && counts[i]>0,ExcInternalError());
       for(unsigned int d=0;d<dim;++d)points[i][d]=coordinates[dim*i+d]/counts[i];}
    struct Evaluation
    {
      double value,allowance,pressure_adjustment;
      Vector<double> gradient;
      Vector<double> elastic_gradient;
      LinearAlgebra::BlockVector bulk;
      ReconstructedFaultVector slip;
    };
    unsigned int mechanical_trials=0;
    unsigned int tangent_calls=0,finite_difference_calls=0,line_search_calls=0,phase_iterations=0;
    double mechanical_seconds=0.,tangent_seconds=0.;
    unsigned int quasi_newton_pairs=0,search_inverse_calls=0;
    std::vector<std::pair<Vector<double>,Vector<double>>> secants;
    Evaluation *current_base=nullptr;
    std::function<Vector<double>(const Vector<double> &)> linearized_action;
    ReconstructedFaultVector warm_slip;
    auto evaluate=[&](const Vector<double> &phi)
    {
      linearized_action={};
      const auto started=std::chrono::steady_clock::now();
      Evaluation e;e.gradient.reinit(n);
      LinearAlgebra::BlockVector owned(introspection.index_sets.system_partitioning,mpi_communicator);
      owned=solution;
      for(unsigned int i=0;i<n;++i)
        {
          AssertThrow(phi[i]>=0 && phi[i]<1.,ExcMessage("Undefined endpoint trial normalization."));

        }
      internal::endpoint_assign(owned.block(b),phi);
      solution=owned;current_linearization_point=solution;
      // This solve rolls manager V back and never publishes R,Z,H or q.
      // Like the scratch bulk solution, the previous trial rate is only a
      // nonlinear starting guess. The accepted Z/R anchor remains unchanged.
      e.slip=warm_slip;
      solve_reconstructed_fault_stokes(false,&e.slip,mechanical_trials>0);++mechanical_trials;
      mechanical_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
      warm_slip=e.slip;
      e.pressure_adjustment=last_pressure_normalization_adjustment;
      e.bulk.reinit(solution);e.bulk=solution;
      const auto endpoint=material.make_fe_endpoint_trial(e.bulk,e.slip);
      const auto elastic=material.evaluate_fe_endpoint_energy(e.bulk,endpoint);
      AssertThrow(elastic.admissible,ExcMessage("Endpoint trial energy is undefined."));
      // AT1 at fixed domains and homogeneous Gc has exactly a quadratic
      // CPDI fracture functional. Reuse its actual sparse operator; no
      // particle projection, surrogate stiffness or integration change.
      LinearAlgebra::Vector delta(introspection.index_sets.system_partitioning[b],mpi_communicator),kd(delta);
      Vector<double> increment(phi);increment-=lower;internal::endpoint_assign(delta,increment);
      fracture_matrix.vmult(kd,delta);
      const auto global_kd=internal::endpoint_values(kd,mpi_communicator);
      const double naive_fracture_start=fracture_anchor+.5*(delta*kd);
      double naive_fracture=naive_fracture_start;
      long double linear_fracture=0.,quadratic_fracture=0.;
      e.elastic_gradient.reinit(n);
      double positive=0.;
      for(unsigned int i=0;i<n;++i)
        {
          // Sum increments before adding the large reference constant. This
          // is the same exact quadratic functional, with less cancellation
          // in the true-energy line search near a soft branch.
          linear_fracture+=static_cast<long double>(fracture_gradient[i])*increment[i];
          quadratic_fracture+=static_cast<long double>(increment[i])*global_kd[i];
          naive_fracture+=fracture_gradient[i]*increment[i];
          e.elastic_gradient[i]=elastic.gradient[i]/scale;
          e.gradient[i]=e.elastic_gradient[i]+(fracture_gradient[i]+global_kd[i])/scale;
          positive+=elastic.roundoff_scale[i]*elastic.roundoff_scale[i];
        }
      const long double fracture=fracture_anchor+linear_fracture+.5L*quadratic_fracture;
      e.value=(fracture+elastic.bulk+elastic.cohesive+endpoint.friction_potential)/scale;
      pcout<<"   Endpoint fracture summation: old-new="<<static_cast<double>(naive_fracture-fracture)
           <<", addition estimate="<<n*std::numeric_limits<double>::epsilon()*std::abs(fracture_anchor)<<std::endl;
      e.allowance=(fracture_arithmetic+8*std::numeric_limits<double>::epsilon()*std::sqrt(positive))/scale;
      AssertThrow(material.accepted_fe_strain_state().block(b)==anchor.block(b)
                  && material.accepted_fe_strain_state().block(introspection.block_indices.velocities)
                     ==anchor.block(introspection.block_indices.velocities),
                  ExcMessage("A trial advanced accepted FE history."));
      AssertThrow(manager.get_timestep_committed_slip_rate(0)==committed_slip,
                  ExcMessage("An endpoint trial committed surface slip history."));
      for(unsigned int i=0;i<surface_anchor.size();++i)
        AssertThrow(manager.get_fault(0).get_properties(i)[q_property]==surface_anchor[i].first
                    && manager.get_fault(0).get_properties(i)[I_property]==surface_anchor[i].second,
                    ExcMessage("An endpoint trial changed accepted surface history."));
      unsigned int particle_index=0;
      for(const auto &particle:particles.get_particle_handler())
        AssertThrow(particle.get_properties()[H_property]==H_anchor[particle_index++],
                    ExcMessage("An endpoint trial changed particle peak history."));
      return e;
    };
    auto projected=[&](const Vector<double> &phi,const Vector<double> &g)
    {
      Vector<double> out(g);
      for(unsigned int i=0;i<n;++i)
        if(phi[i]==lower[i] && out[i]>0.)out[i]=0.;
      return out;
    };
    auto action=[&](const Vector<double> &phi,const Vector<double> &direction,const double relative)
    {
      if(relative==1e-5)
        {
          AssertThrow(current_base!=nullptr,ExcInternalError());
          const auto base_phase=internal::endpoint_values(current_base->bulk.block(b),mpi_communicator);
          for(unsigned int i=0;i<n;++i)
            AssertThrow(phi[i]==base_phase[i],ExcMessage("Tangent base differs from the explicit trial."));
          if(!linearized_action)
            {
              const auto endpoint=material.make_fe_endpoint_trial(current_base->bulk,current_base->slip);
              linearized_action=make_reconstructed_fault_endpoint_tangent(current_base->bulk,current_base->slip,endpoint.strain,endpoint.Z);
            }
          const auto started=std::chrono::steady_clock::now();
          Vector<double>result=linearized_action(direction);result/=scale;
          LinearAlgebra::Vector d(introspection.index_sets.system_partitioning[b],mpi_communicator),kd(d);
          internal::endpoint_assign(d,direction);
          fracture_matrix.vmult(kd,d);
          const auto global_kd=internal::endpoint_values(kd,mpi_communicator);
          for(unsigned int i=0;i<n;++i)result[i]+=global_kd[i]/scale;
          ++tangent_calls;
          tangent_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
          return result;
        }
      ++finite_difference_calls;
      // An inexact numerical action of the unchanged reduced Hessian. The
      // actual residual and energy, rather than this action, decide convergence
      // and acceptance. Independent half-spacing actions quantify its error.
      double eps=relative/direction.linfty_norm();
      for(unsigned int i=0;i<n;++i)if(direction[i]!=0.)
        eps=std::min(eps,.25*std::min(phi[i],1.-phi[i])/std::abs(direction[i]));
      AssertThrow(eps>0. && std::isfinite(eps),ExcMessage("No arithmetic room for endpoint tangent action."));
      pcout<<"   Endpoint tangent amplitude="<<eps*direction.linfty_norm()<<std::endl;
      Vector<double> plus(phi),minus(phi);plus.add(eps,direction);minus.add(-eps,direction);
      const auto ep=evaluate(plus),em=evaluate(minus);
      Vector<double> result(ep.elastic_gradient);result-=em.elastic_gradient;result/=2*eps;
      LinearAlgebra::Vector d(introspection.index_sets.system_partitioning[b],mpi_communicator),kd(d);
      internal::endpoint_assign(d,direction);
      fracture_matrix.vmult(kd,d);
      const auto global_kd=internal::endpoint_values(kd,mpi_communicator);
          for(unsigned int i=0;i<n;++i)result[i]+=global_kd[i]/scale;
      return result;
    };
    try
      {
        // Assemble only the sparse native CPDI fracture block. Its shifted
        // diagonal is a search preconditioner, never an energy/Hessian change.
        solution.block(b)=anchor.block(b);
        phase.assemble_phase_field_system(system_matrix,system_rhs,solution,true,&fracture_arithmetic,&fracture_anchor,true);
        fracture_matrix.copy_from(system_matrix.block(b,b));
        fracture_gradient=internal::endpoint_values(system_rhs.block(b),mpi_communicator);fracture_gradient*=-1.;
        std::vector<double> local_diagonal(n,0.);
        for(const auto i:system_rhs.block(b).locally_owned_elements())
          local_diagonal[i]=std::abs(system_matrix.block(b,b).diag_element(i))/scale+mass[i];
        internal::endpoint_sum(local_diagonal,mpi_communicator);
        std::copy(local_diagonal.begin(),local_diagonal.end(),diagonal.begin());
        search_matrix.copy_from(system_matrix.block(b,b));
        for(const auto i:system_rhs.block(b).locally_owned_elements())search_matrix.add(i,i,scale*mass[i]);
        search_matrix.compress(VectorOperation::add);
        auto factor_preconditioner=[&]()
          {
            if(preconditioner_ready && contact==previous_contact)return;
            search_preconditioner.reset();
            restricted_search_matrix.copy_from(search_matrix);
            for(const auto i:system_rhs.block(b).locally_owned_elements())
              if(contact[i])restricted_search_matrix.clear_row(i,1.);
              else
                {
                  const auto end=restricted_search_matrix.end(i);
                  for(auto entry=restricted_search_matrix.begin(i);entry!=end;++entry)
                    if(contact[entry->column()])entry->value()=0.;
                }
            restricted_search_matrix.compress(VectorOperation::insert);
            // Collective Amesos_Klu may gather/factor on rank zero internally.
            const auto factor_started=std::chrono::steady_clock::now();
            // This restricted option can factor its small sparse
            // fracture search operator. Exact application avoids spending
            // many expensive mechanical tangent actions on AMG error in an
            // anisotropic mesh. It is not the physical reduced Hessian.
            search_preconditioner=std::make_unique<TrilinosWrappers::SolverDirect>();
            search_preconditioner->initialize(restricted_search_matrix);
            pcout<<"   Endpoint phase Amesos factor: ranks="<<Utilities::MPI::n_mpi_processes(mpi_communicator)
                 <<", seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-factor_started).count()<<std::endl;
            previous_contact=contact;preconditioner_ready=true;
          };
        auto update_preconditioner=[&](const Vector<double>&gradient)
          {
            for(unsigned int i=0;i<n;++i)contact[i]=x[i]==lower[i] && gradient[i]>=0.;
            factor_preconditioner();
          };
        auto precondition=[&](const Vector<double>&r,Vector<double>&z)
          {
            ++search_inverse_calls;
            LinearAlgebra::Vector src(introspection.index_sets.system_partitioning[b],mpi_communicator),dst(src);
            internal::endpoint_assign(src,r);
            search_preconditioner->vmult(dst,src);
            z=internal::endpoint_values(dst,mpi_communicator);z*=scale;
          };
        Evaluation base=evaluate(x);
        current_base=&base;
        const double initial=projected(x,base.gradient).linfty_norm();
        const double target=std::max(settings.nonlinear_solver_tolerance*initial,base.allowance);
        double forcing_reference=initial;
        pcout<<std::setprecision(17)<<"   Endpoint phase target: initial="<<initial<<", target="<<target<<std::endl;
        bool converged=false;
        for(unsigned int iteration=0;iteration<settings.max_nonlinear_iterations;++iteration)
          {
            ++phase_iterations;
            Vector<double> g=projected(x,base.gradient);
            const double defect=g.linfty_norm();
            Vector<double> step(n);
            bool escape=false;
            if(defect<=target)
              {
                // Independently reassemble the original CPDI integral at
                // every stationary candidate. Qualification and publication
                // use this actual residual, at the original entry target.
                double actual_fracture=0.,actual_arithmetic=0.;
                phase.evaluate_fe_fracture(system_matrix,system_rhs,base.bulk,actual_fracture,actual_arithmetic);
                const auto endpoint=material.make_fe_endpoint_trial(base.bulk,base.slip);
                const auto elastic=material.evaluate_fe_endpoint_energy(base.bulk,endpoint);
                const double energy_error=std::abs(base.value-(actual_fracture+elastic.energy+endpoint.friction_potential)/scale);
                const auto actual_rhs=internal::endpoint_values(system_rhs.block(b),mpi_communicator);
                double gradient_error=0.;
                for(unsigned int i=0;i<n;++i)
                  {
                    const double actual=(elastic.gradient[i]-actual_rhs[i])/scale;
                    gradient_error=std::max(gradient_error,std::abs(actual-base.gradient[i]));
                    base.gradient[i]=actual;
                  }
                // Keep the one reference energy offset throughout this solve.
                // The quadratic form gives consistent energy differences;
                // replacing its offset by a differently rounded particle sum
                // here would bias subsequent tiny line-search comparisons.
                pcout<<"   Endpoint CPDI reuse check: energy="<<energy_error
                     <<", gradient="<<gradient_error<<std::endl;
                if(projected(x,base.gradient).linfty_norm()>target)continue;
                // The first instability of this restricted horizontal-band
                // fixture is translational. Probe the full FE derivative of
                // the current phase, without identifying/reflection of frames.
                std::map<std::pair<double,double>,unsigned int> nodes;
                for(unsigned int i=0;i<n;++i)nodes.emplace(std::make_pair(points[i][0],points[i][1]),i);
                for(unsigned int i=0;i<n;++i)
                  if(x[i]>lower[i]+1e-13)
                    {
                      auto it=nodes.find({points[i][0],points[i][1]});auto left=it,right=it;
                      if(left!=nodes.begin())--left;
                      ++right;
                      if(left->first.first==points[i][0] && right!=nodes.end() && right->first.first==points[i][0])
                        step[i]=-(x[right->second]-x[left->second])/(right->first.second-left->first.second);
                    }
                if(step.linfty_norm()==0.){converged=true;break;}
                step/=step.linfty_norm();
                const auto hd=action(x,step,1e-5),hf=action(x,step,5e-6);
                Vector<double> difference(hd);difference-=hf;
                const double curvature=step*hf,uncertainty=std::abs(step*difference);
                LinearAlgebra::BlockVector direction(introspection.index_sets.system_partitioning,mpi_communicator);
                internal::endpoint_assign(direction.block(b),step);
                LinearAlgebra::BlockVector ghost_direction(solution);ghost_direction=direction;
                FEValues<dim> values(*mapping,finite_element,QGauss<dim>(2),update_values|update_JxW_values);
                std::vector<double> sampled(values.n_quadrature_points);double mass_norm=0.;
                for(const auto &cell:dof_handler.active_cell_iterators())
                  if(cell->is_locally_owned())
                  {
                    values.reinit(cell);values[FEValuesExtractors::Scalar(component)].get_function_values(ghost_direction,sampled);
                    for(unsigned int q=0;q<sampled.size();++q)mass_norm+=values.JxW(q)*sampled[q]*sampled[q];
                  }
                mass_norm=Utilities::MPI::sum(mass_norm,mpi_communicator);
                pcout<<"   Endpoint translation curvature="<<scale*curvature
                     <<", action uncertainty="<<scale*uncertainty
                     <<", L2 Rayleigh quotient="<<scale*curvature/mass_norm<<std::endl;
                if(curvature>=-4*uncertainty){converged=true;break;}
                double feasible[2]={std::numeric_limits<double>::max(),std::numeric_limits<double>::max()};
                for(unsigned int i=0;i<n;++i)
                  for(unsigned int sign=0;sign<2;++sign)
                    {
                      const double d=(sign==0?1.:-1.)*step[i];
                      if(d<0.)feasible[sign]=std::min(feasible[sign],(x[i]-lower[i])/(-d));
                      if(d>0.)feasible[sign]=std::min(feasible[sign],(1.-x[i])/d);
                    }
                bool found=false;Vector<double> best(x);Evaluation selected=base;
                for(unsigned int sign=0;sign<2;++sign)if(feasible[sign]>0.)
                  {
                    Vector<double> trial(x);trial.add((sign==0?1.:-1.)*.5*feasible[sign],step);
                    auto e=evaluate(trial);
                    const double allowance=32*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(base.value));
                    if(e.value<selected.value-allowance){best=trial;selected=std::move(e);found=true;}
                  }
                AssertThrow(found,ExcMessage("Resolved negative endpoint curvature has no resolved admissible descent; retain entry for diagnosis."));
                x=best;base=std::move(selected);escape=true;
                secants.clear();
                // A stationary saddle has a tiny entry residual. Restart only
                // intermediate forcing after departure; the physical entry
                // target above remains immutable.
                forcing_reference=std::max(forcing_reference,projected(x,base.gradient).linfty_norm());
              }
            if(escape)continue;
            bool obstacle_ready=false;
            Vector<double> obstacle_step(n);
            if(settings.endpoint_quasi_newton_search)
              {
                // Approximate the inverse only. Each gradient is the actual
                // mechanically reduced endpoint gradient at the same immutable
                // incoming histories. Keep all native tangential modes.
                update_preconditioner(base.gradient);
                std::vector<Vector<double>> s,y;
                std::vector<double> rho,alpha;
                for(const auto &pair:secants)
                  {
                    Vector<double> restricted_s(pair.first),restricted_y(pair.second);
                    for(unsigned int i=0;i<n;++i)if(contact[i])
                      {restricted_s[i]=0.;restricted_y[i]=0.;}
                    const double pairing=restricted_s*restricted_y;
                    if(pairing>1e-12*restricted_s.l2_norm()*restricted_y.l2_norm())
                      {s.push_back(std::move(restricted_s));y.push_back(std::move(restricted_y));rho.push_back(1./pairing);}
                  }
                Vector<double> q(g),z(n);
                for(unsigned int j=s.size();j>0;--j)
                  {const double a=rho[j-1]*(s[j-1]*q);alpha.push_back(a);q.add(-a,y[j-1]);}
                precondition(q,z);
                for(unsigned int j=0;j<s.size();++j)
                  z.add(alpha[s.size()-1-j]-rho[j]*(y[j]*z),s[j]);
                step=z;step*=-1.;
                pcout<<"   Endpoint quasi-Newton search: stored pairs="<<secants.size()
                     <<", positive restricted pairs="<<s.size()
                     <<", free="<<std::count(contact.begin(),contact.end(),false)<<std::endl;
              }
            else
              {
            // Predict the obstacle at every iterate with the existing positive
            // search metric. Otherwise a clipped Newton step can subsequently
            // admit only one layer per iteration. This metric selects a search
            // set; the physical Hessian, energy and convergence test stay exact.
            Vector<double> bound_step(n);
            for(unsigned int i=0;i<n;++i)bound_step[i]=lower[i]-x[i];
            for(unsigned int i=0;i<n;++i)contact[i]=base.gradient[i]>=0.;
            for(unsigned int k=0;k<settings.max_linear_solver_iterations;++k)
              {
                factor_preconditioner();
                LinearAlgebra::Vector d(introspection.index_sets.system_partitioning[b],mpi_communicator),pd(d);
                Vector<double> prescribed(n);for(unsigned int i=0;i<n;++i)prescribed[i]=contact[i]?bound_step[i]:0.;
                internal::endpoint_assign(d,prescribed);search_matrix.vmult(pd,d);
                auto global_pd=internal::endpoint_values(pd,mpi_communicator);
                Vector<double> rhs(base.gradient);rhs*=-1.;
                for(unsigned int i=0;i<n;++i)rhs[i]-=global_pd[i]/scale;
                for(unsigned int i=0;i<n;++i)if(contact[i])rhs[i]=0.;
                precondition(rhs,obstacle_step);
                for(unsigned int i=0;i<n;++i)prescribed[i]=contact[i]?bound_step[i]:obstacle_step[i];
                internal::endpoint_assign(d,prescribed);search_matrix.vmult(pd,d);
                global_pd=internal::endpoint_values(pd,mpi_communicator);
                std::vector<bool> next=contact;
                double free_residual=0.;
                for(unsigned int i=0;i<n;++i)
                  {
                    obstacle_step[i]=prescribed[i];
                    const double multiplier=base.gradient[i]+global_pd[i]/scale;
                    if(contact[i] && multiplier<-target)next[i]=false;
                    if(!contact[i] && obstacle_step[i]<bound_step[i])next[i]=true;
                    if(!contact[i])free_residual=std::max(free_residual,std::abs(multiplier));
                  }
                if(next==contact)
                  {
                    obstacle_ready=obstacle_step.linfty_norm()>0. && base.gradient*obstacle_step<0.
                              && free_residual<=settings.linear_solver_tolerance*base.gradient.l2_norm()+base.allowance;
                    pcout<<"   Endpoint obstacle search: iterations="<<k+1
                         <<", quadratic free residual="<<free_residual
                         <<", proposed free="<<std::count(contact.begin(),contact.end(),false)<<std::endl;
                    break;
                  }
                contact=std::move(next);
              }
            const bool predicted=iteration==0 && obstacle_ready;
            if(predicted)step=obstacle_step;
            // Preconditioned truncated Newton CG. Negative curvature supplies
            // a descent search direction; it never changes the physical energy.
            if(!predicted)
              {
                if(!obstacle_ready)update_preconditioner(base.gradient);
                else factor_preconditioner();
                Vector<double> residual(base.gradient),z(n),p(n);residual*=-1.;
                if(obstacle_ready)
                  {
                    for(unsigned int i=0;i<n;++i)step[i]=contact[i]?bound_step[i]:0.;
                    if(step.linfty_norm()>0.)residual-=action(x,step,1e-5);
                  }
                for(unsigned int i=0;i<n;++i)if(contact[i])residual[i]=0.;
                const double eta=settings.adaptive_endpoint_linear_accuracy
                  ?std::min(.01,std::max(settings.linear_solver_tolerance,
                    .1*std::sqrt(defect/std::max(forcing_reference,target)))):settings.linear_solver_tolerance;
                const double linear_target=settings.adaptive_endpoint_linear_accuracy
                  ?std::max(eta*residual.l2_norm(),.1*target):eta*residual.l2_norm();
                pcout<<"   Endpoint intermediate accuracy: defect="<<defect<<", reference="<<forcing_reference<<", eta="<<eta
                     <<", rhs="<<residual.l2_norm()<<", target="<<linear_target<<std::endl;
                Vector<double> coarse(n);
                std::vector<Vector<double>> coarse_basis,coarse_actions;
                Vector<double> coarse_eigenvalues;
                if(settings.endpoint_translation_preconditioner)
                  {
                    std::map<std::pair<double,double>,unsigned int> nodes;
                    for(unsigned int i=0;i<n;++i)nodes.emplace(std::make_pair(points[i][0],points[i][1]),i);
                    for(unsigned int i=0;i<n;++i)if(!contact[i])
                      {
                        auto it=nodes.find({points[i][0],points[i][1]});auto left=it,right=it;
                        if(left!=nodes.begin())--left;
                        ++right;
                        if(left->first.first==points[i][0] && right!=nodes.end() && right->first.first==points[i][0])
                          coarse[i]=-(x[right->second]-x[left->second])/(right->first.second-left->first.second);
                      }
                    // One profile-translation block with independent nodal
                    // tangential amplitudes. It is a small search inverse,
                    // never a restriction of the actual phase/mechanical space.
                    std::map<double,std::vector<unsigned int>> columns;
                    // Spend block setup only as intermediate accuracy tightens.
                    // Early searches retain the single global translation.
                    const bool expand=!settings.adaptive_endpoint_linear_accuracy || eta<.01;
                    for(unsigned int i=0;i<n;++i)if(coarse[i]!=0.)columns[expand?points[i][0]:0.].push_back(i);
                    std::vector<Vector<double>> basis,images;
                    double action_uncertainty=0.;
                    for(const auto &column:columns)
                      {
                        Vector<double> v(n);for(const auto i:column.second)v[i]=coarse[i];v/=v.l2_norm();
                        auto hv=action(x,v,1e-5);
                        for(unsigned int i=0;i<n;++i)if(contact[i])hv[i]=0.;
                        action_uncertainty+=hv.norm_sqr();basis.push_back(std::move(v));images.push_back(std::move(hv));
                      }
                    if(!basis.empty())
                      {
                        LAPACKFullMatrix<double> block(basis.size());
                        for(unsigned int i=0;i<basis.size();++i)for(unsigned int j=0;j<basis.size();++j)
                          block(i,j)=.5*(basis[i]*images[j]+basis[j]*images[i]);
                        const double guard=4*settings.linear_solver_tolerance*std::sqrt(action_uncertainty);
                        FullMatrix<double> rotation;
                        Vector<double> eigenvalues;
                        block.compute_eigenvalues_symmetric(0.,2*std::max(guard,block.linfty_norm()),
                                                           0.,eigenvalues,rotation);
                        std::vector<double> resolved;
                        for(unsigned int j=0;j<eigenvalues.size();++j)
                          {
                            Vector<double> v(n),hv(n);
                            for(unsigned int i=0;i<basis.size();++i)
                              {v.add(rotation(i,j),basis[i]);hv.add(rotation(i,j),images[i]);}
                            double curvature=eigenvalues[j],uncertainty=0.;
                            if(curvature<=guard)
                              {
                                const auto direct=action(x,v,1e-5);
                                const double measured=v*direct;uncertainty=std::abs(measured-curvature);
                                hv=direct;for(unsigned int i=0;i<n;++i)if(contact[i])hv[i]=0.;
                                curvature=measured;
                                pcout<<"   Endpoint soft coarse check: eigenvalue="<<eigenvalues[j]
                                     <<", direct="<<measured<<", uncertainty="<<uncertainty<<std::endl;
                              }
                            const double mode_guard=4*std::max(settings.linear_solver_tolerance*hv.l2_norm(),uncertainty);
                            if(curvature>mode_guard)
                              {resolved.push_back(curvature);coarse_basis.push_back(std::move(v));coarse_actions.push_back(std::move(hv));}
                          }
                        coarse_eigenvalues.reinit(resolved.size());
                        for(unsigned int i=0;i<resolved.size();++i)coarse_eigenvalues[i]=resolved[i];
                        pcout<<"   Endpoint translation block: columns="<<basis.size()
                             <<", positive modes="<<coarse_basis.size()<<", uncertainty guard="<<guard<<std::endl;
                      }
                  }
                auto newton_precondition=[&](const Vector<double>&r,Vector<double>&out)
                  {
                    Vector<double> load(r);std::vector<double> coefficients(coarse_basis.size());
                    for(unsigned int j=0;j<coarse_basis.size();++j)
                      {
                        coefficients[j]=(coarse_basis[j]*r)/coarse_eigenvalues[j];
                        load.add(-coefficients[j],coarse_actions[j]);
                      }
                    precondition(load,out);
                    for(unsigned int j=0;j<coarse_basis.size();++j)
                      coefficients[j]-=(coarse_actions[j]*out)/coarse_eigenvalues[j];
                    for(unsigned int j=0;j<coarse_basis.size();++j)out.add(coefficients[j],coarse_basis[j]);
                  };
                newton_precondition(residual,z);
                for(unsigned int i=0;i<n;++i)if(contact[i])z[i]=0.;
                p=z;double rz=residual*z;
                for(unsigned int k=0;k<settings.max_linear_solver_iterations
                    && residual.l2_norm()>linear_target;++k)
                  {
                    if(k%5==0)pcout<<"   Endpoint tangent CG "<<k<<": residual="<<residual.l2_norm()
                                  <<", target="<<linear_target<<std::endl;
                    if(k>=25 && k%25==0)
                      {
                        std::map<double,std::pair<double,double>> rows;
                        double norm=0.,variation=0.,coarse_norm=0.,pairing=0.;
                        for(unsigned int i=0;i<n;++i)
                          {
                            rows[points[i][1]].first+=mass[i];rows[points[i][1]].second+=mass[i]*p[i];
                            norm+=mass[i]*p[i]*p[i];coarse_norm+=mass[i]*coarse[i]*coarse[i];
                            pairing+=mass[i]*p[i]*coarse[i];
                          }
                        for(unsigned int i=0;i<n;++i)
                          {
                            const auto &row=rows.at(points[i][1]);
                            variation+=mass[i]*std::pow(p[i]-row.second/row.first,2);
                          }
                        pcout<<"   Endpoint slow direction: iteration="<<k
                             <<", lumped tangential fraction="<<(norm>0.?variation/norm:0.)
                             <<", translation overlap="<<(norm>0. && coarse_norm>0.?std::abs(pairing)/std::sqrt(norm*coarse_norm):0.)<<std::endl;
                      }
                    Vector<double> hp=action(x,p,1e-5);
                    for(unsigned int i=0;i<n;++i)if(contact[i])hp[i]=0.;
                    const double curvature=p*hp;
                    if(curvature<=0.)
                      {
                        auto finer=action(x,p,5e-6);
                        for(unsigned int i=0;i<n;++i)if(contact[i])finer[i]=0.;
                        pcout<<"   Endpoint CG curvature="<<curvature
                             <<", half-spacing="<<p*finer
                             <<", uncertainty="<<std::abs(p*finer-curvature)<<std::endl;
                        if(k==0)step+=p;
                        break;
                      }
                    const double alpha=rz/curvature;step.add(alpha,p);residual.add(-alpha,hp);
                    newton_precondition(residual,z);
                    for(unsigned int i=0;i<n;++i)if(contact[i])z[i]=0.;
                    const double next=residual*z;p*=next/rz;p+=z;rz=next;
                  }
              }
              }
            if(base.gradient*step>=0. || step.linfty_norm()==0.)
              {
                if(obstacle_ready)step=obstacle_step;
                else for(unsigned int i=0;i<n;++i)step[i]=-g[i]/diagonal[i];
              }
            bool accepted=false;
            for(unsigned int ls=0;ls<settings.max_newton_line_search_iterations;++ls)
              {
                Vector<double> trial(x);
                for(unsigned int i=0;i<n;++i)trial[i]=std::max(lower[i],std::min(1.,x[i]+std::ldexp(1.,-int(ls))*step[i]));
                if(trial.linfty_norm()>=1.)continue;
                Vector<double> change(trial);change-=x;
                if(settings.endpoint_quasi_newton_search
                   && (change.linfty_norm()==0. || base.gradient*change>=0.))continue;
                ++line_search_calls;
                auto e=evaluate(trial);
                const double allowance=32*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(base.value));
                if(e.value<=base.value+1e-4*std::min(0.,base.gradient*change)+allowance)
                  {
                    if(settings.endpoint_quasi_newton_search)
                      {
                        Vector<double> gradient_change(e.gradient);gradient_change-=base.gradient;
                        if(change*gradient_change>1e-12*change.l2_norm()*gradient_change.l2_norm())
                          {
                            secants.emplace_back(change,std::move(gradient_change));++quasi_newton_pairs;
                            if(secants.size()>30)secants.erase(secants.begin());
                          }
                      }
                    x=trial;base=std::move(e);accepted=true;break;
                  }
              }
            AssertThrow(accepted,ExcMessage("Endpoint phase line search failed; no history published."));
            pcout<<"   Endpoint phase iteration "<<iteration<<": KKT="<<projected(x,base.gradient).linfty_norm()
                 <<", mechanical trials="<<mechanical_trials<<std::endl;
          }
        AssertThrow(converged,ExcMessage("Endpoint phase iteration budget exhausted; no history published."));
        solution=base.bulk;current_linearization_point=base.bulk;
        last_pressure_normalization_adjustment=base.pressure_adjustment;
        // Tangent probes may have left a different trial normalization in the
        // derived cache. Rebuild it from the selected endpoint before commit.
        material.prepare_reconstructed_fault_mechanical_solve();
        if(trial_slip_rate)*trial_slip_rate=base.slip;
        if(publish_history)
          {
            std::string precommit_error;
            try {if(benchmark_endpoint_precommit)benchmark_endpoint_precommit();}
            catch(const std::exception &e){precommit_error=e.what();}
            catch(...){precommit_error="Unknown prepublication failure";}
            const auto errors=Utilities::MPI::all_gather(mpi_communicator,precommit_error);
            for(unsigned int rank=0;rank<errors.size();++rank)
              if(!errors[rank].empty())
                {pcout<<"   Coordinated endpoint prepublication failure: rank="<<rank<<std::endl;
                 throw std::runtime_error(errors[rank]);}
            manager.begin_slip_rate_nonlinear_solve();
            try
              {
                manager.begin_slip_rate_trial();manager.set_slip_rate_trial_values(base.slip);manager.accept_slip_rate_trial();
                manager.validate_slip_rate_nonlinear_commit();
                material.commit_reconstructed_fault_mechanical_history(solution);
                manager.commit_slip_rate_nonlinear_solve();
              }
            catch (...){manager.rollback_slip_rate_nonlinear_solve();throw;}
            pcout<<"   Endpoint complete state committed once; mechanical trials="<<mechanical_trials<<std::endl;
          }
        else pcout<<"   Endpoint fixed-entry diagnostic converged; no history published."<<std::endl;
        pcout<<"   Endpoint cost: mechanical seconds="<<mechanical_seconds
             <<", tangent seconds="<<tangent_seconds<<", tangent calls="<<tangent_calls
             <<", finite-difference calls="<<finite_difference_calls<<", line searches="<<line_search_calls
             <<", phase iterations="<<phase_iterations<<std::endl;
        if(settings.endpoint_quasi_newton_search)
          pcout<<"   Endpoint quasi-Newton cost: secant updates="<<quasi_newton_pairs
               <<", search inverse calls="<<search_inverse_calls<<std::endl;
      }
    catch (...)
      {
        solution=saved_solution;current_linearization_point=saved_linearization;
        last_pressure_normalization_adjustment=saved_pressure_adjustment;
        material.prepare_reconstructed_fault_mechanical_solve();
        throw;
      }
  }
}

namespace aspect
{
#define INSTANTIATE(dim) template void Simulator<dim>::solve_reconstructed_fault_endpoint_phase(bool,const LinearAlgebra::BlockVector *,ReconstructedFaultVector *);
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
