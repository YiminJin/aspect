// Opt-in benchmark initialization, not a constitutive or aging-law modification.
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>

namespace aspect
{
  namespace BP3Benchmark
  {
    std::vector<double> weak_initial_state;

    template <int dim>
    void initialize_weak_state(const SimulatorAccess<dim> &sim,
                               const MaterialModel::PhaseFieldFault<dim> &model)
    {
      auto &manager=sim.get_reconstructed_fault_manager();
      auto &fault=manager.get_fault(0);
      const auto &law=model.get_fault_friction();
      const auto &intro=sim.introspection();
      const auto comm=sim.get_mpi_communicator();
      const unsigned int n=fault.n_vertices();
      const auto state=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
      std::vector<unsigned int> properties;
      for (const auto &name:intro.chemical_composition_field_names())
        properties.push_back(manager.get_property_information()[manager.get_property_index(
                               "phase field fault chemical composition "+name)].position);
      const auto fractions=[&](unsigned int j,double xi)
      {
        std::vector<double> chemical;
        for (const auto p:properties)
          chemical.push_back((1-xi)*fault.get_properties(j)[p]+xi*fault.get_properties(j+1)[p]);
        return MaterialModel::MaterialUtilities::compute_composition_fractions(chemical);
      };
      const double target=law.friction_coefficient({0.,1.},BP3::Vp,
                           law.get_characteristic_slip_distance()/BP3::Vp);
      std::vector<double> original(n),nodal(n),z(n);
      for (unsigned int i=0;i<n;++i)
        {
          original[i]=fault.get_properties(i)[state];
          nodal[i]=law.initial_state_for_friction_coefficient(
            fractions(std::min(i,n-2),i==n-1 ? 1. : 0.),BP3::Vp,target);
          z[i]=std::log(nodal[i]);
        }

      // Sample the same owned bulk QPs and source basis as mechanics, including
      // the two continued endpoint wedges. Lift a private FE copy: no solver
      // vector or particle history is changed by this initialization audit.
      LinearAlgebra::BlockVector owned(intro.index_sets.system_partitioning,comm);
      owned=sim.get_solution();
      sim.get_current_constraints().distribute(owned);
      LinearAlgebra::BlockVector lifted(intro.index_sets.system_partitioning,
                                       intro.index_sets.system_relevant_partitioning,comm);
      lifted=owned;
      manager.prepare_stokes_qp_projection_cache();
      const auto &quad=intro.quadratures.velocities;
      FEValues<dim> fe(sim.get_mapping(),sim.get_fe(),quad,
                      update_values|update_quadrature_points|update_JxW_values);
      std::vector<double> phi(quad.size()),temperature(quad.size());
      std::vector<std::vector<double>> chemical(properties.size(),std::vector<double>(quad.size()));
      struct Point {unsigned int j; double xi,weight; std::vector<double> fractions;};
      std::vector<Point> points;
      std::vector<double> mass(n);
      for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe.reinit(cell);
            const auto &associations=manager.get_stokes_qp_fault_associations(cell->id(),quad,fe.get_quadrature_points());
            fe[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)].get_function_values(lifted,phi);
            fe[intro.extractors.temperature].get_function_values(lifted,temperature);
            for (unsigned int c=0;c<chemical.size();++c)
              fe[intro.extractors.compositional_fields[intro.chemical_composition_field_indices()[c]]].get_function_values(lifted,chemical[c]);
            for (unsigned int q=0;q<quad.size();++q)
              if (associations[q].active)
                {
                  const auto &a=associations[q];
                  typename MaterialModel::PhaseFieldFault<dim>::ReconstructedFaultBulkPointInputs in;
                  in.fault_index=0;in.segment_index=a.segment_index;in.xi=a.xi;
                  in.phase_field=phi[q];in.previous_phase_field=phi[q];in.temperature=temperature[q];
                  std::vector<double> c;for (const auto &v:chemical)c.push_back(v[q]);
                  in.bulk_material_fractions=MaterialModel::MaterialUtilities::compute_composition_fractions(c);
                  const double w=fe.JxW(q)*model.evaluate_reconstructed_fault_bulk_point(in).localization_factor;
                  if (w==0.)continue;
                  const auto bg=model.reconstructed_fault_background_tractions(0,a.segment_index,a.xi);
                  AssertThrow(std::abs(bg.second-BP3::sigma0)<1e-6 &&
                    std::abs(bg.first-(BP3::sigma0*target+BP3::damping*BP3::Vp))<1e-6,
                    ExcMessage("Weak initialization requires uniform nominal background traction."));
                  points.push_back({a.segment_index,a.shape_1,w,fractions(a.segment_index,a.xi)});
                  mass[a.segment_index]+=w*a.shape_0;mass[a.segment_index+1]+=w*a.shape_1;
                }
          }
      {const auto local=mass;Utilities::MPI::sum(local,comm,mass);}
      for (const double m:mass) AssertThrow(m>0.,ExcMessage("Missing weak initialization row support."));

      // Solve in log nodal state, but interpolate Theta (not log Theta) at QPs.
      // Rows are friction excess; D_ij contains mu_Theta*N_i*N_j*Theta_j.
      const auto assemble=[&](const std::vector<double> &log_state,
                              std::vector<double> &r,std::vector<double> &band)
      {
        r.assign(n,0.);band.assign(3*n,0.);
        for (const auto &p:points)
          {
            const double N[2]={1-p.xi,p.xi};
            const double theta[2]={std::exp(log_state[p.j]),std::exp(log_state[p.j+1])};
            const double value=N[0]*theta[0]+N[1]*theta[1];
            const double mu=law.friction_coefficient(p.fractions,BP3::Vp,value);
            const double derivative=law.friction_coefficient_derivative_wrt_state(p.fractions,BP3::Vp,value);
            for (unsigned int i=0;i<2;++i)
              {
                r[p.j+i]+=p.weight*N[i]*BP3::sigma0*(mu-target);
                for (unsigned int k=0;k<2;++k)
                  band[3*(p.j+i)+1+int(k)-int(i)]+=p.weight*N[i]*BP3::sigma0*derivative*N[k]*theta[k];
              }
          }
        for (auto *v:{&r,&band}) {const auto local=*v;Utilities::MPI::sum(local,comm,*v);}
      };
      const auto norm=[&](const std::vector<double> &r)
      {double v=0.;for (unsigned int i=0;i<n;++i)v=std::max(v,std::abs(r[i])/mass[i]);return v;};
      std::vector<double> r,band;
      assemble(z,r,band);
      const auto nodal_residual=r;

      // A central directional check tests the actual assembled log-state
      // derivative, including the MPI row reduction and Q1 interpolation.
      auto plus=z,minus=z;
      for (unsigned int i=0;i<n;++i) {plus[i]+=1e-5*std::sin(.31*i);minus[i]-=1e-5*std::sin(.31*i);}
      std::vector<double> rp,rm,unused;assemble(plus,rp,unused);assemble(minus,rm,unused);
      double fd_error=0.,fd_scale=0.;
      for (unsigned int i=0;i<n;++i)
        {
          double action=0.;
          for (int k=-1;k<=1;++k)
            if (int(i)+k>=0 && int(i)+k<int(n)) action+=band[3*i+1+k]*std::sin(.31*(int(i)+k));
          fd_error=std::max(fd_error,std::abs((rp[i]-rm[i])/2e-5-action)/mass[i]);
          fd_scale=std::max(fd_scale,std::abs(action)/mass[i]);
        }
      AssertThrow(fd_error<1e-7*fd_scale,ExcMessage("Weak-state initialization derivative check failed."));
      DynamicSparsityPattern dynamic(n,n);
      for (unsigned int i=0;i<n;++i)
        for (int k=-1;k<=1;++k) if (int(i)+k>=0 && int(i)+k<int(n))dynamic.add(i,int(i)+k);
      SparsityPattern sparsity;sparsity.copy_from(dynamic);
      SparseMatrix<double> matrix(sparsity);
      unsigned int iterations=0;
      while (norm(r)>1e-5)
        {
          AssertThrow(iterations++<12,ExcMessage("Weak-state initialization did not converge."));
          matrix=0.;
          Vector<double> rhs(n),direction(n);
          for (unsigned int i=0;i<n;++i)
            {
              rhs[i]=-r[i];
              for (int k=-1;k<=1;++k)
                if (int(i)+k>=0 && int(i)+k<int(n))matrix.set(i,int(i)+k,band[3*i+1+k]);
            }
          SparseDirectUMFPACK inverse;inverse.initialize(matrix);inverse.vmult(direction,rhs);
          bool accepted=false;
          for (unsigned int backtrack=0;backtrack<16;++backtrack)
            {
              auto candidate=z;
              for (unsigned int i=0;i<n;++i) candidate[i]+=std::ldexp(1.,-int(backtrack))*direction[i];
              assemble(candidate,rp,unused);
              if (norm(rp)<norm(r)) {z=candidate;r=rp;band=unused;accepted=true;break;}
            }
          AssertThrow(accepted,ExcMessage("Weak-state initialization line search exhausted."));
        }

      // Publish only the converged positive initial field. Keep an independent
      // copy for the timestep-zero retention audit; later aging is unchanged.
      weak_initial_state.resize(n);
      for (unsigned int i=0;i<n;++i)
        {
          weak_initial_state[i]=std::exp(z[i]);
          AssertThrow(std::isfinite(weak_initial_state[i]) && weak_initial_state[i]>0.,
                      ExcMessage("Invalid weak initial state."));
        }
      for (unsigned int i=0;i<n;++i)fault.get_properties(i)[state]=weak_initial_state[i];
      if (sim.get_pcout().is_active())
        {
          std::ofstream configuration(sim.get_output_directory()+"friction_configuration.csv");
          configuration<<std::setprecision(17)<<"nominal_q_Pa,mu_target,Dc\n"
                       <<BP3::sigma0*target+BP3::damping*BP3::Vp<<','<<target<<','
                       <<law.get_characteristic_slip_distance()<<'\n';
          std::ofstream out(sim.get_output_directory()+"weak_initialization.csv");
          out<<std::setprecision(17)<<"node,xd,Theta_original,Theta_projected_inverse,Theta_weak,weight,nodal_friction_excess_Pa,weak_friction_excess_Pa\n";
          for (unsigned int i=0;i<n;++i)
            out<<i<<','<<BP3::down_dip(fault.vertex(i)[0],fault.vertex(i)[1])<<','<<original[i]<<','<<nodal[i]<<','
               <<weak_initial_state[i]<<','<<mass[i]<<','<<nodal_residual[i]/mass[i]<<','<<r[i]/mass[i]<<'\n';
          sim.get_pcout()<<"   BP5 initial state: projected-nodal weak error="<<norm(nodal_residual)
                         <<" Pa, final="<<norm(r)<<" Pa, updates="<<iterations
                         <<", derivative relative error="<<fd_error/fd_scale<<std::endl;
        }
    }
  }
}
