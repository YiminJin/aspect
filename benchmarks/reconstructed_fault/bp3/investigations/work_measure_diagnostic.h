// Archived K5 probe; not included by the maintained BP3 plugin.
// One fresh frozen-history work-measure qualification. Never commits histories.
#include <deal.II/numerics/vector_tools_interpolate.h>

namespace aspect
{
  namespace BP3Benchmark
  {
    template <int dim>
    class WorkBulkDirection : public Function<dim>
    {
      public:
        explicit WorkBulkDirection(const Introspection<dim> &intro)
          : Function<dim>(intro.n_components),intro(intro) {}
        void vector_value(const Point<dim> &p,Vector<double> &v) const override
        {
          v=0.;const double X=p[0]/BP3::box_size,Y=p[1]/BP3::box_size;
          const double speed=.1*BP3::Vp*X*(1-X)*Y;
          v[intro.component_indices.velocities[0]]=BP3::cosine*speed;
          v[intro.component_indices.velocities[1]]=BP3::sine*speed;
          v[intro.component_indices.pressure]=1000.*X*(1-X)*(.5+Y);
        }
      private:
        const Introspection<dim> &intro;
    };

    template <int dim>
    void prepare_work_measure_test(const SimulatorAccess<dim> &sim)
    {
      const auto &intro=sim.introspection();const auto comm=sim.get_mpi_communicator();
      auto &manager=sim.get_reconstructed_fault_manager();const auto &fault=manager.get_fault(0);
      auto &surface=sim.get_reconstructed_fault_surface_system();
      const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model());
      const auto &partition=intro.index_sets.system_partitioning;
      LinearAlgebra::BlockVector direction(partition,comm),base_owned(partition,comm);
      VectorTools::interpolate(sim.get_mapping(),sim.get_dof_handler(),WorkBulkDirection<dim>(intro),direction);
      // Copy the existing physical constraints but homogenize their offsets.
      AffineConstraints<double> homogeneous(sim.get_current_constraints());
      for (const auto &line:sim.get_current_constraints().get_lines()) homogeneous.set_inhomogeneity(line.index,0.);
      homogeneous.distribute(direction);
      base_owned=sim.get_solution();base_owned.add(1.,direction);
      sim.get_current_constraints().distribute(base_owned);
      // This is a fresh initial guess, not a timestep publication. The solver
      // snapshots this state and the existing rollback observer verifies it.
      auto &initial=const_cast<LinearAlgebra::BlockVector&>(sim.get_solution());initial=base_owned;
      auto &linearization=const_cast<LinearAlgebra::BlockVector&>(sim.get_current_linearization_point());linearization=base_owned;
      auto V=manager.get_timestep_committed_slip_rate(0);
      const auto prescribed=manager.prescribed_slip_rate_mask();
      ReconstructedFaultVector rates(1,V),dV(1,std::vector<double>(V.size()));
      for (unsigned int i=0;i<V.size();++i)
        if (!prescribed[0][i])
          {
            const auto p=fault.vertex(i);dV[0][i]=.1*BP3::Vp*std::exp(-std::max(0.,BP3::down_dip(p[0],p[1]))/1000.);
            rates[0][i]+=dV[0][i];
          }
      AssertThrow(!prescribed[0].back() && dV[0].back()>0.,ExcMessage("Work test does not exercise a free top."));
      manager.initialize_slip_rate(0,rates[0]);
      const auto ghost=[&](const LinearAlgebra::BlockVector &owned)
      {
        LinearAlgebra::BlockVector result(partition,intro.index_sets.system_relevant_partitioning,comm);
        result=owned;return result;
      };
      const auto retained_initial=surface.linearize_surface_system(ghost(base_owned),rates);
      const double retained_initial_rms=surface.surface_residual_rms(retained_initial,prescribed);
      // Make the affine-history check decisive without modifying actual inputs:
      // only this private probe gets a nonzero smooth FE old-stress tensor.
      std::map<unsigned int,unsigned int> stress_components;
      for (const auto &m:sim.get_parameters().mapped_particle_properties)
        if (m.second.first=="maxwell stress") stress_components[intro.component_indices.compositional_fields[m.first]]=m.second.second;
      std::vector<types::global_dof_index> indices(sim.get_fe().n_dofs_per_cell());
      for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
        if (!cell->is_artificial())
          {
            cell->get_dof_indices(indices);
            for (unsigned int i=0;i<indices.size();++i)
              {
                const auto c=stress_components.find(sim.get_fe().system_to_component_index(i).first);
                if (c==stress_components.end() || !sim.get_dof_handler().locally_owned_dofs().is_element(indices[i])) continue;
                const auto p=sim.get_mapping().transform_unit_to_real_cell(cell,sim.get_fe().get_unit_support_points()[i]);
                const double tensor[]={1.,-.4,.3};
                base_owned[indices[i]]=1000.*tensor[c->second]*(1+p[0]/BP3::box_size)*p[1]/BP3::box_size;
              }
          }
      base_owned.compress(VectorOperation::insert);sim.get_current_constraints().distribute(base_owned);
      const auto base=ghost(base_owned),bulk_direction=ghost(direction);
      const auto initial_residual=surface.linearize_surface_system(base,rates);
      ReconstructedFaultVector KdV,Gdir;
      surface.apply_surface_jacobian(dV,KdV);surface.apply_G(bulk_direction,Gdir);
      constexpr double epsilon=1e-3;
      auto plus=rates,minus=rates;
      for (unsigned int i=0;i<V.size();++i)
        {plus[0][i]+=epsilon*dV[0][i];minus[0][i]-=epsilon*dV[0][i];}
      const auto rp=surface.evaluate_surface_residual(base,plus),rm=surface.evaluate_surface_residual(base,minus);
      auto xp=base_owned,xm=base_owned;xp.add(epsilon,direction);xm.add(-epsilon,direction);
      const auto gp=surface.evaluate_surface_residual(ghost(xp),rates),gm=surface.evaluate_surface_residual(ghost(xm),rates);
      double Kerror=0.,Kscale=0.,Gerror=0.,Gscale=0.;
      for (unsigned int i=0;i<V.size();++i)
        {
          Kerror=std::max(Kerror,std::abs((rp.values[0][i]-rm.values[0][i])/(2*epsilon)+KdV[0][i]));
          Kscale=std::max(Kscale,std::abs(KdV[0][i]));
          Gerror=std::max(Gerror,std::abs((gp.values[0][i]-gm.values[0][i])/(2*epsilon)-Gdir[0][i]));
          Gscale=std::max(Gscale,std::abs(Gdir[0][i]));
        }

      // Isolate G's non-associated normal/friction part on the same bulk QPs.
      // The remainder must be work-adjoint to B, not to the full G.
      const auto &quad=intro.quadratures.velocities;
      FEValues<dim> fe(sim.get_mapping(),sim.get_fe(),quad,
        update_values|update_gradients|update_quadrature_points|update_JxW_values);
      const unsigned int nq=quad.size();
      std::vector<double> phi(nq),temp(nq),p(nq),delta_p(nq);
      std::vector<SymmetricTensor<2,dim>> eps(nq),delta_eps(nq);
      std::vector<std::vector<double>> c(intro.n_compositional_fields,std::vector<double>(nq));
      std::array<unsigned int,3> stress_fields;
      for (const auto &m:sim.get_parameters().mapped_particle_properties)
        if (m.second.first=="maxwell stress") stress_fields[m.second.second]=m.first;
      std::vector<double> normal(V.size()),independent_shear(V.size()),measure(V.size());
      double direct_B_work=0.;unsigned long long visits=0,positive=0,intact=0,wedge=0;
      double affine_max=0.,coefficient_max=0.,old_shear_max=0.;
      for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe.reinit(cell);const auto &assoc=manager.get_stokes_qp_fault_associations(cell->id(),quad,fe.get_quadrature_points());
            fe[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)].get_function_values(base,phi);
            fe[intro.extractors.temperature].get_function_values(base,temp);
            fe[intro.extractors.pressure].get_function_values(base,p);
            fe[intro.extractors.pressure].get_function_values(bulk_direction,delta_p);
            fe[intro.extractors.velocities].get_function_symmetric_gradients(base,eps);
            fe[intro.extractors.velocities].get_function_symmetric_gradients(bulk_direction,delta_eps);
            for (unsigned int j=0;j<c.size();++j) fe[intro.extractors.compositional_fields[j]].get_function_values(base,c[j]);
            for (unsigned int q=0;q<nq;++q)
              {
                ++visits;if (!assoc[q].active) continue;
                const auto &a=assoc[q];const double N[2]={1-a.xi,a.xi};
                typename MaterialModel::PhaseFieldFault<dim>::ReconstructedFaultPointInputs in;
                in.fault_index=0;in.segment_index=a.segment_index;in.xi=a.xi;in.position=fe.quadrature_point(q);
                in.phase_field=phi[q];in.previous_phase_field=phi[q];in.temperature=temp[q];in.dynamic_pressure=p[q];in.strain_rate=eps[q];
                in.slip_tensor=symmetrize(outer_product(a.tangent,a.normal));in.normal_tensor=symmetrize(outer_product(a.normal,a.normal));
                std::vector<double> chemical,all(c.size());
                for (unsigned int j=0;j<c.size();++j) all[j]=c[j][q];
                for (const auto j:intro.chemical_composition_field_indices()) chemical.push_back(all[j]);
                in.bulk_material_fractions=MaterialModel::MaterialUtilities::compute_composition_fractions(chemical);
                for (unsigned int j=0;j<3;++j) in.old_maxwell_stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(j)]=all[stress_fields[j]];
                in.slip_rate=N[0]*rates[0][a.segment_index]+N[1]*rates[0][a.segment_index+1];
                const auto r=model.evaluate_reconstructed_fault_point(in);
                if (r.localization_factor==0.) {++intact;continue;}
                ++positive;
                if (!manager.project_to_normal_profiles(in.position).active) ++wedge;
                typename MaterialModel::PhaseFieldFault<dim>::ReconstructedFaultBulkPointInputs bi;
                bi.fault_index=0;bi.segment_index=a.segment_index;bi.xi=a.xi;bi.phase_field=phi[q];bi.previous_phase_field=phi[q];
                bi.temperature=temp[q];bi.bulk_material_fractions=in.bulk_material_fractions;
                const auto br=model.evaluate_reconstructed_fault_bulk_point(bi);
                coefficient_max=std::max(coefficient_max,std::abs(br.localization_factor-r.localization_factor));
                coefficient_max=std::max(coefficient_max,std::abs(br.kappa/r.kappa-1.));
                const auto old=model.evaluate_frozen_maxwell_stress(temp[q],all,in.old_maxwell_stress);
                old_shear_max=std::max(old_shear_max,std::abs(old*in.slip_tensor));
                const auto stress=2*br.kappa*(eps[q]-br.localization_factor*in.slip_rate*in.slip_tensor)+old;
                const double expected=model.reconstructed_fault_background_tractions(0,a.segment_index,a.xi).first+stress*in.slip_tensor;
                affine_max=std::max(affine_max,std::abs(expected-r.shear_traction));
                const double weight=fe.JxW(q)*r.localization_factor;
                const double nonassociated=r.friction_coefficient*(2*r.kappa*(in.normal_tensor*delta_eps[q])-delta_p[q]);
                double dv=0.;
                for (unsigned int j=0;j<2;++j)
                  {
                    const unsigned int vertex=a.segment_index+j;
                    normal[vertex]+=weight*N[j]*nonassociated;
                    independent_shear[vertex]+=weight*N[j]*expected;
                    measure[vertex]+=weight*N[j];dv+=N[j]*dV[0][vertex];
                  }
                direct_B_work+=weight*2*r.kappa*dv*(in.slip_tensor*delta_eps[q]);
              }
          }
      for (auto *v:{&normal,&independent_shear,&measure})
        {const auto local=*v;Utilities::MPI::sum(local,comm,*v);}
      direct_B_work=Utilities::MPI::sum(direct_B_work,comm);
      visits=Utilities::MPI::sum(visits,comm);positive=Utilities::MPI::sum(positive,comm);
      intact=Utilities::MPI::sum(intact,comm);wedge=Utilities::MPI::sum(wedge,comm);
      affine_max=Utilities::MPI::max(affine_max,comm);coefficient_max=Utilities::MPI::max(coefficient_max,comm);
      old_shear_max=Utilities::MPI::max(old_shear_max,comm);
      double Gshear_work=0.,measure_error=0.,affine_load_error=0.;
      for (unsigned int i=0;i<V.size();++i)
        {
          Gshear_work+=dV[0][i]*(Gdir[0][i]-normal[i]);
          double row=initial_residual.mass_diagonal[0][i];
          if (i) row+=initial_residual.mass_off_diagonal[0][i-1];
          if (i+1<V.size()) row+=initial_residual.mass_off_diagonal[0][i];
          AssertThrow(row>0.,ExcMessage("Work row has no positive measure."));
          measure_error=std::max(measure_error,std::abs(row-measure[i])/row);
          affine_load_error=std::max(affine_load_error,std::abs(initial_residual.shear_traction[0][i]-independent_shear[i])/row);
        }
      auto &bulk=sim.get_reconstructed_fault_stokes_coupling();bulk.linearize_B(base);
      LinearAlgebra::BlockVector Bdv(partition,comm);bulk.apply_B(dV,Bdv);
      const double Bwork=direction*Bdv;
      const double work_scale=std::max(std::abs(Bwork),std::abs(direct_B_work));
      const double work_error=std::max(std::abs(Bwork-Gshear_work),std::abs(Bwork-direct_B_work))/work_scale;
      const double initial_rms=retained_initial_rms;
      if (sim.get_pcout().is_active())
        {
          std::ofstream out(sim.get_output_directory()+"work_measure_checks.csv");
          out<<std::setprecision(17)<<"K_relative,G_relative,shear_work_relative,B_work,direct_work,G_shear_work,affine_point_error_Pa,affine_load_error_Pa,coefficient_error,row_measure_relative,owned_qps,positive_qps,intact_qps,wedge_qps,initial_surface_rms_Pa,top_initial_V,probe_old_shear_max_Pa\n"
             <<Kerror/Kscale<<','<<Gerror/Gscale<<','<<work_error<<','<<Bwork<<','<<direct_B_work<<','<<Gshear_work<<','
             <<affine_max<<','<<affine_load_error<<','<<coefficient_max<<','<<measure_error<<','<<visits<<','<<positive<<','<<intact<<','<<wedge<<','<<initial_rms<<','<<rates[0].back()<<','<<old_shear_max<<'\n';
          std::ofstream rows(sim.get_output_directory()+"work_initial_rows.csv");
          rows<<std::setprecision(17)<<"node,measure_m,residual_Pa\n";
          for (unsigned int i=0;i<V.size();++i) rows<<i<<','<<measure[i]<<','<<retained_initial.values[0][i]/measure[i]<<'\n';
        }
      AssertThrow(Kerror/Kscale<2e-7 && Gerror/Gscale<2e-7 && work_error<2e-10
                  && coefficient_max<1e-14 && affine_load_error<1e-6 && old_shear_max>1. && measure_error<1e-12
                  && visits==sim.get_triangulation().n_global_active_cells()*nq && intact>0 && wedge>0,
                  ExcMessage("Work-measure production derivative/work/coverage check failed."));
      sim.get_pcout()<<"Work-measure nonuniform free-top production checks passed; starting the bounded noncommitting coupled solve."<<std::endl;
    }
  }
}
