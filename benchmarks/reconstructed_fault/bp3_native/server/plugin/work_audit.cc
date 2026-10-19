#include "runtime.h"
#include "bp3_model.h"
#include "output_files.h"

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <aspect/plugins.h>
#include <aspect/utilities.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

#include <fstream>
#include <iomanip>
#include <sstream>
// Accepted-state observations. Never refresh FE histories or update stress here.
namespace aspect
{
  namespace BP3Benchmark
  {
    // Fixed-profile audit baseline: capture once at initialization and serialize
    // it with the benchmark history, never recapture from a restarted state.
    std::vector<Point<2>> work_initial_geometry;
    std::vector<double> work_initial_I;

    template <int dim>
    void capture_work_invariants(const SimulatorAccess<dim> &sim)
    {
      work_initial_geometry.clear();
      work_initial_I.clear();
      const auto &manager=sim.get_reconstructed_fault_manager();
      const auto &fault=manager.get_fault(0);
      const auto I=manager.get_property_information()[manager.get_property_index("phase field fault previous I h")].position;
      for (unsigned int j=0; j<fault.n_vertices(); ++j)
        {
          Point<2> p;
          p[0]=fault.vertex(j)[0];
          p[1]=fault.vertex(j)[1];
          work_initial_geometry.push_back(p);
          work_initial_I.push_back(fault.get_properties(j)[I]);
        }
    }

    template <int dim>
    void verify_accepted_work(
      const SimulatorAccess<dim> &sim,const ReconstructedFaultSurfaceResidual &weak,const bool write_files)
    {
      const auto &intro=sim.introspection();
      const auto comm=sim.get_mpi_communicator();
      auto &manager=sim.get_reconstructed_fault_manager();
      const auto &fault=manager.get_fault(0);
      const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model());
      const auto &working=sim.get_current_linearization_point();
      const auto &V=manager.get_timestep_committed_slip_rate(0);
      const unsigned int step=sim.get_timestep_number(),n=V.size();
      const unsigned int rank=Utilities::MPI::this_mpi_process(comm);
      const auto &pm=sim.get_phase_field_handler().get_associated_particle_manager();
      const auto &particles=pm.get_particle_handler();
      const auto &info=pm.get_property_manager().get_data_info();
      const auto H=info.get_position_by_field_name("crack_driving_force");
      const auto stress_position=info.get_position_by_field_name("maxwell stress");
      std::array<unsigned int,3> fields{};
      const auto stress_fields=intro.get_indices_for_fields_of_type(CompositionalFieldDescription::stress);
      AssertThrow(stress_fields.size()==fields.size(),ExcMessage("BP3 work audit requires three stress fields."));
      std::copy(stress_fields.begin(),stress_fields.end(),fields.begin());
      for (const auto &m:sim.get_parameters().mapped_particle_properties)
        if (m.second.first=="maxwell stress") fields[m.second.second]=m.first;
      const auto I=manager.get_property_information()[manager.get_property_index("phase field fault previous I h")].position;
      // Point/domain diagnostics own a segment index rather than a cached
      // Stokes association. Use that segment's native frame, as in DG stress
      // reconstruction. Mechanical QPs below use the actual cached frame.
      const auto segment_frame=[&](const unsigned int segment)
      {
        auto tangent=fault.vertex(segment+1)-fault.vertex(segment);
        tangent/=tangent.norm();
        Tensor<1,dim> normal;
        normal[0]=-tangent[1];
        normal[1]=tangent[0];
        return std::make_pair(manager.get_shear_sense(0)*symmetrize(outer_product(tangent,normal)),
                              symmetrize(outer_product(normal,normal)));
      };
      // Retain the old first-segment projection only as a counterfactual trace.
      // It must not determine the accepted-work guard or exported work loads.
      const auto first_frame=segment_frame(0);

      // Particle baselines follow native births/migration/rollback. Geometry
      // and completed profile retain the original timestep-zero baseline.
      if (step==0 && work_initial_geometry.empty()) capture_work_invariants(sim);
      const auto &initial_H=work_initial_H;
      const auto &initial_geometry=work_initial_geometry;
      const auto &initial_I=work_initial_I;
      bool history_ok=true;
      for (const auto &p:particles)
        {
          const auto old=initial_H.find(p.get_id());
          history_ok=history_ok && old!=initial_H.end() && old->second==p.get_properties()[H];
        }
      for (unsigned int j=0; j<n; ++j)
        history_ok=history_ok && initial_geometry[j][0]==fault.vertex(j)[0]
                   && initial_geometry[j][1]==fault.vertex(j)[1] && initial_I[j]==fault.get_properties(j)[I];
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(history_ok),comm),
                  ExcMessage("Work replay changed inert H, geometry or completed I_h."));

      struct Sample
      {
        SymmetricTensor<2,dim> tau;
        double chi,eta_ve,rate,bg,bg_normal,q,sigma,Ih;
      };
      const auto evaluate=[&](const unsigned int segment,const double xi,const bool active,
                              const double phi,const double temp,const double pressure,
                              const SymmetricTensor<2,dim> &eps,const std::vector<double> &all,
                              const SymmetricTensor<2,dim> &published_fe_stress,
                              const SymmetricTensor<2,dim> &S,const SymmetricTensor<2,dim> &N)
      {
        std::vector<double> chemical;
        for (const auto c:intro.chemical_composition_field_indices()) chemical.push_back(all[c]);
        typename MaterialModel::PhaseFieldFault<dim>::ReconstructedFaultBulkPointInputs in;
        in.fault_index=0;
        in.segment_index=segment;
        in.xi=xi;
        in.phase_field=phi;
        in.previous_phase_field=phi;
        in.temperature=temp;
        in.bulk_material_fractions=MaterialModel::MaterialUtilities::compute_composition_fractions(chemical);
        const auto r=model.evaluate_reconstructed_fault_bulk_point(in);
        AssertThrow(r.history_correction==0.,ExcMessage("Mature frozen-profile history correction changed."));
        Sample result;
        result.chi=active ? r.localization_factor : 0.;
        result.eta_ve=r.eta_ve;
        result.rate=manager.interpolate_slip_rate(0,segment,xi);
        SymmetricTensor<2,dim> old;
        old[0][0]=all[fields[0]];
        old[1][1]=all[fields[1]];
        old[0][1]=all[fields[2]];
        const auto strain=model.uses_deviatoric_strain_rate() ? Utilities::Tensors::consistent_deviator(eps) : eps;
        // This observer runs after terminal publication. FE samples already
        // contain this Maxwell interval; never advance that state a second time.
        result.tau=model.uses_independent_bulk_history() && step>0 ? published_fe_stress
                   : 2*r.eta_ve*(strain-result.chi*result.rate*S)+model.evaluate_frozen_maxwell_stress(temp,all,old);
        const auto bg=model.reconstructed_fault_background_tractions(0,segment,xi);
        result.bg=bg.first;
        result.bg_normal=bg.second;
        result.q=bg.first+result.tau*S;
        result.sigma=bg.second+pressure-result.tau*N;
        result.Ih=(1-xi)*fault.get_properties(segment)[I]+xi*fault.get_properties(segment+1)[I];
        return result;
      };

      // Native work diagnostics use the accepted velocity/pressure and still
      // frozen working FE history, not the newly committed particle stress.
      const auto quad=fault_mechanical_quadrature(sim);
      FEValues<dim> fe(sim.get_mapping(),sim.get_fe(),quad,
                       update_values|update_gradients|update_quadrature_points|update_JxW_values);
      const unsigned int nq=quad.size();
      std::vector<double> phi(nq),temp(nq),pressure(nq);
      std::vector<SymmetricTensor<2,dim>> eps(nq);
      std::vector<std::vector<double>> composition(intro.n_compositional_fields,std::vector<double>(nq));
      std::vector<double> native(6*n),common(write_files ? 6*n : 0);
      // The authoritative native array now uses the mechanical QP frames.
      // Retain the old first-segment projections and direct frame differences
      // separately so the repaired guard remains independently diagnosable.
      std::vector<double> reference_frame(4*n);
      std::ofstream raw;
      if (write_files)
        {
          raw.open(sim.get_output_directory()+"work_qp_"+std::to_string(step)+"_rank"+std::to_string(rank)+".csv");
          raw<<std::setprecision(17)<<"cell,qp,x,y,xd,r,JxW,source_active,segment,xi,phi,Ih,chi,V,p,tau_xx,tau_yy,tau_xy,tauN,sigma_n,q,eps_xx,eps_yy,eps_xy,elastic_norm\n";
        }
      const auto accumulate=[](std::vector<double> &v,unsigned int j,double xi,double weight,double p,const Sample &s,double tauN)
      {
        const double values[]= {1.,p,tauN,s.q,s.sigma,s.bg};
        for (unsigned int end=0; end<2; ++end)
          for (unsigned int c=0; c<6; ++c) v[6*(j+end)+c]+=weight*(end ? xi : 1-xi)*values[c];
      };
      for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe.reinit(cell);
            const auto &a=manager.get_stokes_qp_fault_associations(cell->id(),quad,fe.get_quadrature_points());
            const auto *retained=model.uses_stationary_mature_dg_optimizations()
              ? &model.retained_dg_qp_stress(cell->id(),quad,fe.get_quadrature_points()) : nullptr;
            fe[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)].get_function_values(working,phi);
            fe[intro.extractors.temperature].get_function_values(working,temp);
            fe[intro.extractors.pressure].get_function_values(working,pressure);
            fe[intro.extractors.velocities].get_function_symmetric_gradients(working,eps);
            for (unsigned int c=0; c<composition.size(); ++c) fe[intro.extractors.compositional_fields[c]].get_function_values(working,composition[c]);
            for (unsigned int q=0; q<nq; ++q)
              {
                const auto p=fe.quadrature_point(q);
                const double xd=BP3::down_dip(p[0],p[1]);
                const bool window=(xd>=BP3::geometry().weakening_length-2000. && xd<=BP3::geometry().weakening_length+5000.)
                                  || xd<2000. || xd>BP3::geometry().length-2000.;
                if (!a[q].active && !(write_files && phi[q]>0. && window)) continue;
                std::vector<double> all(composition.size());
                for (unsigned int c=0; c<all.size(); ++c) all[c]=composition[c][q];
                const auto j=a[q].active ? a[q].segment_index : 0u;
                const auto xi=a[q].active ? a[q].xi : 0.;
                const auto published=retained ? (*retained)[q] : (model.uses_independent_bulk_history()
                                     ? model.retained_history_stress(cell->id(),p, {}) : SymmetricTensor<2,dim>());
                const auto S=a[q].active ? manager.get_shear_sense(a[q].fault_index)
                                          *symmetrize(outer_product(a[q].tangent,a[q].normal)) : first_frame.first;
                const auto N=a[q].active ? symmetrize(outer_product(a[q].normal,a[q].normal)) : first_frame.second;
                const auto s=evaluate(j,xi,a[q].active,phi[q],temp[q],pressure[q],eps[q],all,published,S,N);
                if (a[q].active && s.chi>0.)
                  {
                    accumulate(native,j,a[q].shape_1,fe.JxW(q)*s.chi,pressure[q],s,s.tau*N);
                    const double values[]={s.bg+s.tau*first_frame.first,
                                           s.bg_normal+pressure[q]-s.tau*first_frame.second,
                                           s.tau*(first_frame.first-S),-s.tau*(first_frame.second-N)};
                    for (unsigned int end=0; end<2; ++end)
                      for (unsigned int c=0; c<4; ++c)
                        reference_frame[4*(j+end)+c]+=fe.JxW(q)*s.chi
                                                   *(end ? a[q].shape_1 : 1-a[q].shape_1)*values[c];
                  }
                if (write_files && window && phi[q]>0.)
                  {
                    raw<<cell->id().to_string()<<','<<q<<','<<p[0]<<','<<p[1]<<','<<xd<<','
                       <<BP3::signed_normal(p[0],p[1])<<','<<fe.JxW(q)<<','
                       <<a[q].active<<','<<j<<','<<xi<<','<<phi[q]<<','<<s.Ih<<','<<s.chi<<','<<s.rate<<','<<pressure[q]<<','
                       <<s.tau[0][0]<<','<<s.tau[1][1]<<','<<s.tau[0][1]<<','<<s.tau *N<<','<<s.sigma<<','<<s.q<<','
                       <<eps[q][0][0]<<','<<eps[q][1][1]<<','<<eps[q][0][1]<<','<<(eps[q]-s.chi*s.rate*S).norm()<<'\n';
                  }
              }
          }

      if (write_files)
        {
          raw.close();
          AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(bool(raw)),comm),
                      ExcMessage("Cannot write native work quadrature diagnostics."));
        }

      // Particle point evaluation is required for the first Maxwell check only.
      // Later common-FE/parent-P0 comparisons are diagnostic, not mechanics.
      if (write_files || step==1)
        {
          const auto &associations=manager.get_locally_owned_particle_fault_associations();
          std::vector<Point<dim>> points;
          for (const auto &p:particles) points.push_back(p.get_location());
          Utilities::MPI::RemotePointEvaluation<dim> cache;
          cache.reinit(sim.get_phase_field_handler().get_grid_cache(),points);
          const auto gradients=VectorTools::point_gradients<dim>(cache,sim.get_dof_handler(),working,VectorTools::EvaluationFlags::avg,intro.component_indices.velocities[0]);
          const auto ph=VectorTools::point_values<1>(cache,sim.get_dof_handler(),working,VectorTools::EvaluationFlags::avg,intro.variable("phase_field").first_component_index);
          const auto T=VectorTools::point_values<1>(cache,sim.get_dof_handler(),working,VectorTools::EvaluationFlags::avg,intro.component_indices.temperature);
          const auto p=VectorTools::point_values<1>(cache,sim.get_dof_handler(),working,VectorTools::EvaluationFlags::avg,intro.component_indices.pressure);
          std::vector<std::vector<double>> comps(intro.n_compositional_fields);
          for (unsigned int c=0; c<comps.size(); ++c)
            comps[c]=VectorTools::point_values<1>(cache,sim.get_dof_handler(),working,VectorTools::EvaluationFlags::avg,intro.component_indices.compositional_fields[c]);
          unsigned int k=0;
          double first_error=0.,first_scale=0.,first_old=0.;
          for (const auto &particle:particles)
            {
              Assert(particle.get_id()==associations[k].particle_id,ExcInternalError());
              std::vector<double> all(comps.size());
              for (unsigned int c=0; c<all.size(); ++c) all[c]=comps[c][k];
              if (write_files)
                for (const auto &q:associations[k].quadrature)
                  {
                    const auto published=model.uses_independent_bulk_history()
                                         ? model.retained_history_stress(particle.get_surrounding_cell()->id(),particle.get_location(), {})
                                         : SymmetricTensor<2,dim>();
                    const auto frame=segment_frame(q.segment_index);
                    const auto s=evaluate(q.segment_index,q.xi,true,ph[k],T[k],p[k],symmetrize(gradients[k]),all,published,
                                          frame.first,frame.second);
                    accumulate(common,q.segment_index,q.xi,q.weight,p[k],s,s.tau*frame.second);
                  }
              if (step==1)
                {
                  const auto a=manager.project_to_bulk_source(particle.get_location());
                  const auto frame=segment_frame(a.active ? a.segment_index : 0u);
                  const auto s=evaluate(a.active ? a.segment_index : 0u,a.active ? a.xi : 0.,a.active,
                                        ph[k],T[k],p[k],symmetrize(gradients[k]),all, {},frame.first,frame.second);
                  // Timestep zero retained zero particle stress. This independent
                  // first finite-step formula must match the newly committed array.
                  const auto strain=model.uses_deviatoric_strain_rate()
                                    ? Utilities::Tensors::consistent_deviator(symmetrize(gradients[k])) : symmetrize(gradients[k]);
                  const auto expected=2*s.eta_ve*(strain-s.chi*s.rate*frame.first);
                  for (unsigned int c=0; c<3; ++c)
                    {
                      first_old=std::max(first_old,std::abs(all[fields[c]]));
                      const double actual=particle.get_properties()[stress_position+c];
                      first_error=std::max(first_error,std::abs(actual-expected[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]));
                      first_scale=std::max(first_scale,std::abs(actual));
                    }
                }
              ++k;
            }
          if (step==1)
            {
              first_error=Utilities::MPI::max(first_error,comm);
              first_scale=Utilities::MPI::max(first_scale,comm);
              first_old=Utilities::MPI::max(first_old,comm);
              AssertThrow(first_old==0. && first_scale>0. && first_error<=1e-6+1e-11*first_scale,
                          ExcMessage("Work replay first Maxwell publication disagrees with the retained-zero-history update."));
              BP3::collective_root_write(comm,[&]()
              {
                std::ofstream gate(sim.get_output_directory()+"first_update_maxwell.csv");
                gate.exceptions(std::ios::failbit|std::ios::badbit);
                gate<<std::setprecision(17)<<"error_Pa,scale_Pa,old_FE_max_Pa,stable_H_ids\n"
                    <<first_error<<','<<first_scale<<','<<first_old<<','<<initial_H.size()<<'\n';
                gate.close();
              });
            }

          if (write_files)
            {
              const auto local=common;
              Utilities::MPI::sum(local,comm,common);
            }
        }

      const auto local_native=native;
      Utilities::MPI::sum(local_native,comm,native);
      const auto local_reference_frame=reference_frame;
      Utilities::MPI::sum(local_reference_frame,comm,reference_frame);
#ifdef ASPECT_BP3_TEST_WORK_CONSISTENCY_FAILURE
      // Negative diagnostic-only fixture: never alters mechanics or history.
      // Every rank owns the same reduced arrays, so the original guard and
      // collective failure export must reject this inconsistent observation.
      if (step==2) native[3]+=2e-5*native[0];
#endif
      double consistency=0.;
      for (unsigned int j=0; j<n; ++j)
        {
          consistency=std::max(consistency,std::abs(native[6*j+3]-weak.shear_traction[0][j])/native[6*j]);
          // Filtering changes the friction input, not the mechanical stress
          // reconstructed by this observer. Compare its raw load with raw
          // assembly; the unfiltered path retains its original check.
          const auto &raw_normal=weak.raw_normal_traction.empty() ? weak.normal_traction : weak.raw_normal_traction;
          consistency=std::max(consistency,std::abs(native[6*j+4]-raw_normal[0][j])/native[6*j]);
        }
      const auto &raw_normal=weak.raw_normal_traction.empty() ? weak.normal_traction : weak.raw_normal_traction;
      std::array<double,6> maxima{};
      std::array<unsigned int,6> worst{};
      for (unsigned int j=0; j<n; ++j)
        {
          const double errors[]={std::abs(reference_frame[4*j]-weak.shear_traction[0][j]),
                                 std::abs(reference_frame[4*j+1]-raw_normal[0][j]),
                                 std::abs(native[6*j+3]-weak.shear_traction[0][j]),
                                 std::abs(native[6*j+4]-raw_normal[0][j]),
                                 std::abs(reference_frame[4*j+2]),std::abs(reference_frame[4*j+3])};
          for (unsigned int c=0; c<maxima.size(); ++c)
            if (errors[c]/native[6*j]>maxima[c])
              {
                maxima[c]=errors[c]/native[6*j];
                worst[c]=j;
              }
        }
      std::ostringstream trace;
      trace<<std::setprecision(17)<<"Work consistency trace: step="<<step
           <<", time_s="<<sim.get_time()<<", dt_s="<<sim.get_timestep()
           <<", original_shear_Pa="<<maxima[0]<<", node="<<worst[0]
           <<", original_raw_normal_Pa="<<maxima[1]<<", node="<<worst[1]
           <<", matching_shear_Pa="<<maxima[2]<<", node="<<worst[2]
           <<", matching_raw_normal_Pa="<<maxima[3]<<", node="<<worst[3]
           <<", direct_frame_shear_Pa="<<maxima[4]<<", node="<<worst[4]
           <<", direct_frame_raw_normal_Pa="<<maxima[5]<<", node="<<worst[5]
           <<", bound_Pa=1e-5, guard_uses_matching_frame=1";
      sim.get_pcout()<<trace.str()<<std::endl;
      // All ranks must take the same branch before collective_root_write.
      if (Utilities::MPI::max(static_cast<unsigned int>(!(consistency<1e-5)),comm))
        BP3::collective_root_write(comm,[&]()
        {
          std::ofstream out(sim.get_output_directory()+"work_consistency_failure_"+std::to_string(step)+".csv");
          out.exceptions(std::ios::failbit|std::ios::badbit);
          out<<std::setprecision(17)
             <<"step,time_s,dt_s,node,xd_m,mass_row_m,mechanical_shear_Pa_m,mechanical_raw_normal_Pa_m,"
             <<"original_shear_Pa_m,original_raw_normal_Pa_m,matching_shear_Pa_m,matching_raw_normal_Pa_m,"
             <<"direct_frame_shear_Pa_m,direct_frame_raw_normal_Pa_m,original_shear_error_Pa,"
             <<"original_raw_normal_error_Pa,matching_shear_error_Pa,matching_raw_normal_error_Pa\n";
          for (unsigned int j=0; j<n; ++j)
            out<<step<<','<<sim.get_time()<<','<<sim.get_timestep()<<','<<j<<','
               <<BP3::down_dip(fault.vertex(j)[0],fault.vertex(j)[1])<<','<<native[6*j]<<','
               <<weak.shear_traction[0][j]<<','<<raw_normal[0][j]<<','<<reference_frame[4*j]<<','<<reference_frame[4*j+1]<<','
               <<native[6*j+3]<<','<<native[6*j+4]<<','<<reference_frame[4*j+2]<<','<<reference_frame[4*j+3]<<','
               <<(reference_frame[4*j]-weak.shear_traction[0][j])/native[6*j]<<','
               <<(reference_frame[4*j+1]-raw_normal[0][j])/native[6*j]<<','
               <<(native[6*j+3]-weak.shear_traction[0][j])/native[6*j]<<','
               <<(native[6*j+4]-raw_normal[0][j])/native[6*j]<<'\n';
          out.close();
        });
      AssertThrow(consistency<1e-5,ExcMessage("Accepted work observer does not reproduce frozen mechanical traction."));
      if (write_files)
        BP3::collective_root_write(comm,[&]()
        {
          for (unsigned int mode=0; mode<2; ++mode)
            {
              const auto &v=mode==0 ? native : common;
              std::ofstream out(sim.get_output_directory()+(mode==0 ? "work_weak_" : "common_fe_weak_")+std::to_string(step)+".csv");
              out.exceptions(std::ios::failbit|std::ios::badbit);
              out<<std::setprecision(17)<<"node,xd,weight,p,tauN,q,sigma,bg\n";
              for (unsigned int j=0; j<n; ++j)
                {
                  out<<j<<','<<BP3::down_dip(fault.vertex(j)[0],fault.vertex(j)[1]);
                  for (unsigned int c=0; c<6; ++c)out<<','<<v[6*j+c];
                  out<<'\n';
                }
              out.close();
            }
          sim.get_pcout()<<"Work replay accepted observation: step="<<step<<", frozen weak stress discrepancy="<<consistency
                         <<" Pa; inert H and geometry unchanged."<<std::endl;
        });
    }
  }
}

namespace aspect
{
  template void BP3Benchmark::verify_accepted_work(const SimulatorAccess<2> &, const ReconstructedFaultSurfaceResidual &, const bool);
  template void BP3Benchmark::verify_accepted_work(const SimulatorAccess<3> &, const ReconstructedFaultSurfaceResidual &, const bool);
}
