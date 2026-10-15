/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/simulator.h>
#include <aspect/postprocess/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/geometry_model/box.h>
#include <aspect/phase_field.h>
#include <aspect/utilities.h>
#include <aspect/reconstructed_fault/manager.h>
#include "../../../tests/phase_field_fault_test_access.h"
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace aspect
{
  namespace Postprocess
  {
    // Existing Simulator test friendship; this plugin is loaded independently
    // of the retained P2 MomentCycle observer. It performs no mechanical solve.
    template <int dim>
    class MomentCycle : public Interface<dim>, public SimulatorAccess<dim>
    {
        using F=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
        using Model=MaterialModel::PhaseFieldFault<dim>;
        using BV=LinearAlgebra::BlockVector;

        std::vector<double> independent_normalization(Model &model, const unsigned int order,
                                                      double &outside_max)
        {
          const auto profiles=F::normalization_profiles(model,true);
          const auto &cache=F::normalization_cells(model);
          AssertThrow(cache.supported && cache.profiles.size()==profiles.size(),
                      ExcMessage("Missing general Q1 cell intervals."));
          const QGauss<1> rule(order);
          std::vector<double> integrals(profiles.size(),0.);
          const auto integrate=[&](const double lo,const double hi,const auto &value)
          {
            double sum=0.;
            for (unsigned int q=0; q<rule.size(); ++q)
              sum+=rule.weight(q)*value(lo+(hi-lo)*rule.point(q)[0]);
            return (hi-lo)*sum;
          };
          for (unsigned int p=0; p<profiles.size(); ++p)
            {
              const auto &profile=profiles[p];
              const auto h=[&](double phi)
              {
                phi=F::normalization_effective_phase_field(phi);
                const double g=this->get_phase_field_handler().energetic_degradation(profile.material_fractions,phi);
                return F::normalization_integrand(phi,g);
              };
              // Fixed high-order independent integration of the actual bilinear
              // restriction, across complete cell intervals without adaptive tails.
              for (const auto &cell:cache.profiles[p].intervals)
                integrals[p]+=integrate(cell.lower,cell.upper,[&](const double r)
                {
                  const auto x=cell.reference_origin+r*cell.reference_direction;
                  double phi=0.;
                  for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                    {
                      double shape=1.;
                      for (unsigned int d=0; d<dim; ++d)
                        shape*=(v & (1u<<d)) ? x[d] : 1.-x[d];
                      phi+=shape*this->get_solution()[cell.phase_dofs[v]];
                    }
                  return h(phi);
                });
              double outside=0.;
              const auto &contacts=this->get_reconstructed_fault_manager().get_boundary_contacts();
              const auto &data=F::normalization_continuations(model);
              for (unsigned int c=0; c<contacts.size(); ++c)
                {
                  const auto &contact=contacts[c];
                  const double along=(profile.origin-contact.position)*contact.inward_tangent;
                  const double b=contact.inward_boundary_normal*profile.normal;
                  if (along<0. || along>contact.influence_length || b==0.)
                    continue;
                  const double cut=-(profile.origin-contact.position)*contact.inward_boundary_normal/b;
                  const double lo=b<0. ? std::max(-data[c].extent,cut) : -data[c].extent;
                  const double hi=b>0. ? std::min(data[c].extent,cut) : data[c].extent;
                  if (hi<=lo)
                    continue;
                  // Independently evaluate the prescribed exterior Q1 lattice
                  // with a fixed high-order rule, split at every lattice crossing.
                  std::vector<double> cuts {lo,hi};
                  for (unsigned int d=0; d<dim; ++d)
                    if (profile.normal[d]!=0.)
                      {
                        double a=profile.origin[d]+lo*profile.normal[d];
                        double z=profile.origin[d]+hi*profile.normal[d];
                        if (a>z) std::swap(a,z);
                        const long first=std::floor((a-data[c].lattice_origin[d])/data[c].spacing[d]);
                        const long last=std::ceil((z-data[c].lattice_origin[d])/data[c].spacing[d]);
                        for (long i=first; i<=last; ++i)
                          {
                            const double r=(data[c].lattice_origin[d]+i*data[c].spacing[d]-profile.origin[d])/profile.normal[d];
                            if (r>lo && r<hi) cuts.push_back(r);
                          }
                      }
                  std::sort(cuts.begin(),cuts.end());
                  cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
                  for (unsigned int j=1; j<cuts.size(); ++j)
                    outside+=integrate(cuts[j-1],cuts[j],[&](const double r)
                    {
                      const auto point=profile.origin+r*profile.normal;
                      Point<dim> lower;
                      Tensor<1,dim> fraction;
                      for (unsigned int d=0; d<dim; ++d)
                        {
                          lower[d]=data[c].lattice_origin[d]+data[c].spacing[d]
                                   *std::floor((point[d]-data[c].lattice_origin[d])/data[c].spacing[d]);
                          fraction[d]=(point[d]-lower[d])/data[c].spacing[d];
                        }
                      double phi=0.;
                      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                        {
                          auto node=lower;
                          double shape=1.;
                          for (unsigned int d=0; d<dim; ++d)
                            if (v & (1u<<d))
                              {
                                node[d]+=data[c].spacing[d];
                                shape*=fraction[d];
                              }
                            else shape*=1.-fraction[d];
                          phi+=shape*data[c].profile->value(std::abs((node-contact.position)*contact.normal));
                        }
                      return h(phi);
                    });
                }
              outside_max=std::max(outside_max,outside);
              integrals[p]+=outside;
            }
          return F::probe_normalization_projection(model,integrals);
        }

        void history_path(Model &model, Simulator<dim> &sim, const unsigned int mode,
                          std::ofstream &out)
        {
          auto &manager=this->get_reconstructed_fault_manager();
          auto &rates=const_cast<std::vector<double>&>(manager.get_slip_rate(0));
          const auto &fault=manager.get_fault(0);
          Tensor<1,dim> direction=fault.vertex(fault.n_cells())-fault.vertex(0);
          direction/=direction.norm();
          SymmetricTensor<2,dim> expected_shear;
          expected_shear[0][0]=-std::sqrt(3.)/4.;
          expected_shear[1][1]=std::sqrt(3.)/4.;
          expected_shear[0][1]=.25;
          Tensor<1,dim> perpendicular;
          perpendicular[0]=-direction[1];
          perpendicular[1]=direction[0];
          AssertThrow((manager.get_shear_sense(0)*symmetrize(outer_product(direction,perpendicular))-expected_shear).norm()<1.e-12,
                      ExcMessage("Translated 60 degree BP3 thrust orientation changed."));
          const auto &box=Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());
          const auto origin=box.get_origin(),extent=box.get_extents();
          struct Velocity : Function<dim>
          {
            Point<dim> origin,extent;
            unsigned int mode=0;
            Velocity(unsigned int n):Function<dim>(n) {}
            double value(const Point<dim> &p,const unsigned int c=0) const override
            {
              const double x=(p[0]-origin[0])/extent[0],y=(p[1]-origin[1])/extent[1];
              if (c==0) return 1.e-12*extent[0]*x*x*y;
              if (c==1) return (mode ?.5:-1.)*1.e-12*extent[1]*x*y*y;
              return 0.;
            }
          } velocity(this->get_fe().n_components());
          velocity.origin=origin;
          velocity.extent=extent;
          velocity.mode=mode;
          BV prescribed(this->get_solution());
          VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),velocity,prescribed,
                                   this->introspection().component_masks.velocities);
          AffineConstraints<double> hanging(this->introspection().index_sets.system_relevant_set);
          DoFTools::make_hanging_node_constraints(this->get_dof_handler(),hanging);
          hanging.close();
          BV owned(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
          owned=prescribed;
          hanging.distribute(owned);
          prescribed=owned;
          // The control changes only velocity. Reapplying generic hanging
          // constraints must not perturb the immutable physical phase anchor.
          const auto phase_block=this->introspection().variable("phase_field").block_index;
          prescribed.block(phase_block)=this->get_solution().block(phase_block);
          const auto [G,eta]=F::fe_material_coefficients(model);
          const double lambda=eta/G;
          unsigned int path_step=0;
          for (const double ratio:
          {
            1.e-12,1.e-6,.1,1.,4.,.25
          })
          {
            sim.timestep_number=++path_step;
            sim.time_step=ratio*lambda;
            const bool relax=path_step==6;
            if (relax) prescribed.block(this->introspection().block_indices.velocities)=0.;
            for (unsigned int j=0; j<rates.size(); ++j)
              // The terminal relaxation control retains the RSF admissible
              // minimum rate; exact zero is outside the existing aging law.
              rates[j]=relax ? model.get_fault_friction().get_minimum_slip_rate()
                       : 1.e-9*(1.+.35*std::sin(2.3*j/(rates.size()-1)));
            const auto incoming_Z=F::fe_slip_memory(model);
            const BV incoming(model.accepted_fe_strain_state());
            const auto coefficients=F::maxwell_coefficients(eta,G,sim.time_step);
            const auto candidate=F::prepared_fe_state(model,prescribed);
            // A rejected candidate is entirely scratch. Retry the original
            // rates and prove that both accepted histories remained immutable.
            const double first_rate=rates.front();
            rates.front()=std::numeric_limits<double>::quiet_NaN();
            bool invalid_rejected=false;
            try
              {
                F::prepared_fe_state(model,prescribed);
              }
            catch (const std::exception &)
              {
                invalid_rejected=true;
              }
            rates.front()=first_rate;
            AssertThrow(invalid_rejected && model.accepted_fe_strain_state()==incoming
                        && F::fe_slip_memory(model)==incoming_Z,ExcMessage("Rejected FE candidate changed accepted history."));
            double error=0.,allowance=0.,error_fraction=0.,trace_error=0.,trace_signal=0.,max_trace=0.;
            double slip_work=0.,bulk_form=0.;
            unsigned int count=0,continued=0;
            const auto check=[&](const auto &cell,const std::vector<Point<dim>> &points,
                                 const std::vector<double> *weights)
            {
              std::vector<Point<dim>> units;
              for (const auto &x:points) units.push_back(this->get_mapping().transform_real_to_unit_cell(cell,x));
              FEValues<dim> fe(this->get_mapping(),this->get_fe(),Quadrature<dim>(units),update_values|update_gradients);
              fe.reinit(cell);
              std::vector<SymmetricTensor<2,dim>> strain(points.size());
              std::vector<double> phase(points.size());
              fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(prescribed,strain);
              fe[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)]
              .get_function_values(prescribed,phase);
              for (unsigned int q=0; q<points.size(); ++q)
                {
                  const auto &x=points[q];
                  const auto source=manager.project_to_bulk_source(x);
                  double chiV=0.;
                  SymmetricTensor<2,dim> S;
                  if (source.active)
                    {
                      Tensor<1,dim> tangent=fault.vertex(source.segment_index+1)-fault.vertex(source.segment_index);
                      tangent/=tangent.norm();
                      Tensor<1,dim> normal;
                      normal[0]=-tangent[1];
                      normal[1]=tangent[0];
                      S=manager.get_shear_sense(0)*symmetrize(outer_product(tangent,normal));
                      const auto fractions=F::surface_material_state_at_projection(model,0,source.segment_index,source.xi).first;
                      const double phi=F::normalization_effective_phase_field(phase[q]);
                      const double h=F::normalization_integrand(phi,this->get_phase_field_handler().energetic_degradation(fractions,phi));
                      const auto &I=F::current_normalization_integrals(model).at(0);
                      const unsigned int j=source.segment_index;
                      chiV=h*manager.interpolate_slip_rate(0,j,source.xi)/((1-source.xi)*I[j]+source.xi*I[j+1]);
                      continued+=!manager.project_to_normal_profiles(x).active;
                    }
                  const auto old=model.retained_history_stress(cell->id(),x, {});
                  const auto expected=coefficients.first*old+2*coefficients.second
                                      *(Utilities::Tensors::consistent_deviator(strain[q])-chiV*S);
                  double scale=0.;
                  const auto actual=F::prepared_fe_stress(model,candidate,cell->id(),x,&scale);
                  error=std::max(error,(actual-expected).norm());
                  allowance=std::max(allowance,128*std::numeric_limits<double>::epsilon()*std::max(1.,scale));
                  error_fraction=std::max(error_fraction,(actual-expected).norm()
                                          /(128*std::numeric_limits<double>::epsilon()*std::max(1.,scale)));
                  if ((actual-expected).norm()>128*std::numeric_limits<double>::epsilon()*std::max(1.,scale))
                    {
                      std::ostringstream detail;
                      detail<<std::setprecision(17)<<"Recurrence mismatch: mode="<<mode
                            <<" dt/lambda="<<ratio<<" cell="<<cell->id()<<" x="<<x
                            <<" error="<<(actual-expected).norm()<<" scale="<<scale
                            <<" actual="<<actual<<" expected="<<expected
                            <<" old="<<old<<" strain="<<strain[q]
                            <<" phi="<<phase[q]<<" chiV="<<chiV<<" S="<<S;
                      this->get_pcout()<<detail.str()<<std::endl;
                    }
                  AssertThrow((actual-expected).norm()<=128*std::numeric_limits<double>::epsilon()*std::max(1.,scale),
                              ExcMessage("Mature FE recurrence differs from direct Maxwell evaluation."));
                  const auto full=coefficients.first*old+2*coefficients.second*(strain[q]-chiV*S);
                  const auto difference=actual-full;
                  const auto predicted=-2*coefficients.second*trace(strain[q])/3.*unit_symmetric_tensor<dim>();
                  trace_error=std::max(trace_error,(difference-predicted).norm());
                  trace_signal=std::max(trace_signal,predicted.norm());
                  max_trace=std::max(max_trace,std::abs(trace(strain[q])));
                  AssertThrow((difference-predicted).norm()<=2*128*std::numeric_limits<double>::epsilon()*std::max(1.,scale),
                              ExcMessage("Tensor convention difference is not the derived trace term."));
                  if (weights)
                    {
                      slip_work+=(*weights)[q]*chiV*(S*strain[q]);
                      bulk_form+=(*weights)[q]*(Utilities::Tensors::consistent_deviator(strain[q])*strain[q]);
                    }
                  ++count;
                }
            };
            const auto rule=model.mechanical_quadrature();
            FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_quadrature_points|update_JxW_values);
            manager.prepare_stokes_qp_projection_cache();
            for (const auto &cell:this->get_dof_handler().active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  fe.reinit(cell);
                  const auto &associations=manager.get_stokes_qp_fault_associations(cell->id(),rule,fe.get_quadrature_points());
                  for (unsigned int q=0; q<rule.size(); ++q)
                    {
                      const auto source=manager.project_to_bulk_source(fe.quadrature_point(q));
                      AssertThrow(source.active==associations[q].active
                                  && (!source.active || (source.segment_index==associations[q].segment_index
                                                         && source.xi==associations[q].xi)),
                                  ExcMessage("Mechanical and FE history source associations differ."));
                    }
                  check(cell,fe.get_quadrature_points(),&fe.get_JxW_values());
                }
            auto &particles=this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler();
            for (const auto &particle:particles)
              {
                // Particle cells are triangulation iterators. Field evaluation
                // needs the corresponding DoF iterator, including hanging DoFs.
                const auto cell=particle.get_surrounding_cell();
                const typename DoFHandler<dim>::active_cell_iterator dof_cell(
                  &this->get_triangulation(),cell->level(),cell->index(),&this->get_dof_handler());
                check(dof_cell, {particle.get_location()},nullptr);
              }
            AssertThrow(model.accepted_fe_strain_state()==incoming && F::fe_slip_memory(model)==incoming_Z,
                        ExcMessage("Preparing FE trials advanced accepted history."));
            model.commit_reconstructed_fault_mechanical_history(prescribed);
            double memory_error=0.,memory_scale=1.;
            for (unsigned int j=0; j<rates.size(); ++j)
              {
                const double expected=coefficients.first*incoming_Z[j]+coefficients.second*rates[j];
                memory_error=std::max(memory_error,std::abs(F::fe_slip_memory(model)[j]-expected));
                memory_scale=std::max(memory_scale,std::abs(expected));
                AssertThrow(F::committed_q(model,0,std::min(j,fault.n_cells()-1),j==fault.n_cells() ? 1. : 0.)==0.,
                            ExcMessage("Mature cohesive traction is not zero."));
              }
            AssertThrow(memory_error<=64*std::numeric_limits<double>::epsilon()*memory_scale
                        && *std::max_element(F::fe_slip_memory(model).begin(),F::fe_slip_memory(model).end())>0.,
                        ExcMessage("Mature slip memory was lost."));
            // A duplicate terminal commit must fail before changing any field.
            const BV committed(model.accepted_fe_strain_state());
            const auto committed_Z=F::fe_slip_memory(model);
            bool rejected=false;
            try
              {
                model.commit_reconstructed_fault_mechanical_history(prescribed);
              }
            catch (const std::exception &)
              {
                rejected=true;
              }
            AssertThrow(rejected && model.accepted_fe_strain_state()==committed
                        && F::fe_slip_memory(model)==committed_Z,ExcMessage("Duplicate history publication was not rejected."));
            out<<mode<<','<<ratio<<','<<sim.time_step<<','<<count<<','<<continued<<','<<error<<','<<allowance
               <<','<<trace_error<<','<<trace_signal<<','<<max_trace<<','<<memory_error<<','<<memory_scale
               <<','<<slip_work<<','<<bulk_form<<','<<error_fraction<<'\n';
          }
        }

      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          AssertThrow(dim==2 && this->get_timestep_number()==0
                      && Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1,
                      ExcMessage("History checks are fresh serial stage-only controls."));
          auto &model=const_cast<Model &>(Plugins::get_plugin_as_type<const Model>(this->get_material_model()));
          auto &sim=const_cast<Simulator<dim>&>(this->get_simulator());
          model.prepare_reconstructed_fault_mechanical_solve();
          const auto I=F::current_normalization_integrals(model).at(0);
          double outside16=0.,outside32=0.;
          const auto I16=independent_normalization(model,16,outside16);
          const auto I32=independent_normalization(model,32,outside32);
          double error=0.,reference_error=0.,minimum=*std::min_element(I.begin(),I.end());
          for (unsigned int j=0; j<I.size(); ++j)
            {
              error=std::max(error,std::abs(I[j]-I32[j])/I32[j]);
              reference_error=std::max(reference_error,std::abs(I16[j]-I32[j])/I32[j]);
            }
          AssertThrow(minimum>0. && error<=1.e-9 && reference_error<=1.e-10 && outside32>0.,
                      ExcMessage("Oblique completed normalization disagrees with independent quadrature."));
          std::ofstream normalization(this->get_output_directory()+"native-normalization.csv");
          normalization<<std::setprecision(17)<<"min_I_m,relative_error,reference_error,outside_max_m,cells,constrained_dofs\n"
                       <<minimum<<','<<error<<','<<reference_error<<','<<outside32<<','<<this->get_triangulation().n_global_active_cells()
                       <<','<<this->get_current_constraints().n_constraints()<<'\n';
          auto &manager=this->get_reconstructed_fault_manager();
          const auto rates=manager.get_slip_rate(0);
          const double dt=sim.time_step;
          const auto step=sim.timestep_number;
          auto &particles=this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler();
          std::map<types::particle_index,std::vector<double>> saved_particles;
          for (const auto &p:particles) saved_particles[p.get_id()]= {p.get_properties().begin(),p.get_properties().end()};
          std::vector<std::vector<double>> properties;
          for (unsigned int j=0; j<manager.get_fault(0).n_vertices(); ++j)
            {
              const auto values=manager.get_fault(0).get_properties(j);
              properties.emplace_back(values.begin(),values.end());
            }
          const auto restore_fe=F::save_fe_history(model);
          auto first_particle=particles.begin();
          const auto position=first_particle->get_location();
          auto moved=position;
          moved[0]+=1.e-6;
          first_particle->set_location(moved);
          bool transport_rejected=false;
          try
            {
              model.prepare_reconstructed_fault_mechanical_solve();
            }
          catch (const std::exception &)
            {
              transport_rejected=true;
            }
          first_particle->set_location(position);
          AssertThrow(transport_rejected,ExcMessage("Mature FE history admitted particle transport."));
          const auto restore=[&]()
          {
            restore_fe();
            sim.time_step=dt;
            sim.timestep_number=step;
            const_cast<std::vector<double>&>(manager.get_slip_rate(0))=rates;
            for (auto &p:particles) std::copy(saved_particles.at(p.get_id()).begin(),saved_particles.at(p.get_id()).end(),p.get_properties().begin());
            for (unsigned int j=0; j<properties.size(); ++j)
              std::copy(properties[j].begin(),properties[j].end(),manager.get_fault(0).get_properties(j).begin());
          };
          std::ofstream out(this->get_output_directory()+"native-history.csv");
          out<<std::setprecision(17)<<"mode,dt_over_lambda,dt_s,samples,continued_samples,stress_error_Pa,allowance_Pa,trace_error_Pa,trace_difference_Pa,max_divergence_per_s,Z_error_Pa_m,Z_scale_Pa_m,slip_work_form,bulk_strain_form,error_fraction_of_allowance\n";
          try
            {
              for (unsigned int mode=0; mode<2; ++mode)
                {
                  history_path(model,sim,mode,out);
                  restore();
                }
            }
          catch (...)
            {
              restore();
              throw;
            }
          this->get_pcout()<<"   BP3 native history/geometry checks passed; diagnostic publications restored, no coupled solve."<<std::endl;
          return {"BP3 native history:","passed"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"BP3 native history checks",
                                  "Stage-only mature FE recurrence, oblique normalization and tensor convention checks. No coupled trajectory.")
  }
}
