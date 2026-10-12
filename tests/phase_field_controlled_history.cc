// P1 diagnostic only: production equality operator with externally prescribed H.
// Register the inherited fixture's postprocessor name, but do not execute it.
#include "phase_field_fault_ih.cc"
#include "phase_field_test_access.h"
#include <aspect/particle/particle_domain.h>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class ControlledPhaseHistory : public Interface<dim>, public SimulatorAccess<dim>
    {
      using VectorType = LinearAlgebra::BlockVector;
      using Access = internal::PhaseFieldTestAccess<dim>;

      // No transport occurs inside this probe: property addresses remain stable.
      struct RestoreH
      {
        std::vector<std::pair<double *, double>> entries;
        ~RestoreH() { for (const auto &entry : entries) *entry.first = entry.second; }
      };

      struct RestoreVector
      {
        VectorType &live;
        VectorType saved;
        explicit RestoreVector(VectorType &v) : live(v), saved(v) {}
        ~RestoreVector() { live = saved; }
      };

      // Bitwise comparison preserves NaNs in not-yet-initialized fault properties.
      std::string fingerprint() const
      {
        std::string result;
        const auto add = [&result](const auto &value)
        { result.append(reinterpret_cast<const char *>(&value), sizeof(value)); };
        for (const VectorType *v : {&this->get_solution(), &this->get_old_solution(),
                                   &this->get_system_rhs()})
          for (unsigned int b=0; b<v->n_blocks(); ++b)
            for (const auto i : v->block(b).locally_owned_elements()) add(v->block(b)[i]);
        const auto &pm = this->get_phase_field_handler().get_associated_particle_manager();
        for (const auto &p : pm.get_particle_handler())
          {
            add(p.get_id());
            for (unsigned int d=0; d<dim; ++d) add(p.get_location()[d]);
            for (const double value : p.get_properties()) add(value);
          }
        add(pm.get_particle_domain_handler().geometry_version());
        for (const auto &fault : this->get_reconstructed_fault_manager().get_faults())
          {
            add(fault.geometry_version());
            add(fault.n_vertices());
            for (unsigned int v=0; v<fault.n_vertices(); ++v)
              {
                for (unsigned int d=0; d<dim; ++d) add(fault.vertex(v)[d]);
                for (const double value : fault.get_properties(v)) add(value);
              }
          }
        return result;
      }

      std::vector<double> matrix_entries(const LinearAlgebra::BlockSparseMatrix &matrix,
                                         const unsigned int b) const
      {
        std::vector<double> entries;
        for (const auto row : this->introspection().index_sets.system_partitioning[b])
          for (auto p=matrix.block(b,b).begin(row); p!=matrix.block(b,b).end(row); ++p)
            entries.push_back(p->value());
        return entries;
      }

      struct Measures
      {
        double node_min=1e100, node_max=-1e100, cpdi_min=1e100, cpdi_max=-1e100;
        double change=0, healing=0, integral=0;
        Point<dim> minimum_location, healing_location, cpdi_minimum_location;
        double H_at_cpdi_minimum=0;
        unsigned int negative_nodes=0, negative_particles=0;
      };

      Measures measure(const VectorType &state, const VectorType &previous) const
      {
        Measures m;
        const auto &handler = this->get_phase_field_handler();
        const auto &pm = handler.get_associated_particle_manager();
        const auto &dofs = Access::vertex_dofs(handler);
        const unsigned int b = this->introspection().variable("phase_field").block_index;
        for (const auto i : state.block(b).locally_owned_elements())
          {
            const double phi = state.block(b)[i];
            m.node_min=std::min(m.node_min,phi); m.node_max=std::max(m.node_max,phi);
            m.change=std::max(m.change,std::abs(phi-previous.block(b)[i]));
            m.healing=std::max(m.healing,previous.block(b)[i]-phi);
            m.negative_nodes+=(phi<0);
          }
        const auto &vertices=this->get_triangulation().get_vertices();
        for (unsigned int v=0; v<dofs.size(); ++v)
          if (dofs[v]!=numbers::invalid_dof_index && state.locally_owned_elements().is_element(dofs[v]))
            {
              if (state[dofs[v]]==m.node_min) m.minimum_location=vertices[v];
              if (previous[dofs[v]]-state[dofs[v]]==m.healing) m.healing_location=vertices[v];
            }
        const auto Hindex=pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
        for (const auto &p : pm.get_particle_handler())
          {
            const auto domain = pm.get_particle_domain_handler().get_particle_domain(p.get_local_index());
            double phi=0;
            Tensor<1,dim> gradient;
            for (unsigned int i=0; i<domain.n_relevant_vertices(); ++i)
              {
                const double value=state[dofs[domain.relevant_vertex_index(i)]];
                phi+=value*domain.weighting_function_value(i);
                gradient+=value*domain.weighting_function_gradient(i);
              }
            if (phi<m.cpdi_min)
              { m.cpdi_minimum_location=p.get_location(); m.H_at_cpdi_minimum=p.get_properties()[Hindex]; }
            m.cpdi_min=std::min(m.cpdi_min,phi); m.cpdi_max=std::max(m.cpdi_max,phi);
            m.negative_particles+=(phi<0);
            m.integral+=domain.volume()*handler.crack_surface_density(phi,gradient);
          }
        return m;
      }

      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          AssertThrow(dim==2 && Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1,
                      ExcMessage("P1 is a bounded serial diagnostic."));
          AssertThrow(this->get_timestep_number()==0, ExcMessage("P1 uses only initialization."));
          auto &handler=this->get_phase_field_handler();
          auto &pm=handler.get_associated_particle_manager();
          auto &particles=pm.get_particle_handler();
          const auto Hindex=pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
          const auto b=this->introspection().variable("phase_field").block_index;
          const auto &dofs=Access::vertex_dofs(handler);
          const auto &parameters=Access::parameters(handler);
          const auto comm=this->get_mpi_communicator();
          const std::string before=fingerprint();
          LinearAlgebra::BlockSparseMatrix matrix;
          matrix.reinit(this->get_system_matrix().n_block_rows(),this->get_system_matrix().n_block_cols());
          for (unsigned int r=0; r<matrix.n_block_rows(); ++r)
            for (unsigned int c=0; c<matrix.n_block_cols(); ++c)
              matrix.block(r,c).copy_from(this->get_system_matrix().block(r,c));
          matrix.collect_sizes();
          VectorType rhs(this->introspection().index_sets.system_partitioning,comm);
          VectorType owned(rhs), correction(rhs), residual_copy(rhs);
          VectorType state(this->get_solution()), previous(state);
          auto &live=const_cast<VectorType &>(this->get_solution());
          std::ofstream csv(this->get_output_directory()+"controlled-history.csv");
          csv << std::setprecision(17)
              << "stage,load_factor,initial_residual,roundoff,target,final_residual,node_min,node_max,cpdi_min,cpdi_max,change,healing,crack_integral,H_min,H_max,H_increment_min,correction,uncertainty,masked_particles,node_min_x,node_min_y,healing_x,healing_y,cpdi_min_x,cpdi_min_y,H_at_cpdi_min,negative_nodes,negative_particles,positive_offdiagonals\n";
          std::vector<Measures> measured;
          std::vector<double> uncertainties;
          unsigned int masked=0;
          {
            RestoreH restore_H;
            std::vector<bool> mask;
            for (auto &p : particles)
              {
                restore_H.entries.emplace_back(&p.get_properties()[Hindex],p.get_properties()[Hindex]);
                const auto domain=pm.get_particle_domain_handler().get_particle_domain(p.get_local_index());
                double phi=0;
                for (unsigned int i=0; i<domain.n_relevant_vertices(); ++i)
                  phi+=state[dofs[domain.relevant_vertex_index(i)]]*domain.weighting_function_value(i);
                mask.push_back(phi>0.1 && phi<0.7);
                masked+=mask.back();
              }
            AssertThrow(masked>=4, ExcMessage("P1 has no resolved partially damaged load region."));

            // Verify RHS-only and Jacobian/roundoff paths with supplied scratch phi.
            owned=this->get_solution(); owned.block(b)=0.2; state=owned;
            double allowance=0;
            AssertThrow(Access::assemble(handler,matrix,rhs,state,true,&allowance),ExcInternalError());
            const auto matrix_before=matrix_entries(matrix,b);
            residual_copy=rhs;
            RestoreVector restore_live(live);
            owned=live; owned.block(b)=1.25; live=owned;
            double poisoned_allowance=0;
            AssertThrow(Access::assemble(handler,matrix,rhs,state,true,&poisoned_allowance),ExcInternalError());
            AssertThrow(matrix_entries(matrix,b)==matrix_before && allowance==poisoned_allowance,
                        ExcMessage("Assembly Jacobian/scale depended on live phi."));
            VectorType difference(rhs); difference-=residual_copy;
            AssertThrow(difference.block(b).l2_norm()==0,ExcMessage("Assembly residual depended on live phi."));
            AssertThrow(Access::assemble(handler,matrix,rhs,state,false),ExcInternalError());
            difference=rhs; difference-=residual_copy;
            AssertThrow(difference.block(b).l2_norm()==0,ExcMessage("Residual-only assembly changed scratch evaluation."));
            owned=state; owned.block(b)=0.25; state=owned;
            AssertThrow(Access::assemble(handler,matrix,rhs,state,true),ExcInternalError());
            difference=rhs; difference-=residual_copy;
            AssertThrow(difference.block(b).l2_norm()>1e-6*residual_copy.block(b).l2_norm()
                        && matrix_entries(matrix,b)!=matrix_before,
                        ExcMessage("Assembly did not respond to scratch phi."));
            this->get_pcout() << "P1 scratch residual/Jacobian/roundoff isolation verified\n";

            state=restore_live.saved;
            // Compare initialized FE profile with the native prescribed profile,
            // separately from subsequent equilibrium holds.
            VectorType analytic(owned);
            analytic=restore_live.saved;
            const auto &faults=this->get_reconstructed_fault_manager().get_prescribed_faults();
            AssertThrow(faults.size()==1 && faults[0].vertices.size()==2,
                        ExcMessage("P1 profile diagnostic requires one straight segment."));
            const auto &fault=faults[0];
            const double core=fault.core_phase_field_values[0];
            AssertThrow(fault.core_phase_field_values[1]==core && fault.vertices[0][1]==fault.vertices[1][1],
                        ExcMessage("P1 profile diagnostic requires constant core and horizontal geometry."));
            const auto profiles=handler.get_phase_field_profiles(core);
            const auto &vertices=this->get_triangulation().get_vertices();
            for (unsigned int v=0; v<dofs.size(); ++v)
              if (dofs[v]!=numbers::invalid_dof_index && analytic.locally_owned_elements().is_element(dofs[v]))
                {
                  const double distance=ReconstructedFaultUtilities::closest_point_distance_and_core_phase_field(fault,vertices[v]).first;
                  const double phi=profiles[0]->value(distance);
                  for (const auto &profile : profiles)
                    AssertThrow(profile->value(distance)==phi,
                                ExcMessage("P1 profile diagnostic requires identical material profiles."));
                  analytic[dofs[v]]=phi;
                }
            VectorType analytic_ghost(state); analytic_ghost=analytic;
            const auto initialization=measure(state,analytic_ghost);
            this->get_pcout() << std::setprecision(17) << "P1 initialization profile difference="
              << initialization.change << " raw_min=" << initialization.node_min << " raw_max=" << initialization.node_max
              << " cells=" << this->get_triangulation().n_global_active_cells()
              << " particles=" << particles.n_global_particles() << " masked=" << masked << std::endl;

            // Observe the supplied native profile and the converged initialization.
            // Clipping below is an assembly-only diagnostic; never solve or publish it.
            AssertThrow(Access::assemble(handler,matrix,rhs,analytic_ghost,false),ExcInternalError());
            const VectorType analytic_rhs(rhs);
            const double profile_residual=rhs.block(b).l2_norm();
            AssertThrow(Access::assemble(handler,matrix,rhs,state,false),ExcInternalError());
            const VectorType initialized_rhs(rhs);
            const double initialized_residual=rhs.block(b).l2_norm();
            owned=state;
            for (const auto i : owned.block(b).locally_owned_elements())
              owned.block(b)[i]=std::max(0.0,static_cast<double>(owned.block(b)[i]));
            VectorType clipped(state); clipped=owned;
            AssertThrow(Access::assemble(handler,matrix,rhs,clipped,false),ExcInternalError());
            this->get_pcout() << "P1 tail support_radius=" << profiles[0]->get_coordinate_values().back()
                << " profile_residual=" << profile_residual << " initialized_residual=" << initialized_residual
                << " clipped_residual=" << rhs.block(b).l2_norm() << std::endl;
            const double midpoint=0.5*(fault.vertices[0][0]+fault.vertices[1][0]);
            std::ofstream tail(this->get_output_directory()+"initialization-nodes.csv");
            tail << std::setprecision(17) << "x,y,profile,initialized,profile_rhs,initialized_rhs,clipped_rhs\n";
            for (unsigned int v=0; v<dofs.size(); ++v)
              if (dofs[v]!=numbers::invalid_dof_index && analytic.locally_owned_elements().is_element(dofs[v])
                  && (std::abs(vertices[v][0]-midpoint)<1e-12 || vertices[v][1]==initialization.minimum_location[1]))
                tail << vertices[v][0] << ',' << vertices[v][1] << ',' << analytic[dofs[v]] << ',' << state[dofs[v]]
                     << ',' << analytic_rhs[dofs[v]] << ',' << initialized_rhs[dofs[v]] << ',' << rhs[dofs[v]] << '\n';
            // One nearest particle column, not a whole-domain history dump.
            double column=1e100;
            for (const auto &p : particles)
              column=std::min(column,std::abs(p.get_location()[0]-midpoint));
            std::ofstream tail_H(this->get_output_directory()+"initialization-particles.csv");
            tail_H << std::setprecision(17) << "x,y,profile,H,initialized_cpdi\n";
            for (const auto &p : particles)
              if (p.get_location()[0]>=midpoint && std::abs(p.get_location()[0]-midpoint)<=column+1e-12)
                {
                  const auto domain=pm.get_particle_domain_handler().get_particle_domain(p.get_local_index());
                  double phi=0;
                  for (unsigned int i=0; i<domain.n_relevant_vertices(); ++i)
                    phi+=state[dofs[domain.relevant_vertex_index(i)]]*domain.weighting_function_value(i);
                  const double distance=ReconstructedFaultUtilities::closest_point_distance_and_core_phase_field(fault,p.get_location()).first;
                  tail_H << p.get_location()[0] << ',' << p.get_location()[1] << ',' << profiles[0]->value(distance)
                         << ',' << p.get_properties()[Hindex] << ',' << phi << '\n';
                }

            const std::vector<std::string> labels={"hold", "hold_repeat", "ramp_05", "ramp_10", "unload"};
            const std::vector<double> factors={0.9,0.9,1.05,1.10,1.0};
            for (unsigned int k=0; k<labels.size(); ++k)
              {
                double Hmin=1e100,Hmax=0,Hincrement=1e100;
                for (unsigned int p=0; p<restore_H.entries.size(); ++p)
                  {
                    auto &entry=restore_H.entries[p];
                    const double old=*entry.first;
                    if (mask[p]) *entry.first=std::max(old,factors[k]*entry.second);
                    Hincrement=std::min(Hincrement,*entry.first-old);
                    Hmin=std::min(Hmin,*entry.first); Hmax=std::max(Hmax,*entry.first);
                  }
                AssertThrow(Hincrement>=0,ExcMessage("Prescribed diagnostic H decreased."));
                previous=state;
                AssertThrow(Access::assemble(handler,matrix,rhs,state,false,&allowance),ExcInternalError());
                const double initial=rhs.block(b).l2_norm();
                const double target=std::max(parameters.nonlinear_solver_tolerance*initial,allowance);
                this->get_pcout() << "P1 solve " << labels[k] << std::endl;
                handler.evolve_phase_field(matrix,rhs,state);
                double final_allowance=0;
                AssertThrow(Access::assemble(handler,matrix,rhs,state,true,&final_allowance),ExcInternalError());
                const double final=rhs.block(b).l2_norm();
                AssertThrow(std::isfinite(final) && final<=target,ExcMessage("P1 fresh equality residual failed."));
                correction=0;
                if (final>0) Access::solve(handler,matrix,rhs,correction);
                const double correction_norm=correction.block(b).linfty_norm();
                // Directional estimate, NOT a bound on ||J^-1|| or physical error.
                const double estimate=correction_norm*std::max(1.0,final_allowance/std::max(final,std::numeric_limits<double>::min()));
                const double uncertainty=std::max(estimate,std::numeric_limits<double>::epsilon()*std::max(1.0,state.block(b).linfty_norm()));
                const auto m=measure(state,previous);
                measured.push_back(m); uncertainties.push_back(uncertainty);
                unsigned int positive_offdiagonals=0;
                for (const auto row : rhs.block(b).locally_owned_elements())
                  for (auto entry=matrix.block(b,b).begin(row); entry!=matrix.block(b,b).end(row); ++entry)
                    if (entry->column()!=row && entry->value()>0) ++positive_offdiagonals;
                csv << labels[k] << ',' << factors[k] << ',' << initial << ',' << allowance << ',' << target << ',' << final
                    << ',' << m.node_min << ',' << m.node_max << ',' << m.cpdi_min << ',' << m.cpdi_max
                    << ',' << m.change << ',' << m.healing << ',' << m.integral << ',' << Hmin << ',' << Hmax
                    << ',' << Hincrement << ',' << correction_norm << ',' << uncertainty << ',' << masked
                    << ',' << m.minimum_location[0] << ',' << m.minimum_location[1]
                    << ',' << m.healing_location[0] << ',' << m.healing_location[1]
                    << ',' << m.cpdi_minimum_location[0] << ',' << m.cpdi_minimum_location[1]
                    << ',' << m.H_at_cpdi_minimum << ',' << m.negative_nodes << ',' << m.negative_particles
                    << ',' << positive_offdiagonals << '\n';
                csv.flush();
                this->get_pcout() << std::setprecision(17) << "P1 result " << labels[k] << " residual=" << final
                    << " target=" << target << " phi_min=" << m.node_min << " phi_max=" << m.node_max
                    << " change=" << m.change << " healing=" << m.healing << " uncertainty=" << uncertainty << std::endl;
              }
            // Rejected branch probe cannot mutate the accepted scratch state or H.
            const VectorType accepted(state);
            const std::string before_rejection=fingerprint();
            owned=state; owned.block(b)=1.25;
            VectorType invalid(state); invalid=owned;
            AssertThrow(!Access::assemble(handler,matrix,rhs,invalid,false),ExcMessage("Invalid scratch phase was accepted."));
            AssertThrow(fingerprint()==before_rejection,ExcMessage("Rejected assembly changed history."));
            difference=state; difference-=accepted;
            AssertThrow(difference.l2_norm()==0,ExcMessage("Rejected assembly changed accepted scratch phi."));
          }
          AssertThrow(fingerprint()==before,ExcMessage("P1 normal restoration failed."));
          this->get_pcout() << "P1 normal restoration verified\n";

          // Force the actual evolve entry to throw after temporary H/live changes.
          bool rejected=false;
          try
            {
              RestoreH restore_H;
              for (auto &p : particles)
                {
                  restore_H.entries.emplace_back(&p.get_properties()[Hindex],p.get_properties()[Hindex]);
                  p.get_properties()[Hindex]*=1.05;
                }
              RestoreVector restore_live(live);
              owned=live; owned.block(b)=0.3; live=owned;
              owned.block(b)=1.25; state=owned;
              handler.evolve_phase_field(matrix,rhs,state);
            }
          catch (const std::exception &error)
            {
              rejected=std::string(error.what()).find("outside the nonsingular")!=std::string::npos;
            }
          AssertThrow(rejected && fingerprint()==before,ExcMessage("P1 exceptional restoration failed."));
          this->get_pcout() << "P1 exceptional restoration verified\n";

          const double repeat=std::max(measured[1].change,measured[4].change);
          const double uncertainty=*std::max_element(uncertainties.begin(),uncertainties.end());
          const double tolerance=10*std::max(repeat,uncertainty);
          // Repeat drift must itself be compatible with the correction estimate.
          const bool repeat_resolved=repeat<=10*uncertainty;
          const bool evolution=std::min(measured[2].change,measured[3].change)>100*tolerance;
          bool admissible=true;
          for (unsigned int k=0; k<measured.size(); ++k)
            {
              const auto &m=measured[k];
              // First hold finishes initialization; its relaxation is not loading-induced healing.
              admissible=admissible && m.node_min>=-tolerance && m.node_max<=1+tolerance
                         && m.cpdi_min>=-tolerance && m.cpdi_max<=1+tolerance
                         && (k==0 || m.healing<=tolerance);
            }
          this->get_pcout() << std::setprecision(17) << "P1 assessment tolerance=" << tolerance
              << " repeat=" << repeat << " uncertainty=" << uncertainty
              << " repeat_resolved=" << repeat_resolved << " evolution=" << evolution
              << " admissible=" << admissible << std::endl;
          AssertThrow(repeat_resolved && evolution,ExcMessage("P1 evolution/uncertainty is inconclusive."));
          AssertThrow(admissible,ExcMessage("P1 equality solve converged but physical admissibility is not qualified."));
          return {"P1 controlled phase history:","qualified for prescribed H only"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(ControlledPhaseHistory,"controlled phase history",
                                 "Bounded controlled-H production phase-operator diagnostic.")
  }
}
