// Small clean-start diagnostic. No replacement constitutive or transfer law.
#include <aspect/postprocess/interface.h>
#include <aspect/particle/manager.h>
#include <aspect/phase_field.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/plugins.h>
#include <aspect/simulator_signals.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <cstdlib>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class CleanStressCycle : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::list<std::string> required_other_postprocessors() const override { return {"particles"}; }

        static void declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Postprocess");prm.enter_subsection("Clean stress cycle");
          prm.declare_entry("Prescribed slip rate","0.005",Patterns::Double(1e-12),
                            "Diagnostic-only controlled source amplitude; no free RSF solve.");
          prm.declare_entry("Compact output","false",Patterns::Bool(),
                            "Keep invariants but suppress full-domain transfer CSVs in the moment comparison.");
          prm.declare_entry("Fault angle","0",Patterns::Double(0,90),
                            "Diagnostic straight fault angle in degrees on an unrotated Cartesian box, through the origin.");
          prm.declare_entry("Advect particles","false",Patterns::Bool(),
                            "Use ordinary particle advection in the interpolation comparison; default retains the frozen audit.");
          prm.leave_subsection();prm.leave_subsection();
        }
        void parse_parameters(ParameterHandler &prm) override
        {
          prm.enter_subsection("Postprocess");prm.enter_subsection("Clean stress cycle");
          prescribed_rate=prm.get_double("Prescribed slip rate");
          compact=prm.get_bool("Compact output");
          fault_angle=prm.get_double("Fault angle")*numbers::PI/180.;
          advect_particles=prm.get_bool("Advect particles");
          prm.leave_subsection();prm.leave_subsection();
        }

        void initialize() override
        {
          AssertThrow(dim==2 && std::getenv("ASPECT_STRESS_CYCLE_TRACE")
                      && (advect_particles ? !std::getenv("ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION")
                                           : std::getenv("ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION")!=nullptr),
                      ExcMessage("Clean cycle requires 2D and the supplied diagnostic environment."));
          this->get_signals().pre_set_initial_state.connect([this](auto &tria)
          {
            std::ofstream out(this->get_output_directory()+"stress_trace_cells_rank"+rank()+".txt");
            for (const auto &cell:tria.active_cell_iterators())
              if (cell->is_locally_owned() && !compact) out<<cell->id()<<'\n';
          });
          // Prescribe only the Q1 phase component, at every step including zero.
          // The physical stationary distance profile is shared with production.
          this->get_signals().post_constraints_creation.connect(
            [this](const SimulatorAccess<dim> &, AffineConstraints<double> &constraints)
          {
            const auto profiles=this->get_phase_field_handler().get_phase_field_profiles(.6);
            const auto &fe=this->get_fe();
            const auto phi=this->introspection().variable("phase_field").first_component_index;
            std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());
            for (const auto &cell:this->get_dof_handler().active_cell_iterators())
              if (!cell->is_artificial())
                {
                  cell->get_dof_indices(dofs);
                  for (unsigned int j=0;j<dofs.size();++j)
                    if (fe.system_to_component_index(j).first==phi
                        && constraints.can_store_line(dofs[j]) && !constraints.is_constrained(dofs[j]))
                      {
                        const auto p=this->get_mapping().transform_unit_to_real_cell(cell,fe.get_unit_support_points()[j]);
                        constraints.add_line(dofs[j]);
                        const double distance=-std::sin(fault_angle)*p[0]+std::cos(fault_angle)*p[1];
                        constraints.set_inhomogeneity(dofs[j],profiles[0]->value(std::abs(distance)));
                      }
                }
          });
          this->get_signals().post_advection_solver.connect(
            [this](const SimulatorAccess<dim> &,bool temperature,unsigned int,const SolverControl &)
          {
            if (temperature)
              {
                auto &manager=this->get_reconstructed_fault_manager();
                AssertThrow(manager.get_faults().size()==1,ExcMessage("Clean cycle requires one straight fault."));
                // Isolate the history cycle, not the rate/state startup. Bulk
                // mechanics still solves the actual distributed slip source.
                std::vector<std::map<unsigned int,double>> prescribed(1);
                for (unsigned int v=0;v<manager.get_fault(0).n_vertices();++v) prescribed[0][v]=prescribed_rate;
                manager.set_prescribed_slip_rates(prescribed);
                this->get_reconstructed_fault_surface_system().enable_bulk_work_measure();
              }
          });
          this->get_signals().start_timestep.connect([this](const SimulatorAccess<dim> &)
          {
            converged=false;
            write_particles("before");
          });
          this->get_signals().post_nonlinear_solver.connect([this](const SolverControl &control)
          {
            converged=control.last_check()==SolverControl::success
                      && std::isfinite(control.last_value()) && control.last_value()<=control.tolerance();
          });
        }

        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          AssertThrow(converged,ExcMessage("Clean cycle requires genuine nonlinear convergence."));
          write_particles("after");
          const auto &intro=this->introspection();
          const auto &fe=this->get_fe();
          const auto &manager=this->get_reconstructed_fault_manager();
          const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
          const QGauss<dim> quadrature(3);
          FEValues<dim> values(this->get_mapping(),fe,quadrature,
                              update_values|update_gradients|update_quadrature_points|update_JxW_values);
          auto out=file("qp","cell,q,x,y,JxW,old_xx,old_yy,old_xy,published_xx,published_yy,published_xy,grad_xx,grad_xy,grad_yx,grad_yy,phi,chi,V,kappa");
          auto load=file("weak_history","cell,local_velocity_dof,component,load");
          std::array<std::vector<double>,3> old,published;
          for (auto *a:{&old,&published}) for (auto &v:*a) v.resize(quadrature.size());
          std::vector<double> phi(quadrature.size());
          std::vector<Tensor<2,dim>> gradient(quadrature.size());
          const FEValuesExtractors::Scalar phase(intro.variable("phase_field").first_component_index);
          for (const auto &cell:this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                values.reinit(cell);
                for (unsigned int c=0;c<3;++c)
                  {
                    values[intro.extractors.compositional_fields[c]].get_function_values(this->get_current_linearization_point(),old[c]);
                    values[intro.extractors.compositional_fields[c]].get_function_values(this->get_solution(),published[c]);
                  }
                values[intro.extractors.velocities].get_function_gradients(this->get_solution(),gradient);
                values[phase].get_function_values(this->get_solution(),phi);
                const auto &association=manager.get_stokes_qp_fault_associations(cell->id(),quadrature,values.get_quadrature_points());
                std::vector<double> weak(fe.n_dofs_per_cell(),0.);
                for (unsigned int q=0;q<quadrature.size();++q)
                  {
                    SymmetricTensor<2,dim> stress;
                    stress[0][0]=old[0][q];stress[1][1]=old[1][q];stress[0][1]=old[2][q];
                    const auto inherited=model.evaluate_frozen_maxwell_stress(293.,{old[0][q],old[1][q],old[2][q],200.},stress);
                    // The same Q2 velocity test functions and bulk quadrature as
                    // assembly; local signed loads, before global constraints.
                    for (unsigned int j=0;j<weak.size();++j)
                      if (intro.component_masks.velocities[fe.system_to_component_index(j).first])
                        weak[j]+=inherited*values[intro.extractors.velocities].symmetric_gradient(j,q)*values.JxW(q);
                    double chi=0.,V=0.,eta_ve=0.;
                    if (association[q].active)
                      {
                        typename MaterialModel::PhaseFieldFault<dim>::ReconstructedFaultBulkPointInputs in;
                        in.fault_index=association[q].fault_index;in.segment_index=association[q].segment_index;
                        in.xi=association[q].xi;in.phase_field=in.previous_phase_field=phi[q];
                        in.temperature=293.;in.bulk_material_fractions={1.};
                        const auto response=model.evaluate_reconstructed_fault_bulk_point(in);
                        chi=response.localization_factor;eta_ve=response.eta_ve;
                        V=manager.interpolate_slip_rate(in.fault_index,in.segment_index,in.xi);
                        AssertThrow(response.history_correction==0.,ExcMessage("Fixed mature source acquired a history correction."));
                      }
                    const auto x=values.quadrature_point(q);
                    out<<cell->id()<<','<<q<<','<<x[0]<<','<<x[1]<<','<<values.JxW(q);
                    for (const auto &a:{old,published}) for (const auto &v:a) out<<','<<v[q];
                    out<<','<<gradient[q][0][0]<<','<<gradient[q][0][1]<<','<<gradient[q][1][0]<<','<<gradient[q][1][1]
                       <<','<<phi[q]<<','<<chi<<','<<V<<','<<eta_ve<<'\n';
                  }
                for (unsigned int j=0;j<weak.size();++j)
                  if (intro.component_masks.velocities[fe.system_to_component_index(j).first])
                    load<<cell->id()<<','<<j<<','<<fe.system_to_component_index(j).first<<','<<weak[j]<<'\n';
              }
          if (this->get_pcout().is_active())
            {
              auto clock=file("clock","step,time,dt");
              clock<<this->get_timestep_number()<<','<<this->get_time()<<','<<this->get_timestep()<<'\n';
            }
          return {"Clean stress cycle","accepted transfer/update captured"};
        }

      protected:
        bool compact=false;
        bool advect_particles=false;
        double prescribed_rate=.005;
        double fault_angle=0.;
        bool converged=false;
        std::map<types::particle_index,Point<dim>> positions;
        std::string rank() const { return std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator())); }
        std::ofstream file(const std::string &name,const std::string &header) const
        {
          std::ofstream out(this->get_output_directory()+"clean_"+name+"_"+std::to_string(this->get_timestep_number())+"_rank"+rank()+".csv");
          out.exceptions(std::ios::failbit|std::ios::badbit);out<<std::setprecision(17)<<header<<'\n';return out;
        }
        void write_particles(const std::string &stage)
        {
          const auto &pm=this->get_particle_manager(0);
          const auto stress=pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
          std::ofstream out;
          if (!compact || advect_particles) out=file(stage,"cell,id,x,y,ref_x,ref_y,xx,yy,xy");
          for (const auto &p:pm.get_particle_handler())
            {
              const auto x=p.get_location(),r=p.get_reference_location();const auto v=p.get_properties();
              if (!positions.count(p.get_id())) positions.emplace(p.get_id(),x);
              if (!advect_particles) AssertThrow(positions.at(p.get_id())==x,ExcMessage("Clean cycle moved a particle."));
              if (!compact || advect_particles) out<<p.get_surrounding_cell()->id()<<','<<p.get_id()<<','<<x[0]<<','<<x[1]<<','<<r[0]<<','<<r[1]
                 <<','<<v[stress]<<','<<v[stress+1]<<','<<v[stress+2]<<'\n';
            }
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(CleanStressCycle,"clean stress cycle","Small zero-history production transfer/update diagnostic.")
  }
}
