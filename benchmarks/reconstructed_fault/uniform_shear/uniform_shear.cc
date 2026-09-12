/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#include <aspect/simulator_access.h>
#include <aspect/simulator_signals.h>
#include <aspect/postprocess/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/particle/manager.h>
#include <aspect/particle/particle_domain.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/plugins.h>
#include <aspect/boundary_velocity/interface.h>
#include <aspect/geometry_model/interface.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/numerics/vector_tools_evaluate.h>
#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include "evolving/safeguards.h"

namespace aspect
{
  namespace
  {
    // Fixed-mesh pilot snapshots, not a production or restart facility.
    std::map<types::global_dof_index, double> frozen_phi;
    std::map<types::particle_index, double> frozen_H;
    bool coupled_solve_converged = false;
    bool evolving_profile = false;
    std::vector<std::vector<double>> previous_cohesive_traction, previous_Ih;

    // Read-only phase-entry evidence, before composition advection or mechanics.
    // Stable IDs connect these old inputs to the previous accepted publication.
    template <int dim>
    void capture_evolving_inputs(const SimulatorAccess<dim> &simulator)
    {
      if (!evolving_profile)
        return;
      K3::phase_entry(simulator);
      const auto &pm = simulator.get_phase_field_handler().get_associated_particle_manager();
      const auto H_position = pm.get_property_manager().get_data_info()
                              .get_position_by_field_name("crack_driving_force");
      const auto step = simulator.get_timestep_number();
      const auto rank = Utilities::MPI::this_mpi_process(simulator.get_mpi_communicator());
      std::ofstream out(simulator.get_output_directory()+"phase_input_"
                        +std::to_string(step)+"_rank"+std::to_string(rank)+".csv");
      out.exceptions(std::ios::failbit | std::ios::badbit);
      out << std::setprecision(17) << "step,time,id,x,y,H\n";
      for (const auto &particle : pm.get_particle_handler())
        out << step << ',' << simulator.get_time() << ',' << particle.get_id()
            << ',' << particle.get_location()[0] << ',' << particle.get_location()[1]
            << ',' << particle.get_properties()[H_position] << '\n';

      previous_cohesive_traction.clear();
      previous_Ih.clear();
      if (step > 0)
        {
          const auto &manager = simulator.get_reconstructed_fault_manager();
          const auto C = manager.get_property_information()[manager.get_property_index(
                           "phase field fault cohesive traction")].position;
          const auto I = manager.get_property_information()[manager.get_property_index(
                           "phase field fault previous I h")].position;
          for (const auto &fault : manager.get_faults())
            {
              previous_cohesive_traction.emplace_back();
              previous_Ih.emplace_back();
              for (unsigned int v=0; v<fault.n_vertices(); ++v)
                {
                  previous_cohesive_traction.back().push_back(fault.get_properties(v)[C]);
                  previous_Ih.back().push_back(fault.get_properties(v)[I]);
                }
            }
        }
    }

    template <int dim>
    void freeze_phase(const SimulatorAccess<dim> &simulator,
                      AffineConstraints<double> &constraints)
    {
      if (evolving_profile || simulator.get_timestep_number() == 0)
        return;
      AssertThrow(!frozen_phi.empty(), ExcMessage("Missing initial K1 phase snapshot."));
      // Prescribe independent phi DoFs only. Existing periodic/hanging-node
      // relations remain, and no mechanical or other field DoF is touched.
      for (const auto &entry : frozen_phi)
        if (constraints.can_store_line(entry.first)
            && !constraints.is_constrained(entry.first))
          {
            constraints.add_line(entry.first);
            constraints.set_inhomogeneity(entry.first, entry.second);
          }
    }
  }

  template <int dim>
  void connect_uniform_shear(SimulatorSignals<dim> &signals)
  {
    signals.post_constraints_creation.connect(&freeze_phase<dim>);
    signals.start_timestep.connect(&capture_evolving_inputs<dim>);
    signals.post_advection_solver.connect([](const SimulatorAccess<dim> &sim,
                                             const bool temperature, const unsigned int,
                                             const SolverControl &)
    { if (evolving_profile && temperature) K3::phase_exit(sim); });
    signals.start_timestep.connect([](const SimulatorAccess<dim> &)
    {
      coupled_solve_converged = false;
    });
    signals.post_nonlinear_solver.connect([](const SolverControl &control)
    {
      coupled_solve_converged = control.last_check() == SolverControl::success
        && std::isfinite(control.last_value())
        && control.last_value() < control.tolerance();
    });
  }
  ASPECT_REGISTER_SIGNALS_CONNECTOR(connect_uniform_shear<2>, connect_uniform_shear<3>)

  namespace Postprocess
  {
    template <int dim>
    class UniformShear : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("Uniform shear pilot");
          prm.declare_entry("Evolving profile", "false", Patterns::Bool(),
                            "Opt in to the K3 evolving-profile diagnostic mode. "
                            "Do not constrain phi or require H to remain frozen. "
                            "The material's Evolve phase field parameter still controls H updates.");
          prm.leave_subsection();
          prm.leave_subsection();
        }

        void parse_parameters(ParameterHandler &prm) override
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("Uniform shear pilot");
          evolving_profile = prm.get_bool("Evolving profile");
          prm.leave_subsection();
          prm.leave_subsection();
        }

        // The fixed-profile fixture must restore its original snapshots, not
        // silently redefine them from the resumed solution or evolved histories.
        void save(std::map<std::string,std::string> &strings) const override
        {
          std::ostringstream output;
          {
            aspect::oarchive archive(output);
            archive << frozen_phi << frozen_H;
          }
          strings["UniformShearFrozenFields"] = output.str();
        }

        void load(const std::map<std::string,std::string> &strings) override
        {
          std::istringstream input(strings.at("UniformShearFrozenFields"));
          aspect::iarchive archive(input);
          archive >> frozen_phi >> frozen_H;
        }

        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          AssertThrow(coupled_solve_converged,
                      ExcMessage("Benchmark output requires both final nonlinear "
                                 "convergence criteria; a continued failed solve "
                                 "is not an accepted trajectory sample."));
          AssertThrow(dim == 2, ExcMessage("This diagnostic pilot is two-dimensional only."));
          if (evolving_profile) K3::accepted(*this);
          const auto communicator = this->get_mpi_communicator();
          const bool distributed = Utilities::MPI::n_mpi_processes(communicator)>1;
          const std::string rank_suffix = distributed
            ? "_rank"+std::to_string(Utilities::MPI::this_mpi_process(communicator)) : "";
          const unsigned int step = this->get_timestep_number();
          const auto &intro = this->introspection();
          const auto &fe = this->get_fe();
          const auto &solution = this->get_solution();
          const unsigned int phi_component = intro.variable("phase_field").first_component_index;
          const FEValuesExtractors::Scalar phi_extractor(phi_component);
          auto &manager = this->get_reconstructed_fault_manager();
          const auto &model = Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
            this->get_material_model());
          auto csv = [&](const std::string &name, const std::string &header)
          {
            std::ofstream file(this->get_output_directory()+name+rank_suffix+"_"+std::to_string(step)+".csv");
            file.exceptions(std::ios::failbit | std::ios::badbit);
            file << std::setprecision(17) << header << '\n';
            return file;
          };

          // Export the actual accepted time/load sequence. At zero, dt=2 is
          // the numerical Maxwell interval, not physical history advancement.
          auto times = csv("time", "step,time,dt,U");
          double loading = 1e-4*(1+0.2*std::min(this->get_time()/4,1.0));
          if (evolving_profile)
            {
              const auto &geometry = this->get_geometry_model();
              const auto &boundary = this->get_boundary_velocity_manager();
              Point<dim> top, bottom;
              top[0]=bottom[0]=.125; top[1]=.5; bottom[1]=-.5;
              loading = boundary.boundary_velocity(geometry.translate_symbolic_boundary_name_to_id("top"),top)[0]
                        -boundary.boundary_velocity(geometry.translate_symbolic_boundary_name_to_id("bottom"),bottom)[0];
            }
          times << step << ',' << this->get_time() << ','
                << (step == 0 ? 2.0 : this->get_timestep()) << ','
                << loading << '\n';

          // Export all Q1 support values, independently addressable as a
          // rectangular FE grid by the reference. Repeated cell nodes agree.
          auto nodes = csv("phase", "dof,x,y,phi");
          std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
          const auto &supports = fe.get_unit_support_points();
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                cell->get_dof_indices(indices);
                for (unsigned int i=0; i<indices.size(); ++i)
                  if (fe.system_to_component_index(i).first == phi_component)
                    {
                      const auto point = this->get_mapping().transform_unit_to_real_cell(cell, supports[i]);
                      const double value = solution[indices[i]];
                      if (step == 0)
                        frozen_phi[indices[i]] = value;
                      else if (!evolving_profile)
                        AssertThrow(value == frozen_phi.at(indices[i]),
                                    ExcMessage("K1 constraints did not freeze the initial Q1 phase field."));
                      nodes << indices[i] << ',' << point[0] << ',' << point[1] << ',' << value << '\n';
                    }
              }

          // These are committed histories. They must not be confused with the
          // timestep-zero evaluated Maxwell/cohesive responses in the reference.
          auto surface = csv("surface", "fault,node,x,y,V,Theta,C,Ih");
          const auto position = [&](const std::string &name)
          { return manager.get_property_information()[manager.get_property_index(name)].position; };
          const unsigned int theta_position = position("phase field fault state");
          const unsigned int C_position = position("phase field fault cohesive traction");
          const unsigned int Ih_position = position("phase field fault previous I h");
          auto segments = csv("segments", "fault,segment,x,y,nx,ny,half_width_minus,half_width_plus");
          // The final linearization precedes history publication. Its frozen
          // weak terms are the actual accepted equation, not a reconstruction
          // from newly committed particle stress or a normal-column average.
          const auto &weak = this->get_reconstructed_fault_surface_system().get_linearization_residual();
          auto weak_output = csv("surface_weak", "fault,node,Mdiag,Moff,q,C,friction,damping,F");
          for (unsigned int f=0; f<weak.values.size(); ++f)
            for (unsigned int i=0; i<weak.values[f].size(); ++i)
              weak_output << f << ',' << i << ',' << weak.mass_diagonal[f][i] << ','
                          << (i<weak.mass_off_diagonal[f].size() ? weak.mass_off_diagonal[f][i] : 0.0)
                          << ',' << weak.shear_traction[f][i] << ',' << weak.cohesive_traction[f][i]
                          << ',' << weak.friction_traction[f][i] << ',' << weak.damping_traction[f][i]
                          << ',' << weak.values[f][i] << '\n';
          for (unsigned int f=0; f<manager.get_faults().size(); ++f)
            {
              const auto &fault = manager.get_fault(f);
              for (unsigned int v=0; v<fault.n_vertices(); ++v)
                {
                  const auto values = fault.get_properties(v);
                  surface << f << ',' << v << ',' << fault.vertex(v)[0] << ',' << fault.vertex(v)[1]
                          << ',' << manager.get_slip_rate(f)[v] << ',' << values[theta_position]
                          << ',' << values[C_position] << ',' << values[Ih_position] << '\n';
                }
              for (unsigned int s=0; s<fault.n_cells(); ++s)
                {
                  const Point<dim> center = 0.5*(fault.vertex(s)+fault.vertex(s+1));
                  Tensor<1,dim> tangent = fault.vertex(s+1)-fault.vertex(s);
                  tangent /= tangent.norm();
                  Tensor<1,dim> normal;
                  normal[0] = -tangent[1]; normal[1] = tangent[0];
                  segments << f << ',' << s << ',' << center[0] << ',' << center[1]
                           << ',' << normal[0] << ',' << normal[1];
                  for (const double sign : {-1.0, 1.0})
                    {
                      double lower=0, upper=0.49;
                      for (unsigned int iteration=0; iteration<45; ++iteration)
                        {
                          const double middle = 0.5*(lower+upper);
                          const auto association = manager.project_to_normal_profiles(center+sign*middle*normal);
                          if (association.active && association.fault_index == f)
                            lower = middle;
                          else
                            upper = middle;
                        }
                      segments << ',' << lower;
                    }
                  segments << '\n';
                }
            }

          // Particle volumes and endpoint support are measured, not inferred
          // from the periodic FE topology or the absence of particle wrapping.
          const auto &pm = this->get_phase_field_handler().get_associated_particle_manager();
          const auto &data = pm.get_property_manager().get_data_info();
          const unsigned int H_position = data.get_position_by_field_name("crack_driving_force");
          const unsigned int stress_position = data.get_position_by_field_name("maxwell stress");
          auto particles = csv("particles", "id,x,y,volume,H,tau_xx,tau_yy,tau_xy,active,fault,segment,xi");
          for (const auto &particle : pm.get_particle_handler())
            {
              const auto values = particle.get_properties();
              if (step == 0)
                frozen_H[particle.get_id()] = values[H_position];
              else if (!evolving_profile)
                AssertThrow(values[H_position] == frozen_H.at(particle.get_id()),
                            ExcMessage("K1 changed frozen particle H."));
              const auto association = manager.project_to_normal_profiles(particle.get_location());
              particles << particle.get_id() << ',' << particle.get_location()[0] << ',' << particle.get_location()[1]
                        << ',' << pm.get_particle_domain_handler().get_particle_domain(particle.get_local_index()).volume()
                        << ',' << values[H_position] << ',' << values[stress_position] << ',' << values[stress_position+1]
                        << ',' << values[stress_position+2] << ',' << association.active;
              if (association.active)
                particles << ',' << association.fault_index << ',' << association.segment_index << ',' << association.xi;
              else
                particles << ",-1,-1,0";
              particles << '\n';
            }

          // At real steps the published particle stress is the accepted stress.
          // At zero it is deliberately retained instead: export the evaluated
          // traction and production point balance before any physical evolution.
          if (step == 0)
            {
              const auto &associations = manager.get_locally_owned_particle_fault_associations();
              std::vector<Point<dim>> points;
              for (const auto &a : associations)
                points.push_back(a.position);
              GridTools::Cache<dim> grid(this->get_triangulation(), this->get_mapping());
              Utilities::MPI::RemotePointEvaluation<dim> cache;
              cache.reinit(grid, points);
              const auto strain = VectorTools::point_gradients<dim>(cache,
                this->get_dof_handler(), solution, VectorTools::EvaluationFlags::avg,
                intro.component_indices.velocities[0]);
              const auto phase = VectorTools::point_values<1>(cache,
                this->get_dof_handler(), solution, VectorTools::EvaluationFlags::avg, phi_component);
              const auto temperatures = VectorTools::point_values<1>(cache,
                this->get_dof_handler(), solution, VectorTools::EvaluationFlags::avg,
                intro.component_indices.temperature);
              const auto pressures = VectorTools::point_values<1>(cache,
                this->get_dof_handler(), solution, VectorTools::EvaluationFlags::avg,
                intro.component_indices.pressure);
              auto traction = csv("particle_traction_initial", "id,x,y,fault,segment,xi,volume,V,phi,q,mu,F");
              unsigned int p=0;
              for (const auto &particle : pm.get_particle_handler())
                {
                  const auto &a = associations[p];
                  if (a.active)
                    {
                      const auto &fault = manager.get_fault(a.fault_index);
                      Tensor<1,dim> tangent = fault.vertex(a.segment_index+1)-fault.vertex(a.segment_index);
                      tangent /= tangent.norm();
                      Tensor<1,dim> normal;
                      normal[0]=-tangent[1]; normal[1]=tangent[0];
                      typename MaterialModel::PhaseFieldFault<dim>::ReconstructedFaultPointInputs input;
                      input.fault_index=a.fault_index; input.segment_index=a.segment_index; input.xi=a.xi;
                      input.position=a.position;
                      input.slip_rate=manager.interpolate_slip_rate(a.fault_index,a.segment_index,a.xi);
                      input.phase_field=phase[p]; input.previous_phase_field=phase[p];
                      input.temperature=temperatures[p]; input.dynamic_pressure=pressures[p];
                      input.strain_rate=symmetrize(strain[p]);
                      input.slip_tensor=symmetrize(outer_product(tangent,normal));
                      input.normal_tensor=symmetrize(outer_product(normal,normal));
                      input.bulk_material_fractions={1.0};
                      for (unsigned int c=0; c<SymmetricTensor<2,dim>::n_independent_components; ++c)
                        input.old_maxwell_stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]
                          =particle.get_properties()[stress_position+c];
                      const auto response=model.evaluate_reconstructed_fault_point(input);
                      // Fixed initial profile has no history-slip correction.
                      // Reuse production beta*tau_old rather than retained tau0.
                      std::vector<double> composition(intro.n_compositional_fields,0.0);
                      const auto stress = 2*response.kappa*(input.strain_rate
                        -response.localization_factor*input.slip_rate*input.slip_tensor)
                        +model.evaluate_frozen_maxwell_stress(input.temperature,composition,input.old_maxwell_stress);
                      traction << particle.get_id() << ',' << a.position[0] << ',' << a.position[1]
                               << ',' << a.fault_index << ',' << a.segment_index << ',' << a.xi
                               << ',' << a.particle_domain_volume << ',' << input.slip_rate << ',' << phase[p]
                               << ',' << stress*input.slip_tensor << ',' << response.friction_coefficient
                               << ',' << response.residual_density << '\n';
                    }
                  ++p;
                }
            }

          // Use the exact assembler quadrature/cache. Export resolved gradients
          // and the FE old stress so reference analysis can reconstruct the
          // evaluated stress independently of the committed particle stress.
          const auto &quadrature = intro.quadratures.velocities;
          FEValues<dim> values(this->get_mapping(), fe, quadrature,
                               update_values | update_gradients | update_quadrature_points | update_JxW_values);
          const unsigned int nq = quadrature.size();
          std::vector<Tensor<1,dim>> velocity(nq);
          std::vector<Tensor<2,dim>> gradients(nq);
          std::vector<double> pressure(nq), phi(nq), old_phi(nq), old_stress(nq);
          std::array<std::vector<double>,3> published_history, assembly_history;
          for (unsigned int c=0; c<3; ++c)
            {
              published_history[c].resize(nq);
              assembly_history[c].resize(nq);
            }
          auto bulk = csv("bulk", "x,y,weight,ux,uy,ux_x,ux_y,uy_x,uy_y,p,phi,old_tau_xy,active,V,chi,kappa,history");
          // Publication retains non-Stokes fields from solution, whereas the
          // accepted linearization retains the constrained working fields used
          // in assembly. Export both histories; do not infer one from the other.
          auto transfer = csv("history_transfer", "row,published_xx,published_yy,published_xy,assembly_xx,assembly_yy,assembly_xy");
          // Explicit provenance for the existing QP rows; no change to their
          // values or ordering. Cell vertices also identify the saved native mesh.
          auto qp_cells = csv("bulk_cell_ids", "row,cell_id,q,v0,v1,v2,v3");
          unsigned int bulk_row = 0;
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                values.reinit(cell);
                values[intro.extractors.velocities].get_function_values(solution, velocity);
                values[intro.extractors.velocities].get_function_gradients(solution, gradients);
                values[intro.extractors.pressure].get_function_values(solution, pressure);
                values[phi_extractor].get_function_values(solution, phi);
                values[phi_extractor].get_function_values(this->get_old_solution(), old_phi);
                values[intro.extractors.compositional_fields[2]].get_function_values(solution, old_stress);
                for (unsigned int c=0; c<3; ++c)
                  {
                    values[intro.extractors.compositional_fields[c]].get_function_values(solution, published_history[c]);
                    values[intro.extractors.compositional_fields[c]].get_function_values(
                      this->get_current_linearization_point(), assembly_history[c]);
                  }
                const auto &associations = manager.get_stokes_qp_fault_associations(
                  cell->id(), quadrature, values.get_quadrature_points());
                for (unsigned int q=0; q<nq; ++q)
                  {
                    transfer << bulk_row;
                    for (unsigned int c=0; c<3; ++c)
                      transfer << ',' << published_history[c][q];
                    for (unsigned int c=0; c<3; ++c)
                      transfer << ',' << assembly_history[c][q];
                    transfer << '\n';
                    qp_cells << bulk_row++ << ',' << cell->id().to_string() << ',' << q;
                    for (unsigned int v=0; v<4; ++v)
                      qp_cells << ',' << cell->vertex_index(v);
                    qp_cells << '\n';
                    const auto &association = associations[q];
                    double V=0, chi=0, kappa=-1e8*std::expm1(-(step == 0 ? 2.0 : this->get_timestep())/100.), history=0;
                    if (association.active)
                      {
                        typename MaterialModel::PhaseFieldFault<dim>::ReconstructedFaultBulkPointInputs input;
                        input.fault_index=association.fault_index; input.segment_index=association.segment_index;
                        input.xi=association.xi; input.phase_field=phi[q]; input.previous_phase_field=old_phi[q];
                        input.temperature=293; input.bulk_material_fractions={1.0};
                        V=manager.interpolate_slip_rate(input.fault_index,input.segment_index,input.xi);
                        if (evolving_profile && step > 0)
                          {
                            // Postprocessing follows commit. Reconstruct the diagnostic
                            // from SAVED old C/Ih, never from already advanced history.
                            const auto &fault = manager.get_fault(input.fault_index);
                            const unsigned int s = input.segment_index;
                            const double xi = input.xi;
                            const double I = (1-xi)*fault.get_properties(s)[Ih_position]
                                             +xi*fault.get_properties(s+1)[Ih_position];
                            const double Ip = (1-xi)*previous_Ih[input.fault_index][s]
                                              +xi*previous_Ih[input.fault_index][s+1];
                            const double Cp = (1-xi)*previous_cohesive_traction[input.fault_index][s]
                                              +xi*previous_cohesive_traction[input.fault_index][s+1];
                            const auto &phase = this->get_phase_field_handler();
                            const double h = 1/phase.energetic_degradation({1.0},std::max(phi[q],0.0))-1;
                            const double hp = 1/phase.energetic_degradation({1.0},std::max(old_phi[q],0.0))-1;
                            chi=h/I;
                            history=std::exp(-this->get_timestep()/100.)*Cp/kappa*(h*Ip/I-hp);
                          }
                        else
                          {
                            const auto response = model.evaluate_reconstructed_fault_bulk_point(input);
                            chi=response.localization_factor; kappa=response.kappa; history=response.history_correction;
                          }
                      }
                    bulk << values.quadrature_point(q)[0] << ',' << values.quadrature_point(q)[1] << ',' << values.JxW(q)
                         << ',' << velocity[q][0] << ',' << velocity[q][1] << ',' << gradients[q][0][0] << ',' << gradients[q][0][1]
                         << ',' << gradients[q][1][0] << ',' << gradients[q][1][1] << ',' << pressure[q] << ',' << phi[q]
                         << ',' << old_stress[q] << ',' << association.active << ',' << V << ',' << chi << ',' << kappa << ',' << history << '\n';
                  }
              }
          // Replicate only the initial diagnostic snapshots: particles may
          // migrate, and relevant constraint lines extend beyond owned cells.
          // Output stays rank-local, avoiding concurrent writes to one file.
          if (step == 0 && distributed)
            {
              for (const auto &snapshot : Utilities::MPI::all_gather(communicator, frozen_phi))
                frozen_phi.insert(snapshot.begin(), snapshot.end());
              for (const auto &snapshot : Utilities::MPI::all_gather(communicator, frozen_H))
                frozen_H.insert(snapshot.begin(), snapshot.end());
            }
          if (evolving_profile)
            {
              // Flush accepted exports before the read-only bounded gate. A
              // failed gate stops this smoke before another timestep starts.
              nodes.close(); surface.close(); segments.close(); particles.close();
              bulk.close(); times.close(); weak_output.close(); transfer.close(); qp_cells.close();
              const std::string command = std::string("OPENBLAS_NUM_THREADS=1 timeout 30 python3 \"")
                +ASPECT_SOURCE_DIR+"/benchmarks/reconstructed_fault/uniform_shear/evolving/check_smoke.py\" \""
                +this->get_output_directory()+"\" --step "+std::to_string(step);
              AssertThrow(std::system(command.c_str())==0,
                          ExcMessage("K3 evolving smoke gate failed; inspect guard JSON. Do not retry."));
            }
          return {"K1 accepted-state export:", std::to_string(step)};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(UniformShear, "uniform shear pilot",
                                  "Freeze the initial Q1 phase field and export rank-local K1 diagnostics.")
  }
}
