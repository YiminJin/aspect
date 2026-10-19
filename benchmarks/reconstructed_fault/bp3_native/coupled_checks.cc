/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/postprocess/interface.h>
#include <aspect/simulator.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/phase_field_fault/dg_history_internal.h>
#include <aspect/particle/manager.h>
#include <aspect/particle/integrator/rk_2.h>
#include <aspect/particle/particle_domain.h>
#include <aspect/particle/interpolator/interface.h>
#include <aspect/phase_field.h>
#include <aspect/time_stepping/interface.h>
#include <aspect/initial_composition/interface.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/reconstructed_fault/linear_performance.h>
#include <aspect/simulator/assemblers/reconstructed_fault_stokes.h>
#include <aspect/simulator/solver/reconstructed_fault_condensed_system.h>
#include <aspect/utilities.h>
#include "../../../tests/phase_field_fault_test_access.h"
#include "../bp3/plugin/bp3_model.h"
#include "../bp3/plugin/runtime.h"
#include "server/plugin/first_event.h"
#include "server/plugin/output_schedule.h"
#include "server/plugin/output_files.h"
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools_evaluate.h>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <set>
#include <sstream>

namespace aspect
{
  namespace TimeStepping
  {
    template<int dim> class BP3OneRejection : public Interface<dim>, public SimulatorAccess<dim>
    {
      bool rejected=false;
    public:
      double execute() override {return std::numeric_limits<double>::max();}
      std::pair<Reaction,double> determine_reaction(const TimeStepInfo &) override
      {
        if(this->get_timestep_number()==2 && !rejected)
          {
            rejected=true;
            return {Reaction::repeat_step,.5*this->get_timestep()};
          }
        return {Reaction::advance,std::numeric_limits<double>::max()};
      }
    };
    ASPECT_REGISTER_TIME_STEPPING_MODEL(BP3OneRejection,"BP3 one outer rejection",
      "Bounded mature-history test: reject the completed step-two trial once and halve its dt.")
  }
  namespace Particle
  {
    namespace Interpolator
    {
      /** Read-only probe around the maintained BP3 interpolator, including births. */
      template <int dim>
      class ManagedHistoryProbe : public Interface<dim>, public SimulatorAccess<dim>
      {
          const Simulator<dim> *simulator = nullptr;
          std::unique_ptr<Interface<dim>> native;
        public:
          mutable Point<dim> last_point;
          mutable std::vector<double> last_values;
          void initialize_simulator(const Simulator<dim> &sim) override
          {
            SimulatorAccess<dim>::initialize_simulator(sim);
            simulator=&sim;
          }
          static void declare_parameters(ParameterHandler &) {}
          void parse_parameters(ParameterHandler &prm) override
          {
            const auto name=prm.get("Interpolation scheme");
            prm.set("Interpolation scheme","BP3 history linear least squares");
            native=create_particle_interpolator<dim>(prm);
            prm.set("Interpolation scheme",name);
            dynamic_cast<SimulatorAccess<dim> &>(*native).initialize_simulator(*simulator);
            native->set_particle_manager_index(this->get_particle_manager_index());
            native->parse_parameters(prm);
          }
          void initialize() override
          {
            native->initialize();
          }
          std::vector<std::vector<double>> properties_at_points(
            const ParticleHandler<dim> &handler,const std::vector<Point<dim>> &points,
            const ComponentMask &mask,
            const typename parallel::distributed::Triangulation<dim>::active_cell_iterator &cell) const override
          {
            auto values=native->properties_at_points(handler,points,mask,cell);
            if (points.size()==1)
              {
                last_point=points[0];
                last_values=values[0];
              }
            return values;
          }
      };
      ASPECT_REGISTER_PARTICLE_INTERPOLATOR(ManagedHistoryProbe,"BP3 managed history probe",
                                            "Read-only observations around the maintained BP3 LLS, including native late initialization.")
    }
  }
  namespace Postprocess
  {
    // Existing test friendship, loaded separately from the manufactured observer.
    // Capture incoming anchors before mechanics, then audit native publication.
    template <int dim>
    class MomentCycle : public Interface<dim>, public SimulatorAccess<dim>
    {
        using Model = MaterialModel::PhaseFieldFault<dim>;
        using Access = MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
        using BV = LinearAlgebra::BlockVector;
        struct ParticleAnchor
        {
          types::particle_index id;
          Point<dim> position;
          CellId cell;
          SymmetricTensor<2,dim> stress;
        };
        BV initial_phase, incoming_fe;
        std::shared_ptr<typename Model::DGHistory> incoming_dg, incoming_dg_snapshot;
        std::vector<double> initial_I, incoming_Z, previous_slip;
        std::map<types::particle_index, std::pair<Point<dim>,double> > fixed_particles;
        std::vector<ParticleAnchor> anchors;
        unsigned int incoming_step = numbers::invalid_unsigned_int;
        unsigned int solver_publications = 0;
        unsigned int last_output = numbers::invalid_unsigned_int;
        bool write_authoritative_stress = false;
        struct CommonPoints
        {
          std::vector<Point<dim>> physical;
          std::vector<double> weights;
          std::vector<unsigned int> ids;
        };
        std::string common_points_file;
        std::map<std::string,CommonPoints> common_points;
        bool write_coefficients = false, inject_failure = false, failure_done = false, material_only = false;
        bool profile_only = false, prescribed_schedule = true, lifecycle = false, nonlinear_failure = false;
        bool scheduled_fields = false;
        BP3::OutputSchedule field_schedule;
        BP3::FirstEvent field_event;
        std::vector<std::vector<double>> incoming_surface;
        std::map<types::particle_index,std::vector<double>> incoming_particles;
        double incoming_dt=0.,incoming_time=0.;
        std::uint64_t incoming_publications=0;
        std::vector<double> incoming_rates;
        bool moving_particles = false, moving_attempt = false;
        bool population_event_audit = false;
        std::string audit_filter_mode;
        double audit_filter_length = 0.;
        std::unique_ptr<ParticleAnchor> face_probe;
        class PopulationAuditRK2 : public Particle::Integrator::RK2<dim>
        {
            std::function<void()> observe;
          public:
            PopulationAuditRK2(const Particle::Integrator::RK2<dim> &native, std::function<void()> callback)
              : Particle::Integrator::RK2<dim>(native), observe(std::move(callback)) {}
            bool new_integration_step() override
            {
              const bool again = Particle::Integrator::RK2<dim>::new_integration_step();
              if (!again) observe();
              return again;
            }
        };
        struct ManagedAnchor
        {
          Point<dim> position;
          CellId cell;
          std::array<double,4> history;
        };
        std::map<types::particle_index,ManagedAnchor> managed_backup;
        std::set<types::particle_index> managed_births;
        unsigned int managed_lost=0, managed_removed=0, managed_crossings=0;
        unsigned int minimum_particles=0, maximum_particles=0;
        unsigned long total_births=0,total_removed=0,total_crossings=0;
        double maximum_motion=0, maximum_birth_stress=0;
        std::uint64_t incoming_domain_version=0;
        std::vector<double> incoming_theta;
        unsigned int tangent_step=numbers::invalid_unsigned_int;
        std::shared_ptr<const InitialComposition::Manager<dim> > initial_composition;

        // A scoped native reconstruction/assembly with immutable mechanical
        // fields. Only derived caches and temporary stress blocks are rebuilt.
        void audit_fixed_fields(const std::string &label, const bool save_particles)
        {
          auto &sim = const_cast<Simulator<dim> &>(this->get_simulator());
          auto &pm = const_cast<Particle::Manager<dim> &>(this->get_phase_field_handler().get_associated_particle_manager());
          auto &domains = const_cast<Particle::ParticleDomainHandler<dim> &>(pm.get_particle_domain_handler());
          const BV saved(sim.solution), saved_old(sim.old_solution);
          const auto &manager = this->get_reconstructed_fault_manager();
          const auto &fault = manager.get_fault(0);
          const auto rates = manager.get_timestep_committed_slip_rate(0);
          const auto &model = Plugins::get_plugin_as_type<const Model>(this->get_material_model());
          const auto saved_I = Access::current_normalization_integrals(model);
          domains.generate_particle_domains();
          std::vector<AdvectionField> fields;
          std::set<unsigned int> stress_blocks;
          for (const auto &entry : this->get_parameters().mapped_particle_properties)
            if (entry.second.first == "maxwell stress")
              {
                fields.push_back(AdvectionField::composition(entry.first));
                stress_blocks.insert(this->introspection().block_indices.compositional_fields[entry.first]);
              }
          try
            {
              sim.interpolate_particle_properties(fields);
              for (unsigned int block = 0; block < saved.n_blocks(); ++block)
                if (!stress_blocks.count(block))
                  AssertThrow(saved.block(block) == sim.solution.block(block), ExcMessage("Fixed-field audit changed a mechanical field."));
              ReconstructedFaultSurfaceSystem<dim> probe(this->get_simulator());
              probe.enable_bulk_work_measure();
              probe.set_normal_stress_filter(audit_filter_mode, audit_filter_length);
              probe.set_normal_traction_diagnostic([](const Point<dim> &p)
              {
                return BP3::down_dip(p[0], p[1]) > BP3::geometry().length - 200.;
              });
              const auto &weak = probe.linearize_surface_system(sim.solution, {rates});
              const auto &diagnostic = probe.get_normal_traction_diagnostic();
              const auto project = [&weak](const std::vector<double> &load)
              {
                return ReconstructedFaultUtilities::solve_tridiagonal_system(weak.mass_diagonal[0], weak.mass_off_diagonal[0], load);
              };
              const auto raw = project(weak.raw_normal_traction[0]), filtered = project(weak.normal_traction[0]);
              const auto shear = project(weak.shear_traction[0]), pressure = project(diagnostic.pressure_load[0]);
              const auto tensor = project(diagnostic.deviatoric_load[0]);
              const auto theta_position = manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
              std::ofstream load(local_file(label + "-loads.csv"));
              load << std::setprecision(17) << "node,s_m,V,theta,weak_residual,weak_shear,weak_friction,weak_raw_normal,weak_normal,weak_pressure,weak_tensor,mass_diagonal,mass_edge,raw_Pa,filtered_Pa,shear_Pa,pressure_Pa,tensor_Pa\n";
              for (unsigned int i = 0; i < rates.size(); ++i)
                load << i << ',' << BP3::down_dip(fault.vertex(i)[0], fault.vertex(i)[1]) << ',' << rates[i] << ',' << fault.get_properties(i)[theta_position]
                     << ',' << weak.values[0][i] << ',' << weak.shear_traction[0][i] << ',' << weak.friction_traction[0][i]
                     << ',' << weak.raw_normal_traction[0][i] << ',' << weak.normal_traction[0][i] << ',' << diagnostic.pressure_load[0][i]
                     << ',' << diagnostic.deviatoric_load[0][i] << ',' << weak.mass_diagonal[0][i] << ',' << (i + 1 < rates.size() ? weak.mass_off_diagonal[0][i] : 0.)
                     << ',' << raw[i] << ',' << filtered[i] << ',' << shear[i] << ',' << pressure[i] << ',' << tensor[i] << '\n';
              std::ofstream qp(local_file(label + "-qp.csv"));
              qp << std::setprecision(17) << "cell,q,x,y,weight,phi,chi,pressure,old_xx,old_yy,old_xy,lls_xx,lls_yy,lls_xy,stress_xx,stress_yy,stress_xy\n";
              for (const auto &q : diagnostic.samples)
                qp << q.cell << ',' << q.qp << ',' << q.position[0] << ',' << q.position[1] << ',' << q.weight << ',' << q.phase << ',' << q.chi << ',' << q.pressure
                   << ',' << q.incoming_stress[0][0] << ',' << q.incoming_stress[1][1] << ',' << q.incoming_stress[0][1]
                   << ',' << q.particle_interpolated_stress[0][0] << ',' << q.particle_interpolated_stress[1][1] << ',' << q.particle_interpolated_stress[0][1]
                   << ',' << q.stress[0][0] << ',' << q.stress[1][1] << ',' << q.stress[0][1] << '\n';
              if (save_particles)
                {
                  std::ofstream particles(local_file(label + "-particles.csv"));
                  particles << std::setprecision(17) << "id,cell,x,y";
                  const auto count = pm.get_property_manager().get_data_info().n_components();
                  for (unsigned int j = 0; j < count; ++j) particles << ",p" << j;
                  particles << '\n';
                  for (const auto &p : pm.get_particle_handler())
                    {
                      particles << p.get_id() << ',' << p.get_surrounding_cell()->id() << ',' << p.get_location()[0] << ',' << p.get_location()[1];
                      for (const double value : p.get_properties()) particles << ',' << value;
                      particles << '\n';
                    }
                  std::ofstream state(local_file(label + "-fe.csv"));
                  state << std::setprecision(17) << "block,index,frozen,reconstructed\n";
                  for (unsigned int block = 0; block < saved.n_blocks(); ++block)
                    for (const auto i : saved.block(block).locally_owned_elements())
                      state << block << ',' << i << ',' << saved.block(block)[i] << ',' << sim.solution.block(block)[i] << '\n';
                  std::ofstream rng(local_file(label + "-rng.txt"));
                  rng << pm.get_random_number_state();
                }
              AssertThrow(load && qp && Access::current_normalization_integrals(model) == saved_I
                          && manager.get_timestep_committed_slip_rate(0) == rates, ExcMessage("Fixed-state audit changed history or failed to write."));
            }
          catch (...)
            {
              sim.solution = saved;
              throw;
            }
          sim.solution = saved;
          AssertThrow(sim.old_solution == saved_old, ExcMessage("Audit changed accepted old bulk fields."));
        }

        void audit_near_face(const ParticleAnchor &a)
        {
          const auto &model = Plugins::get_plugin_as_type<const Model>(this->get_material_model());
          const auto &intro = this->introspection();
          const auto &manager = this->get_reconstructed_fault_manager();
          const auto &fault = manager.get_fault(0);
          const auto coefficients = Model::compute_maxwell_coefficients(model.reference_viscosities[0], model.elastic_shear_moduli[0], this->get_timestep());
          const auto tria_cell = this->get_triangulation().create_cell_iterator(a.cell);
          const typename DoFHandler<dim>::active_cell_iterator cell(&this->get_triangulation(), tria_cell->level(), tria_cell->index(), &this->get_dof_handler());
          auto unit = this->get_mapping().transform_real_to_unit_cell(cell, a.position);
          unsigned int direction = 0;
          double distance = 1.;
          for (unsigned int d = 0; d < dim; ++d)
            if (std::min(unit[d], 1. - unit[d]) < distance)
              {
                direction = d;
                distance = std::min(unit[d], 1. - unit[d]);
              }
          const double face = unit[direction] < .5 ? 0. : 1.;
          std::vector<Point<dim>> points;
          const std::vector<double> offsets = { -4.e-6, -1.1e-6, -.9e-6, 0., .9e-6, 1.1e-6, 4.e-6};
          for (const double offset : offsets)
            {
              auto p = unit;
              p[direction] = face + offset;
              points.push_back(this->get_mapping().transform_unit_to_real_cell(cell, p));
            }
          points.push_back(a.position);
          Utilities::MPI::RemotePointEvaluation<dim> cache;
          cache.reinit(this->get_phase_field_handler().get_grid_cache(), points);
          const auto gradients = VectorTools::point_gradients<dim>(cache, this->get_dof_handler(), this->get_solution(), VectorTools::EvaluationFlags::avg, intro.component_indices.velocities[0]);
          const auto phases = VectorTools::point_values<1>(cache, this->get_dof_handler(), this->get_solution(), VectorTools::EvaluationFlags::avg, intro.variable("phase_field").first_component_index);
          using Cache = Utilities::MPI::RemotePointEvaluation<dim>;
          const auto traces = cache.template evaluate_and_process<SymmetricTensor<2, dim>>(
                                [&](const ArrayView<SymmetricTensor<2, dim>> &out, const typename Cache::CellData & cells)
          {
            for (const auto c : cells.cell_indices())
              {
                const auto tria = cells.get_active_cell_iterator(c);
                const typename DoFHandler<dim>::active_cell_iterator dof(&this->get_triangulation(), tria->level(), tria->index(), &this->get_dof_handler());
                const auto refs = cells.get_unit_points(c);
                FEValues<dim> fe(this->get_mapping(), this->get_fe(), Quadrature<dim>(std::vector<Point<dim>>(refs.begin(), refs.end())), update_gradients);
                fe.reinit(dof);
                std::vector<SymmetricTensor<2, dim>> values(refs.size());
                fe[intro.extractors.velocities].get_function_symmetric_gradients(this->get_solution(), values);
                auto data = cells.get_data_view(c, out);
                std::copy(values.begin(), values.end(), data.begin());
              }
          });
          const auto expected = [&](const Point<dim> &p, const SymmetricTensor<2, dim> &strain, const double phi)
          {
            const auto source = manager.project_to_bulk_source(p);
            AssertThrow(source.active, ExcMessage("Face probe left the source support."));
            auto t = fault.vertex(source.segment_index + 1) - fault.vertex(source.segment_index);
            t /= t.norm();
            Tensor<1, dim> n;
            n[0] = -t[1];
            n[1] = t[0];
            const auto S = manager.get_shear_sense(0) * symmetrize(outer_product(t, n));
            const auto l = model.evaluate_reconstructed_fault_localization(0, source.segment_index, source.xi, phi, phi, "fixed face audit");
            return coefficients.beta * a.stress + 2 * coefficients.eta_ve * (Utilities::Tensors::consistent_deviator(strain)
                                                                             - l.current_h / l.current_I_h * manager.interpolate_slip_rate(0, source.segment_index, source.xi) * S);
          };
          std::ofstream sweep(local_file("face-34-sweep.csv"));
          sweep << std::setprecision(17) << "offset_host_reference,x,y,traces,FE_average_error_per_s,stress_xx,stress_yy,stress_xy\n";
          const auto &ptr = cache.get_point_ptrs();
          for (unsigned int i = 0; i < points.size(); ++i)
            {
              SymmetricTensor<2, dim> mean;
              for (unsigned int j = ptr[i]; j < ptr[i + 1]; ++j) mean += traces[j];
              mean /= ptr[i + 1] - ptr[i];
              const auto stress = expected(points[i], symmetrize(gradients[i]), phases[i]);
              sweep << (i < offsets.size() ? offsets[i] : unit[direction] - face) << ',' << points[i][0] << ',' << points[i][1] << ',' << ptr[i + 1] - ptr[i]
                    << ',' << (mean - symmetrize(gradients[i])).norm() << ',' << stress[0][0] << ',' << stress[1][1] << ',' << stress[0][1] << '\n';
            }
          FEValues<dim> fe(this->get_mapping(), this->get_fe(), Quadrature<dim>({unit}), update_values | update_gradients);
          fe.reinit(cell);
          std::vector<SymmetricTensor<2, dim>> strain(1);
          std::vector<double> phi(1);
          fe[intro.extractors.velocities].get_function_symmetric_gradients(this->get_solution(), strain);
          fe[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)].get_function_values(this->get_solution(), phi);
          const auto host = expected(a.position, strain[0], phi[0]);
          auto &pm = const_cast<Particle::Manager<dim> &>(this->get_phase_field_handler().get_associated_particle_manager());
          const auto position = pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
          audit_fixed_fields("face-34-native", false);
          for (auto &p : pm.get_particle_handler())
            if (p.get_id() == a.id)
              {
                std::array<double, 3> saved;
                for (unsigned int c = 0; c < 3; ++c)
                  {
                    saved[c] = p.get_properties()[position + c];
                    p.get_properties()[position + c] = host[SymmetricTensor<2, dim>::unrolled_to_component_indices(c)];
                  }
                try
                  {
                    audit_fixed_fields("face-34-host", false);
                  }
                catch (...)
                  {
                    for (unsigned int c = 0; c < 3; ++c) p.get_properties()[position + c] = saved[c];
                    throw;
                  }
                for (unsigned int c = 0; c < 3; ++c) p.get_properties()[position + c] = saved[c];
                std::ofstream identity(local_file("face-34-particle.txt"));
                identity << std::setprecision(17) << "id " << a.id << " cell " << a.cell << " position " << a.position << " unit " << unit
                         << "\nold " << a.stress << "\nhost " << host << "\nnative " << expected(a.position, symmetrize(gradients.back()), phases.back()) << '\n';
              }
        }

        void verify_moving_tangent()
        {
          const auto step=this->get_timestep_number();
          if ((step!=0 && step!=2) || tangent_step==step) return;
          tangent_step=step;
          const auto &surface=this->get_reconstructed_fault_surface_system();
          BV owned(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
          owned=0;
          bool selected=false;
          std::vector<types::global_dof_index> indices(this->get_fe().dofs_per_cell);
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned() && !selected
                && BP3::normal_distance(cell->center()[0],cell->center()[1])<.25*this->get_phase_field_handler().get_length_scale()
                && BP3::down_dip(cell->center()[0],cell->center()[1])>.25*BP3::geometry().length
                && BP3::down_dip(cell->center()[0],cell->center()[1])<.75*BP3::geometry().length
                && this->get_reconstructed_fault_manager().project_to_bulk_source(cell->center()).active)
              {
                cell->get_dof_indices(indices);
                for (unsigned int i=0; i<indices.size() && !selected; ++i)
                  if (this->get_fe().system_to_component_index(i).first==0
                      && !this->get_current_constraints().is_constrained(indices[i]))
                    {
                      owned[indices[i]]=1.e-13;
                      selected=true;
                    }
              }
          AssertThrow(selected,ExcMessage("No interior nonzero-trace velocity direction for G check."));
          owned.compress(VectorOperation::insert);
          BV direction(this->introspection().index_sets.system_partitioning,
                       this->introspection().index_sets.system_relevant_partitioning,this->get_mpi_communicator());
          direction=owned;
          typename ReconstructedFaultSurfaceSystem<dim>::FaultVector action;
          surface.apply_G(direction,action);
          const auto &rates=surface.get_normal_traction_diagnostic().rates;
          BV plus_owned(owned),minus_owned(owned);
          plus_owned=this->get_current_linearization_point();
          minus_owned=plus_owned;
          plus_owned.add(.25,owned);
          minus_owned.add(-.25,owned);
          BV plus(direction),minus(direction);
          plus=plus_owned;
          minus=minus_owned;
          const auto rp=surface.evaluate_surface_residual(plus,rates),rm=surface.evaluate_surface_residual(minus,rates);
          double error=0,scale=0;
          for (unsigned int f=0; f<action.size(); ++f)
            for (unsigned int j=0; j<action[f].size(); ++j)
              {
                const double difference=2*(rp.values[f][j]-rm.values[f][j]);
                error=std::max(error,std::abs(action[f][j]-difference));
                scale=std::max(scale,std::max(std::abs(action[f][j]),std::abs(difference)));
              }
          AssertThrow(scale>0 && error<2.e-7*scale,ExcMessage("Deviatoric filtered G fails its off-trace finite difference."));
          std::ofstream out(local_file("moving-tangent.csv"),step ? std::ios::app : std::ios::out);
          if (!step) out<<"step,time_s,absolute_error,scale,relative_error\n";
          out<<std::setprecision(17)<<step<<','<<this->get_time()<<','<<error<<','<<scale<<','<<error/scale<<'\n';
        }

        void backup_moving_particles()
        {
          const auto &pm=this->get_phase_field_handler().get_associated_particle_manager();
          const auto &info=pm.get_property_manager().get_data_info();
          const auto s=info.get_position_by_field_name("maxwell stress"),H=info.get_position_by_field_name("crack_driving_force");
          managed_backup.clear();
          managed_births.clear();
          managed_lost=managed_removed=managed_crossings=0;
          maximum_motion=maximum_birth_stress=0;
          moving_attempt=false;
          for (const auto &p : pm.get_particle_handler())
            managed_backup.emplace(p.get_id(),ManagedAnchor {p.get_location(),p.get_surrounding_cell()->id(),
              {{p.get_properties()[H],p.get_properties()[s],p.get_properties()[s+1],p.get_properties()[s+2]}}
            });
        }

        void verify_moving_management()
        {
          const auto &pm=this->get_phase_field_handler().get_associated_particle_manager();
          const auto &info=pm.get_property_manager().get_data_info();
          const auto s=info.get_position_by_field_name("maxwell stress"),H=info.get_position_by_field_name("crack_driving_force");
          unsigned int survivors=0;
          for (const auto &p : pm.get_particle_handler())
            if (!managed_births.count(p.get_id()))
              {
                const auto &old=managed_backup.at(p.get_id());
                AssertThrow(old.history[0]==p.get_properties()[H],ExcMessage("Advection/management changed survivor H."));
                for (unsigned int c=0; c<3; ++c)
                  AssertThrow(old.history[c+1]==p.get_properties()[s+c],ExcMessage("Advection/management changed survivor stress."));
                maximum_motion=std::max(maximum_motion,p.get_location().distance(old.position));
                if (p.get_surrounding_cell()->id()!=old.cell) ++managed_crossings;
                ++survivors;
              }
          AssertThrow(managed_backup.size()>=survivors+managed_lost,ExcInternalError());
          managed_removed=managed_backup.size()-survivors-managed_lost;
          total_births+=managed_births.size();
          total_removed+=managed_removed;
          total_crossings+=managed_crossings;
          incoming_domain_version=pm.get_particle_domain_handler().geometry_version();
          capture_incoming(true);
        }

        void write_moving_check()
        {
          const auto &pm=this->get_phase_field_handler().get_associated_particle_manager();
          auto &manager=const_cast<ReconstructedFaultManager<dim> &>(this->get_reconstructed_fault_manager());
          const auto &associations=manager.get_locally_owned_particle_fault_associations();
          AssertThrow(associations.size()==fixed_particles.size(),ExcMessage("Moving association population is stale."));
          const auto version=pm.get_particle_domain_handler().geometry_version();
          AssertThrow(version>incoming_domain_version,ExcMessage("Moved particle domains were not rebuilt."));
          long double volume=0;
          double partition_error=0,linear_error=0;
          unsigned int index=0,minimum=std::numeric_limits<unsigned int>::max(),maximum=0;
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                const auto count=pm.get_particle_handler().n_particles_in_cell(cell);
                minimum=std::min(minimum,unsigned(count));
                maximum=std::max(maximum,unsigned(count));
              }
          for (const auto &p : pm.get_particle_handler())
            {
              const auto domain=pm.get_particle_domain_handler().get_particle_domain(p.get_local_index());
              const auto &a=associations[index++];
              AssertThrow(a.particle_id==p.get_id() && a.position==p.get_location()
                          && a.particle_domain_volume==domain.volume(),ExcMessage("Moving association geometry/measure is stale."));
              volume+=domain.volume();
              double sum=0;
              Tensor<2,dim> gradient;
              for (unsigned int i=0; i<domain.n_relevant_vertices(); ++i)
                {
                  sum+=domain.weighting_function_value(i);
                  const auto offset=this->get_triangulation().get_vertices()[domain.relevant_vertex_index(i)]-p.get_location();
                  for (unsigned int d=0; d<dim; ++d) gradient[d]+=offset[d]*domain.weighting_function_gradient(i);
                }
              partition_error=std::max(partition_error,std::abs(sum-1));
              for (unsigned int d=0; d<dim; ++d) gradient[d][d]-=1;
              linear_error=std::max(linear_error,gradient.norm());
            }
          const double coverage=std::abs(double(volume)/(BP3::geometry().extent[0]*BP3::geometry().extent[1])-1);
          AssertThrow(coverage<1.e-10 && partition_error<1.e-10 && linear_error<1.e-10 && minimum>=minimum_particles && maximum<=maximum_particles,
                      ExcMessage("Managed particle geometry, CPDI or population bounds failed."));
          std::ofstream out(local_file("moving-particles.csv"),this->get_timestep_number() ? std::ios::app : std::ios::out);
          if (!this->get_timestep_number())
            out<<"step,time_s,dt_s,before,after,births,removed,outflow,crossings,total_births,total_removed,total_crossings,min_per_cell,max_per_cell,max_motion_m,max_birth_stress_Pa,domain_version,coverage_relative,partition_error,linear_gradient_error,associations,survivor_history_error\n";
          out<<std::setprecision(17)<<this->get_timestep_number()<<','<<this->get_time()<<','<<this->get_timestep()<<','
             <<managed_backup.size()<<','<<fixed_particles.size()<<','<<managed_births.size()<<','<<managed_removed<<','<<managed_lost<<','
             <<managed_crossings<<','<<total_births<<','<<total_removed<<','<<total_crossings<<','<<minimum<<','<<maximum<<','
             <<maximum_motion<<','<<maximum_birth_stress<<','<<version<<','<<coverage<<','<<partition_error<<','<<linear_error<<','<<associations.size()<<",0\n";
        }

        // Rank-local observations have unique files; replicated surface output
        // is written only by rank zero. A serial filename remains unchanged.
        std::string local_file(const std::string &name) const
        {
          const auto comm=this->get_mpi_communicator();
          return this->get_output_directory()+
                 (Utilities::MPI::n_mpi_processes(comm)==1 ? "" :
                  "rank-"+std::to_string(Utilities::MPI::this_mpi_process(comm))+"-")+name;
        }

        void verify_candidate_rejection()
        {
          // Use the native nonpublishing mechanical solve and history stages.
          // Arm a bookkeeping guard; physical accepted histories remain immutable.
          auto &sim=const_cast<Simulator<dim>&>(this->get_simulator());
          auto &model=const_cast<Model &>(Plugins::get_plugin_as_type<const Model>(this->get_material_model()));
          auto &manager=this->get_reconstructed_fault_manager();
          const auto comm=this->get_mpi_communicator();
          const BV saved_solution(this->get_solution()),saved_linear(this->get_current_linearization_point()),
                saved_old(this->get_old_solution()),saved_older(this->get_old_old_solution()),
                saved_fe(model.accepted_fe_strain_state());
          const auto saved_Z=Access::fe_slip_memory(model);
          const auto saved_I=Access::current_normalization_integrals(model);
          const auto saved_step=model.fe_strain_state->committed_step;
          const auto saved_publications=model.fe_strain_state->accepted_publications;
          const auto saved_dg=model.uses_dg_trial_stress_history()
            ? std::make_shared<typename Model::DGHistory>(*model.fe_strain_state->dg_history) : nullptr;
          const auto saved_V=manager.get_timestep_committed_slip_rate(0);
          const auto saved_pressure=sim.last_pressure_normalization_adjustment;
          const auto saved_iteration=sim.nonlinear_iteration;
          const auto saved_statistics=sim.statistics;
          std::vector<std::vector<double> > properties;
          const auto &fault=manager.get_fault(0);
          for (unsigned int j=0; j<fault.n_vertices(); ++j)
            {
              const auto data=fault.get_properties(j);
              properties.emplace_back(data.begin(),data.end());
            }
          auto &pm=this->get_phase_field_handler().get_associated_particle_manager();
          std::map<types::particle_index,std::vector<double> > particles;
          for (const auto &p:pm.get_particle_handler())
            {
              const auto data=p.get_properties();
              particles.emplace(p.get_id(),std::vector<double>(data.begin(),data.end()));
            }
          ReconstructedFaultVector rates;
          sim.solve_reconstructed_fault_stokes(false,&rates);
          manager.begin_slip_rate_nonlinear_solve();
          manager.begin_slip_rate_trial();
          manager.set_slip_rate_trial_values(rates);
          manager.accept_slip_rate_trial();
          const unsigned int owner=Utilities::MPI::n_mpi_processes(comm)-1;
          const bool selected=Utilities::MPI::this_mpi_process(comm)==owner;
          // Arm a mismatch only in one rank's fixed-sample registry. No
          // physical particle moves. Native history candidates are built first;
          // FE staging then collectively checks this registry before proceeding
          // to vector collectives or terminal publication on any rank.
          auto marker=model.fe_strain_state->particles.begin();
          AssertThrow(marker!=model.fe_strain_state->particles.end(),ExcMessage("Failure owner has no fixed sample."));
          const auto saved_position=marker->second;
          if (selected)
            marker->second[0]+=1.;
          bool caught=false;
          try
            {
              model.commit_reconstructed_fault_mechanical_history(this->get_solution());
            }
          catch (const std::exception &e)
            {
              caught=std::string(e.what()).find("Accepted FE strain rejects history transport")!=std::string::npos;
            }
          AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(caught),comm)==1,
                      ExcMessage("One-owner FE staging rejection did not reach every rank."));
          auto expected_marker=saved_position;
          if (selected) expected_marker[0]+=1.;
          AssertThrow(marker->second==expected_marker,
                      ExcMessage("Failed staging overwrote its registry-marker input."));
          marker->second=saved_position;
          // Only the injected registry entry was restored. Physical accepted
          // histories and particle/surface property bytes must be unchanged
          // before restoring any scratch mechanical fields.
          bool unchanged=model.accepted_fe_strain_state()==saved_fe
                         && Access::fe_slip_memory(model)==saved_Z
                         && Access::current_normalization_integrals(model)==saved_I
                         && model.fe_strain_state->committed_step==saved_step
                         && manager.get_timestep_committed_slip_rate(0)==saved_V;
          for (unsigned int j=0; j<fault.n_vertices(); ++j)
            {
              const auto data=fault.get_properties(j);
              unchanged &= data.size()==properties[j].size()
                           && std::memcmp(data.data(),properties[j].data(),data.size()*sizeof(double))==0;
            }
          for (const auto &p:pm.get_particle_handler())
            {
              const auto data=p.get_properties();
              const auto &saved=particles.at(p.get_id());
              unchanged &= data.size()==saved.size() && std::memcmp(data.data(),saved.data(),data.size()*sizeof(double))==0;
            }
          unchanged=unchanged && model.fe_strain_state->accepted_publications==saved_publications;
          if(saved_dg)for(unsigned int c=0;c<saved_dg->trial_stress.size();++c)
            for(const auto i:saved_dg->trial_stress[c].locally_owned_elements())
              unchanged=unchanged && model.fe_strain_state->dg_history->trial_stress[c][i]==saved_dg->trial_stress[c][i];
          AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(unchanged),comm)==1,
                      ExcMessage("Rejected candidate changed an accepted history."));
          manager.rollback_slip_rate_nonlinear_solve();
          sim.solution=saved_solution;
          sim.current_linearization_point=saved_linear;
          sim.last_pressure_normalization_adjustment=saved_pressure;
          sim.nonlinear_iteration=saved_iteration;
          sim.statistics=saved_statistics;
          sim.rebuild_stokes_matrix=sim.rebuild_stokes_preconditioner=true;
          AssertThrow(this->get_solution()==saved_solution && this->get_current_linearization_point()==saved_linear
                      && this->get_old_solution()==saved_old && this->get_old_old_solution()==saved_older
                      && manager.get_slip_rate(0)==saved_V && solver_publications==0,
                      ExcMessage("Candidate-stage rollback did not restore scratch mechanics."));
          std::ofstream out(local_file("candidate-rejection.csv"));
          out<<"step,time_s,dt_s,owner,caught,accepted_histories_unchanged,published_before_retry\n"
             <<this->get_timestep_number()<<','<<this->get_time()<<','<<this->get_timestep()<<','<<owner<<",1,1,0\n";
          this->get_pcout()<<"   BP3 one-owner complete candidate rejection: accepted histories unchanged; identical-dt retry follows."<<std::endl;
        }

        std::pair<std::string,std::string> verify_transition()
        {
#ifndef ASPECT_BP3_LOCAL_BOTTOM_TEST
          const auto &model=Plugins::get_plugin_as_type<const Model>(this->get_material_model());
          const auto &law=model.get_fault_friction();
          const auto &g=BP3::geometry();
          const double target=law.friction_coefficient({0.,1.},BP3::Vinit,law.get_characteristic_slip_distance()/BP3::Vinit);
          AssertThrow(std::abs(BP3::sigma0*target+BP3::damping*BP3::Vinit-BP3::tau0)
                      <128*std::numeric_limits<double>::epsilon()*BP3::tau0,
                      ExcMessage("Configured steady friction and background shear disagree."));
          AssertThrow(g.weakening_length==15000. && BP3::fraction(15000.)==0. && BP3::fraction(18000.)==1.,
                      ExcMessage("The ordinary BP3 15--18 km transition is not active."));
          double max_a_error=0.,max_f_error=0.,max_state_error=0.;
          std::ofstream out(this->get_output_directory()+"transition-material.csv");
          out<<std::setprecision(17)<<"xd_m,bulk_y_m,fraction,a_expected,a_native,theta_s,friction,background_shear_Pa\n";
          for (const double xd :
          {
            0.,14999.,15000.,15750.,16500.,17250.,18000.,18001.,20000.
          })
          {
            const double y=g.upper[1]-xd*g.sine;
            const double f=BP3::fraction(xd),bulk=BP3::depth_fraction(y);
            const double theta=BP3::configured_initial_state(xd,law);
            Point<dim> point;
            for (unsigned int d=0; d<2; ++d) point[d]=g.upper[d]+xd*g.tangent[d];
            const auto &initial=*initial_composition;
            const auto strengthening=this->introspection().compositional_index_for_name("strengthening");
            const auto theta_field=this->introspection().compositional_index_for_name("theta_initial");
            const double assigned_fraction=initial.initial_composition(point,strengthening);
            const double assigned_theta=initial.initial_composition(point,theta_field);
            AssertThrow(std::abs(assigned_fraction-f)<1.e-14 && std::abs(assigned_theta-theta)<1.e-13*theta,
                        ExcMessage("Native initial-composition assignment differs from the transition chart/state."));
            const double a=BP3::direct_effect(xd);
            const double actual_a=BP3::Vinit*law.friction_coefficient_derivative_wrt_slip_rate({1-f,f},BP3::Vinit,theta);
            const double friction=law.friction_coefficient({1-f,f},BP3::Vinit,theta);
            AssertThrow(std::abs(f-bulk)<1.e-14 && std::isfinite(theta) && theta>0.,ExcMessage("Bulk/surface transition chart or state is invalid."));
            max_a_error=std::max(max_a_error,std::abs(a-actual_a));
            max_f_error=std::max(max_f_error,std::abs(target-friction));
            out<<xd<<','<<y<<','<<f<<','<<a<<','<<actual_a<<','<<theta<<','<<friction<<','<<BP3::tau0<<'\n';
          }
          const auto &manager=this->get_reconstructed_fault_manager();
          const auto &fault=manager.get_fault(0);
          const auto state=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
          const auto background=manager.get_property_information()[manager.get_property_index("background tractions")].position;
          const auto correction=manager.get_property_information()[manager.get_property_index("BP3 fixed shear correction")].position;
          std::ofstream surface(this->get_output_directory()+"surface-initialization.csv");
          surface<<std::setprecision(17)<<"node,s_m,fraction,a,theta_s,friction,balance_Pa,I_h_m\n";
          for (unsigned int j=0; j<fault.n_vertices(); ++j)
            {
              const auto p=fault.get_properties(j);
              const auto xd=BP3::down_dip(fault.vertex(j)[0],fault.vertex(j)[1]);
              const double expected=BP3::configured_initial_state(xd,law);
              max_state_error=std::max(max_state_error,std::abs(p[state]-expected)/expected);
              AssertThrow(p[background]==BP3::tau0 && p[background+1]==BP3::sigma0
                          && p[correction]==0. && p[correction+1]==0.,ExcMessage("BP3 background stress was altered during initialization."));
              const auto composition=model.fault_property_indices.chemical_compositions.at(0);
              const auto pos=manager.get_property_information()[composition].position;
              AssertThrow(std::abs(p[pos]-BP3::fraction(xd))<1.e-13,ExcMessage("Surface material fractions do not match the native bulk assignment."));
              const std::vector<double> fractions{1-p[pos],p[pos]};
              const double mu=law.friction_coefficient(fractions,BP3::Vinit,p[state]);
              const double a=BP3::Vinit*law.friction_coefficient_derivative_wrt_slip_rate(fractions,BP3::Vinit,p[state]);
              const double balance=BP3::sigma0*mu+BP3::damping*BP3::Vinit-BP3::tau0;
              AssertThrow(std::abs(balance)<1e-7 && std::abs(a-BP3::direct_effect(xd))<1e-14,
                          ExcMessage("Initialized native surface law does not balance BP3 prestress."));
              surface<<j<<','<<xd<<','<<p[pos]<<','<<a<<','<<p[state]<<','<<mu<<','<<balance<<','
                     <<Access::current_normalization_integrals(model).at(0)[j]<<'\n';
            }
          std::ofstream interior(this->get_output_directory()+"surface-initialization-interior.csv");
          interior<<std::setprecision(17)<<"segment,xi,s_m,fraction,fraction_analytic,theta_s,theta_analytic_s,a,friction,balance_Pa\n";
          const QGauss<1> gauss(3);
          const auto composition=model.fault_property_indices.chemical_compositions.at(0);
          const auto pos=manager.get_property_information()[composition].position;
          for(unsigned int j=0;j<fault.n_cells();++j)
            for(unsigned int q=0;q<gauss.size();++q)
              {
                const double xi=gauss.point(q)[0];
                const auto point=(1-xi)*fault.vertex(j)+xi*fault.vertex(j+1);
                const double xd=BP3::down_dip(point[0],point[1]);
                const double fraction=(1-xi)*fault.get_properties(j)[pos]+xi*fault.get_properties(j+1)[pos];
                const double theta=(1-xi)*fault.get_properties(j)[state]+xi*fault.get_properties(j+1)[state];
                const std::vector<double> fractions{1-fraction,fraction};
                const double mu=law.friction_coefficient(fractions,BP3::Vinit,theta);
                const double a=BP3::Vinit*law.friction_coefficient_derivative_wrt_slip_rate(fractions,BP3::Vinit,theta);
                interior<<j<<','<<xi<<','<<xd<<','<<fraction<<','<<BP3::fraction(xd)<<','<<theta<<','
                        <<BP3::configured_initial_state(xd,law)<<','<<a<<','<<mu<<','
                        <<BP3::sigma0*mu+BP3::damping*BP3::Vinit-BP3::tau0<<'\n';
              }
          const auto &pm=this->get_phase_field_handler().get_associated_particle_manager();
          const auto &info=pm.get_property_manager().get_data_info();
          const auto bulk_fraction=info.get_position_by_field_name("initial strengthening");
          const bool has_particle_state=info.fieldname_exists("phase field fault state");
          AssertThrow(model.uses_homogeneous_mesh_data() || has_particle_state,
                      ExcMessage("Legacy transition control requires particle state."));
          const auto bulk_state=has_particle_state ? info.get_position_by_field_name("phase field fault state") : 0;
          const auto stress=info.get_position_by_field_name("maxwell stress");
          for (const auto &p:pm.get_particle_handler())
            {
              const auto data=p.get_properties();
              const double xd=(g.upper[1]-p.get_location()[1])/g.sine;
              const double expected=BP3::configured_initial_state(xd,law);
              if(has_particle_state)
                max_state_error=std::max(max_state_error,std::abs(data[bulk_state]-expected)/expected);
              AssertThrow(std::abs(data[bulk_fraction]-BP3::depth_fraction(p.get_location()[1]))<1.e-14,
                          ExcMessage("Bulk strengthening was not initialized from the ordinary BP3 transition."));
              for (unsigned int c=0; c<SymmetricTensor<2,dim>::n_independent_components; ++c)
                AssertThrow(data[stress+c]==0.,ExcMessage("Background prestress entered bulk perturbation history."));
            }
          AssertThrow(max_a_error<1.e-14 && max_f_error<1.e-14 && max_state_error<1.e-13,
                      ExcMessage("Ordinary BP3 transition/material initialization check failed."));
          std::ofstream summary(this->get_output_directory()+"transition-check.csv");
          summary<<std::setprecision(17)<<"max_a_error,max_f_error,max_theta_relative_error,particles,vertices\n"
                 <<max_a_error<<','<<max_f_error<<','<<max_state_error<<','<<pm.get_particle_handler().n_global_particles()<<','<<fault.n_vertices()<<'\n';
          return {"BP3 actual transition material/initialization","passed (material probes include 15--18 km; no coupled extension)"};
#else
          AssertThrow(false,ExcMessage("Transition check must use an ordinary BP3 plugin and observer without the bottom-test definition."));
          return {};
#endif
        }

        std::pair<std::string,std::string> write_initial_profile()
        {
          auto &model=const_cast<Model &>(Plugins::get_plugin_as_type<const Model>(this->get_material_model()));
          model.prepare_reconstructed_fault_mechanical_solve();
          const auto &manager=this->get_reconstructed_fault_manager();
          const auto &g=BP3::geometry();
          const auto &I=Access::current_normalization_integrals(model).at(0);
          const auto profiles=this->get_phase_field_handler().get_phase_field_profiles(g.peak_phase);
          const auto &contacts=manager.get_boundary_contacts();
          const auto &continuations=Access::normalization_continuations(model);
          AssertThrow(this->get_timestep_number()==0 && contacts.size()==2
                      && model.accepted_fe_strain_state().block(this->introspection().block_indices.velocities).l2_norm()==0.
                      && *std::min_element(I.begin(),I.end())>0.,
                      ExcMessage("Profile probe requires fresh zero FE history and admitted paired completion."));
          std::ofstream surface(local_file("initial-normalization.csv"));
          surface<<std::setprecision(17)<<"s_m,I_h_m\n";
          const auto &fault=manager.get_fault(0);
          for (unsigned int j=0; j<I.size(); ++j)
            surface<<BP3::down_dip(fault.vertex(j)[0],fault.vertex(j)[1])<<','<<I[j]<<'\n';
          std::ofstream out(local_file("initial-profile-cuts.csv"));
          out<<std::setprecision(17)<<"s_m,r_m,exterior,phi,physical_phi,I_h_m,h,chi_per_m\n";
          Tensor<1,dim> normal,tangent;
          for (unsigned int d=0; d<2; ++d)
            {
              normal[d]=g.normal[d];
              tangent[d]=g.tangent[d];
            }
          const std::vector<double> stations {0.,10.,20.,50.,g.length/4.,g.length/2.,3*g.length/4.,g.length-50.,g.length-20.,g.length-10.,g.length};
          const auto write=[&](const double s,const double r,const bool exterior,const double phi)
          {
            Point<dim> x;
            // I_h belongs to the surface foot, including for exterior
            // completion samples outside the physical bulk-source domain.
            for (unsigned int d=0; d<2; ++d) x[d]=g.upper[d]+s*tangent[d];
            const auto source=manager.project_to_bulk_source(x);
            AssertThrow(source.active,ExcMessage("Profile cut left the admitted bulk-source association."));
            const auto l=model.evaluate_reconstructed_fault_localization(0,source.segment_index,source.xi,phi,phi,"section 8 profile cut");
            out<<s<<','<<r<<','<<exterior<<','<<phi<<','<<profiles[0]->value(std::abs(r))
               <<','<<l.current_I_h<<','<<l.current_h<<','<<l.current_h/l.current_I_h<<'\n';
          };
          for (const double s:stations)
            {
              Point<dim> foot;
              for (unsigned int d=0; d<2; ++d) foot[d]=g.upper[d]+s*tangent[d];
              for (const auto &cell:this->get_dof_handler().active_cell_iterators())
                if (cell->is_locally_owned())
                  {
                    const auto bounds=cell->bounding_box().get_boundary_points();
                    double lo=-60.,hi=60.;
                    for (unsigned int d=0; d<dim; ++d)
                      {
                        double a=(bounds.first[d]-foot[d])/normal[d],b=(bounds.second[d]-foot[d])/normal[d];
                        if (a>b) std::swap(a,b);
                        lo=std::max(lo,a);
                        hi=std::min(hi,b);
                      }
                    if (hi<=lo) continue;
                    std::vector<Point<dim> > units;
                    std::vector<double> radii;
                    for (int i=std::ceil(2*lo); .5*i<=hi; ++i)
                      {
                        const double r=.5*i;
                        auto unit=this->get_mapping().transform_real_to_unit_cell(cell,foot+r*normal);
                        AssertThrow(GeometryInfo<dim>::distance_to_unit_cell(unit)<1.e-12,ExcMessage("Profile cut clipping failed."));
                        for (unsigned int d=0; d<dim; ++d) unit[d]=std::clamp(unit[d],0.,1.);
                        units.push_back(unit);
                        radii.push_back(r);
                      }
                    if (units.empty()) continue;
                    FEValues<dim> fe(this->get_mapping(),this->get_fe(),Quadrature<dim>(units),update_values);
                    fe.reinit(cell);
                    std::vector<double> phi(units.size());
                    fe[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)].get_function_values(this->get_solution(),phi);
                    for (unsigned int q=0; q<radii.size(); ++q) write(s,radii[q],false,phi[q]);
                  }
              // Evaluate the same prescribed exterior Q1 lattice used by
              // native completion, as in the retained independent reference.
              for (unsigned int c=0; c<contacts.size(); ++c)
                {
                  const auto &contact=contacts[c];
                  const double along=(foot-contact.position)*contact.inward_tangent;
                  if (along<0. || along>contact.influence_length) continue;
                  for (int i=-120; i<=120; ++i)
                    {
                      const double r=.5*i;
                      const auto x=foot+r*normal;
                      if ((x-contact.position)*contact.inward_boundary_normal>=0. || std::abs(r)>continuations[c].extent) continue;
                      const auto &data=continuations[c];
                      Point<dim> lower;
                      Tensor<1,dim> fraction;
                      for (unsigned int d=0; d<dim; ++d)
                        {
                          lower[d]=data.lattice_origin[d]+data.spacing[d]*std::floor((x[d]-data.lattice_origin[d])/data.spacing[d]);
                          fraction[d]=(x[d]-lower[d])/data.spacing[d];
                        }
                      double phi=0.;
                      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                        {
                          auto node=lower;
                          double shape=1.;
                          for (unsigned int d=0; d<dim; ++d)
                            if (v & (1u<<d))
                              {
                                node[d]+=data.spacing[d];
                                shape*=fraction[d];
                              }
                            else shape*=1.-fraction[d];
                          phi+=shape*data.profile->value(std::abs((node-contact.position)*contact.normal));
                        }
                      write(s,r,true,phi);
                    }
                }
            }
          std::ofstream summary(local_file("initial-profile-summary.csv"));
          summary<<"cells,dofs,particles,vertices,contacts,min_I_m,max_I_m,zero_FE_history\n"<<std::setprecision(17)
                 <<this->get_triangulation().n_global_active_cells()<<','<<this->get_dof_handler().n_dofs()<<','
                 <<this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler().n_global_particles()<<','
                 <<fault.n_vertices()<<','<<contacts.size()<<','<<*std::min_element(I.begin(),I.end())<<','<<*std::max_element(I.begin(),I.end())<<",1\n";
          AssertThrow(out && surface && summary,ExcMessage("Cannot write initialization profile check."));
          return {"BP3 initialization/profile cuts","passed; no mechanical trajectory"};
        }

        void capture_incoming(const bool temperature)
        {
          if (!temperature)
            return;
          const auto step = this->get_timestep_number();
          AssertThrow(incoming_step != step || lifecycle, ExcMessage("7B requires one incoming anchor per physical step."));
          auto &model = const_cast<Model &>(Plugins::get_plugin_as_type<const Model>(this->get_material_model()));
          // A restart rebuilds transient boundary associations after the
          // material's start-step update. Point diagnostics require the same
          // preparation that the native mechanical solve performs later.
          if(lifecycle)model.prepare_reconstructed_fault_mechanical_solve();
          const auto &manager = this->get_reconstructed_fault_manager();
          const auto &fault = manager.get_fault(0);
          const auto &pm = this->get_phase_field_handler().get_associated_particle_manager();
          const auto &info = pm.get_property_manager().get_data_info();
          const auto stress_position = info.get_position_by_field_name("maxwell stress");
          const auto H_position = info.get_position_by_field_name("crack_driving_force");
          const auto phase_block = this->introspection().variable("phase_field").block_index;
          if (lifecycle && incoming_step==step)
            {
              bool same=model.fe_strain_state->dg_history==incoming_dg
                && model.fe_strain_state->Z==incoming_Z
                && model.fe_strain_state->accepted_publications==incoming_publications
                && model.accepted_fe_strain_state()==incoming_fe;
              same=same && model.fe_strain_state->I==initial_I
                && model.fe_strain_state->committed_step+1==step
                && manager.get_timestep_committed_slip_rate(0)==incoming_rates
                && manager.get_slip_rate(0)==incoming_rates;
              for(unsigned int c=0;c<incoming_dg_snapshot->trial_stress.size();++c)
                for(const auto i:incoming_dg_snapshot->trial_stress[c].locally_owned_elements())
                  same=same && model.fe_strain_state->dg_history->trial_stress[c][i]==incoming_dg_snapshot->trial_stress[c][i];
              for(unsigned int j=0;j<fault.n_vertices();++j)
                {
                  const auto values=fault.get_properties(j);
                  // Registry-only provenance markers retain NaN sentinels.
                  // Bitwise equality checks unchanged storage, including NaNs.
                  same=same && values.size()==incoming_surface[j].size()
                    && std::memcmp(values.data(),incoming_surface[j].data(),values.size()*sizeof(double))==0;
                }
              for(const auto &particle:pm.get_particle_handler())
                {
                  const auto values=particle.get_properties();
                  const auto &saved=incoming_particles.at(particle.get_id());
                  same=same && values.size()==saved.size()
                    && std::memcmp(values.data(),saved.data(),values.size()*sizeof(double))==0;
                }
              AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(same),this->get_mpi_communicator())==1
                          && solver_publications==0 && this->get_timestep()<incoming_dt
                          && this->get_time()==incoming_time-incoming_dt+this->get_timestep(),
                          ExcMessage("Outer mature rejection mutated an accepted history or failed to cut dt."));
              std::ofstream out(local_file("outer-cutback.csv"));
              out<<std::setprecision(17)<<"step,rejected_time,rejected_dt,retry_time,retry_dt,publications,accepted_unchanged\n"
                 <<step<<','<<incoming_time<<','<<incoming_dt<<','<<this->get_time()<<','<<this->get_timestep()<<','<<incoming_publications<<",1\n";
            }
          incoming_step=step;solver_publications=0;
          incoming_dt=this->get_timestep();incoming_time=this->get_time();
          incoming_surface.clear();incoming_particles.clear();
          if(lifecycle)
            {
              incoming_publications=model.fe_strain_state->accepted_publications;
              incoming_rates=manager.get_timestep_committed_slip_rate(0);
              for(unsigned int j=0;j<fault.n_vertices();++j)
                {const auto values=fault.get_properties(j);incoming_surface.emplace_back(values.begin(),values.end());}
              for(const auto &particle:pm.get_particle_handler())
                {const auto values=particle.get_properties();incoming_particles[particle.get_id()]={values.begin(),values.end()};}
            }
          const auto rule = fault_mechanical_quadrature(*this);
          const QGauss<dim> expected(std::max(model.fe_bulk_normal_quadrature_points,this->get_fe().degree+1));
          AssertThrow(rule.get_points() == expected.get_points() && rule.get_weights() == expected.get_weights(),
                      ExcMessage("BP3 operators must use the selected shared tensor rule."));
          if (initial_phase.n_blocks()==0)
            {
              initial_phase.reinit(this->get_solution());
              initial_phase = this->get_solution();
              initial_I = Access::current_normalization_integrals(model).at(0);
              if(step==0)previous_slip.assign(fault.n_vertices(),0.);
              else if(previous_slip.empty())
                {
                  const auto property=manager.get_property_information()[manager.get_property_index("cumulative_signed_slip_m")].position;
                  for(unsigned int j=0;j<fault.n_vertices();++j)previous_slip.push_back(fault.get_properties(j)[property]);
                }
              std::ofstream mesh(local_file("initial-mesh.csv"));
              mesh << std::setprecision(17) << "cell,level,x,y,edge\n";
              for (const auto &cell : this->get_dof_handler().active_cell_iterators())
                if (cell->is_locally_owned())
                  mesh << cell->id() << ',' << cell->level() << ',' << cell->center()[0] << ','
                       << cell->center()[1] << ',' << cell->diameter()/std::sqrt(double(dim)) << '\n';
              std::ofstream phase(local_file("initial-phase.csv"));
              phase << std::setprecision(17) << "dof,phi\n";
              for (const auto i : initial_phase.block(phase_block).locally_owned_elements())
                phase << i << ',' << initial_phase.block(phase_block)[i] << '\n';
              std::ofstream quadrature(local_file("mechanical-quadrature.csv"));
              quadrature << std::setprecision(17) << "q,x,y,weight\n";
              for (unsigned int q=0; q<rule.size(); ++q)
                quadrature << q << ',' << rule.point(q)[0] << ',' << rule.point(q)[1] << ',' << rule.weight(q) << '\n';
              AssertThrow(mesh && phase && quadrature, ExcMessage("Cannot write 7B initial identity."));
            }
          else if (prescribed_schedule)
            AssertThrow(this->get_timestep() == this->get_parameters().maximum_time_step
                        && this->get_time() == step*this->get_timestep(),
                        ExcMessage("7B prescribed accepted schedule changed; no cutback continuation is allowed."));
          AssertThrow(this->get_solution().block(phase_block) == initial_phase.block(phase_block)
                      && Access::current_normalization_integrals(model).at(0) == initial_I,
                      ExcMessage("7B phase or normalization changed."));
          if (model.uses_independent_bulk_history())
            {
              incoming_fe.reinit(model.accepted_fe_strain_state());
              incoming_fe = model.accepted_fe_strain_state();
              incoming_Z = Access::fe_slip_memory(model);
              incoming_dg = model.fe_strain_state->dg_history;
              if(lifecycle)incoming_dg_snapshot=std::make_shared<typename Model::DGHistory>(*incoming_dg);
            }
          anchors.clear();
          if (moving_particles)
            {
              fixed_particles.clear();
              incoming_theta.clear();
              const auto position=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
              for (unsigned int j=0; j<fault.n_vertices(); ++j) incoming_theta.push_back(fault.get_properties(j)[position]);
            }
          const bool first_particle_capture=fixed_particles.empty();
          unsigned int index = 0;
          // Compact deterministic point controls; no domain-wide particle dump.
          for (const auto &p : pm.get_particle_handler())
            {
              if (first_particle_capture || step == 0 || moving_particles)
                fixed_particles.emplace(p.get_id(),std::make_pair(p.get_location(),p.get_properties()[H_position]));
              AssertThrow(fixed_particles.at(p.get_id()).first == p.get_location()
                          && fixed_particles.at(p.get_id()).second == p.get_properties()[H_position],
                          ExcMessage("7B stationary particle identity or inert H changed."));
              const bool active = manager.project_to_bulk_source(p.get_location()).active;
              if ((moving_particles && (!population_event_audit || step == 34 || step == 44)) || index % 257 == 0 || (active && index % 97 == 0))
                {
                  SymmetricTensor<2,dim> stress;
                  for (unsigned int c=0; c<SymmetricTensor<2,dim>::n_independent_components; ++c)
                    stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)] = p.get_properties()[stress_position+c];
                  anchors.push_back({p.get_id(),p.get_location(),p.get_surrounding_cell()->id(),stress});
                }
              ++index;
            }
          AssertThrow(fixed_particles.size() == pm.get_particle_handler().n_locally_owned_particles(),
                      ExcMessage("7B particle insertion/deletion is unsupported."));
          if (step == 0)
            {
              std::uint64_t hash = 14695981039346656037ull;
              const auto bytes = [&hash](const auto &value)
              {
                const auto *p = reinterpret_cast<const unsigned char *>(&value);
                for (std::size_t i=0; i<sizeof(value); ++i)
                  hash = (hash^p[i])*1099511628211ull;
              };
              for (const auto &p : pm.get_particle_handler())
                {
                  const auto id = p.get_id();
                  bytes(id);
                  for (unsigned int d=0; d<dim; ++d) bytes(p.get_location()[d]);
                  // Only physical fields: integrator/property offsets are not identity.
                  for (const std::string field :
                  {"crack_driving_force","maxwell stress","initial strengthening","phase field fault state"
                  })
                    {
                      // Static diagnostic composition has its native initial-*
                      // name; hash the same physical Theta field in both modes.
                      const std::string name=field=="phase field fault state" && !info.fieldname_exists(field)
                                               && info.fieldname_exists("initial theta_initial")
                                             ? "initial theta_initial" : field;
                      if (info.fieldname_exists(name))
                        {
                          const auto pos = info.get_position_by_field_name(name);
                          for (unsigned int c=0; c<info.get_components_by_field_name(name); ++c) bytes(p.get_properties()[pos+c]);
                        }
                    }
                }
              std::ofstream out(local_file("initial-particle-identity.csv"));
              out << "particles,FNV1a_physical_fields,sampled_particles\n" << fixed_particles.size() << ',' << hash << ',' << anchors.size() << '\n';
            }
          this->get_pcout() << "   BP3 7B incoming anchor: step=" << step << ", quadrature=" << rule.size()
                            << ", sampled particles=" << anchors.size() << ", fixed population=" << fixed_particles.size() << std::endl;
        }

        void verify_stationary_operators()
        {
          auto &model=const_cast<Model &>(Plugins::get_plugin_as_type<const Model>(this->get_material_model()));
          if(!model.uses_stationary_mature_dg_optimizations())return;
          const auto original=model.dg_history_values();
          const auto publications=model.fe_strain_state->accepted_publications;
          const auto Z=model.fe_strain_state->Z;
          auto &manager=this->get_reconstructed_fault_manager();
          auto &surface=this->get_reconstructed_fault_surface_system();
          auto &B=this->get_reconstructed_fault_stokes_coupling();
          std::vector<std::vector<double>> rates;
          for(unsigned int f=0;f<manager.get_faults().size();++f)rates.push_back(manager.get_timestep_committed_slip_rate(f));
          // Freeze a common (accepted history, physical trial, dt) tuple. These
          // noncommitting kernels are not an additional physical time step.
          StokesSolver::ReconstructedFaultCondensedSystem<dim> system(this->get_simulator());
          auto free=system.linearize(this->get_system_matrix(),this->get_solution(),rates);
          BV direction(this->introspection().index_sets.stokes_partitioning,this->get_mpi_communicator());
          for(unsigned int b=0;b<2;++b)
            for(const auto i:direction.block(b).locally_owned_elements())
              direction.block(b)[i]=(b==0 ? 1e-9 : 1000./this->get_pressure_scaling())*std::sin(.017*(i+3));
          direction.compress(VectorOperation::insert);
          BV action(direction);free.vmult(action,direction); // Audited against sampled B/G and actual free K_V.
          for(unsigned int sample=0;sample<4;++sample)
            {
              auto probe=rates;
              for(auto &fault:probe)for(unsigned int j=0;j<fault.size();++j)
                fault[j]=sample<3 ? (j==sample*(fault.size()-1)/2 ? 1. : 0.) : std::sin(.19*(j+1));
              BV load(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
              B.apply_B(probe,load); // The existing diagnostic switch verifies the reference action.
              BV owned(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
              owned.block(0)=direction.block(0);owned.block(1)=direction.block(1);
              owned.block(1)*=this->get_pressure_scaling();
              AffineConstraints<double> constraints(this->get_current_constraints());
              for(const auto &line:constraints.get_lines())constraints.set_inhomogeneity(line.index,0.);
              constraints.distribute(owned);owned.compress(VectorOperation::insert);
              BV physical(this->introspection().index_sets.system_partitioning,this->introspection().index_sets.system_relevant_partitioning,this->get_mpi_communicator());physical=owned;
              surface.apply_G(physical,probe);
            }
          auto mask=manager.prescribed_slip_rate_mask();unsigned int restricted_rows=0;
          for(auto &fault:mask)
            for(const unsigned int j:{0u,static_cast<unsigned int>(fault.size()/2),static_cast<unsigned int>(fault.size()-1)})
              {fault[j]=true;++restricted_rows;}
          auto solve=surface.create_restricted_linear_solve(mask);
          auto restricted=free.with_surface_solve(*solve);
          restricted.vmult(action,direction); // Audits the actual new scratch path with nonempty active constraints.

          // All nine incoming modes/components survive a source update. Compare
          // mass/inverse and roundoff caching against the unchanged reference
          // arithmetic on the SAME manufactured native Q2 velocity and history.
          auto incoming=*model.fe_strain_state->dg_history;
          for(unsigned int c=0;c<incoming.trial_stress.size();++c)
            {
              LinearAlgebra::Vector owned(incoming.space->dofs.locally_owned_dofs(),this->get_mpi_communicator());
              for(const auto i:owned.locally_owned_elements())owned[i]=100.*std::sin(.13*(i+1+17*c));
              owned.compress(VectorOperation::insert);incoming.trial_stress[c]=owned;
            }
          BV bulk(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());bulk=this->get_solution();
          for(const auto i:bulk.block(0).locally_owned_elements())bulk.block(0)[i]=1e-9*(std::cos(.07*i)+.2*std::sin(.3*i));
          bulk.compress(VectorOperation::insert);
          BV ghost(this->introspection().index_sets.system_partitioning,this->introspection().index_sets.system_relevant_partitioning,this->get_mpi_communicator());ghost=bulk;
          const auto coefficients=Model::compute_maxwell_coefficients(model.reference_viscosities[0],model.elastic_shear_moduli[0],
            this->get_timestep_number() ? this->get_timestep() : model.initial_time_step);
          const auto reference=model.make_dg_history_trial(incoming,ghost,coefficients,true,true);
          const auto optimized=model.make_dg_history_trial(incoming,ghost,coefficients,true,false);
          double source_error=0.,bound_error=0.;
          for(unsigned int c=0;c<incoming.trial_stress.size();++c)
            {
              LinearAlgebra::Vector difference(incoming.space->dofs.locally_owned_dofs(),this->get_mpi_communicator());
              for(const auto i:difference.locally_owned_elements())difference[i]=reference->trial_stress[c][i]-optimized->trial_stress[c][i];
              difference.compress(VectorOperation::insert);source_error=std::max(source_error,difference.linfty_norm());
            }
          for(const auto &entry:reference->source_roundoff)
            {
              const auto &other=optimized->source_roundoff.at(entry.first);
              bound_error=std::max({bound_error,(entry.second.projection-other.projection).norm(),(entry.second.gradient_scale-other.gradient_scale).norm()});
            }
          bound_error=Utilities::MPI::max(bound_error,this->get_mpi_communicator());
          AssertThrow(source_error==0. && bound_error==0.,ExcMessage("Stationary source cache changed coefficients or operand-derived allowance."));

          double stress_error=0.,stress_scale=0.;std::array<unsigned int,3> checked{};
          const auto quadrature=fault_mechanical_quadrature(*this);
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),quadrature,update_quadrature_points|update_values);
          for(const auto &entry:model.fe_strain_state->cells)
            {
              fe.reinit(entry.second);std::vector<double> phase(quadrature.size());
              fe[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)].get_function_values(model.fe_strain_state->displacement_history,phase);
              if(*std::max_element(phase.begin(),phase.end())==0.)continue;
              const auto center=entry.second->center();const double s=BP3::down_dip(center[0],center[1])/BP3::geometry().length;
              const unsigned int bin=s<.2 ? 0 : (s>.8 ? 2 : 1);
              if((bin==1 && (s<.4 || s>.6)) || checked[bin]>=4)continue;
              ++checked[bin];
              const auto &fast=model.retained_dg_qp_stress(entry.first,quadrature,fe.get_quadrature_points());
              for(unsigned int q=0;q<quadrature.size();++q)
                {
                  double scale=0.;const auto old=model.evaluate_dg_history_stress(*model.fe_strain_state->dg_history,Z,model.fe_strain_state->I,entry.first,fe.quadrature_point(q),&scale,&model.fe_strain_state->displacement_history);
                  const auto d=old-fast[q];stress_error=std::max(stress_error,std::sqrt((d*d)+trace(d)*trace(d)));stress_scale=std::max(stress_scale,scale);
                }
            }
          stress_error=Utilities::MPI::max(stress_error,this->get_mpi_communicator());stress_scale=Utilities::MPI::max(stress_scale,this->get_mpi_communicator());
          AssertThrow(stress_error<=2e-11*std::max(stress_scale,1e-30),ExcMessage("Batched mechanical history changed native point evaluation."));
          const auto n_stokes=direction.block(0).size()+direction.block(1).size();
          unsigned int hanging=0;
          for(const auto &line:this->get_current_constraints().get_lines())hanging+=line.index<n_stokes && !line.entries.empty();
          hanging=Utilities::MPI::sum(hanging,this->get_mpi_communicator());
          double mu_min=std::numeric_limits<double>::infinity(),mu_max=-mu_min;
          for(const auto &sample:surface.get_normal_traction_diagnostic().samples)
            {mu_min=std::min(mu_min,sample.friction_coefficient);mu_max=std::max(mu_max,sample.friction_coefficient);}
          mu_min=Utilities::MPI::min(mu_min,this->get_mpi_communicator());mu_max=Utilities::MPI::max(mu_max,this->get_mpi_communicator());
          AssertThrow(hanging>0 && restricted_rows>0 && this->get_pressure_scaling()!=1. && mu_max>mu_min,
                      ExcMessage("Frozen operator fixture lacks required constraints, scaling or varying friction."));
          AssertThrow(model.dg_history_values()==original && model.fe_strain_state->accepted_publications==publications && model.fe_strain_state->Z==Z,
                      ExcMessage("Stationary frozen operator checks published history."));
          BP3::collective_root_write(this->get_mpi_communicator(),[&]()
            {
              const auto path=this->get_output_directory()+"stationary-operator-checks.csv";const bool header=BP3::needs_header(path);
              std::ofstream out(path,std::ios::app);out.exceptions(std::ios::failbit|std::ios::badbit);
              if(header)out<<"step,source_coefficient_error,source_allowance_error,QP_stress_error_Pa,QP_operand_scale_Pa,hanging_relations,restricted_rows,pressure_scaling,mu_min,mu_max\n";
              out<<std::setprecision(17)<<this->get_timestep_number()<<','<<source_error<<','<<bound_error<<','<<stress_error<<','<<stress_scale<<','<<hanging<<','<<restricted_rows<<','<<this->get_pressure_scaling()<<','<<mu_min<<','<<mu_max<<'\n';
            });
        }

      public:
        static void declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("BP3 coupled checks");
          prm.declare_entry("Population event audit", "false", Patterns::Bool(), "Observe fixed-field native population and near-face events during one bounded baseline replay.");
          prm.declare_entry("Moving particle check", "false", Patterns::Bool(), "Audit native movement, population management and publication in the bounded strengthening control.");
          prm.declare_entry("Transition material check", "false", Patterns::Bool(), "Initialization-only ordinary BP3 material/assignment probes across its complete 15--18 km transition.");
          prm.declare_entry("Profile initialization check", "false", Patterns::Bool(), "Serial initialization-only Q1 phase/localization cuts, including the native exterior completion lattice.");
          prm.declare_entry("Prescribed schedule check", "true", Patterns::Bool(), "Require the fixed local comparison schedule. Disable only for the ordinary BP3 controller startup check.");
          prm.declare_entry("Common stress points file", "", Patterns::Anything(), "Optional serial fixed-mesh physical quadrature points with explicit native cell-trace ownership; diagnostic only.");
          prm.declare_entry("Write authoritative stress", "false", Patterns::Bool(), "Export accepted full tensor stress on common bulk Gauss3 points and independent history coefficients at selected bounded steps.");
          prm.declare_entry("Write FE coefficients", "false", Patterns::Bool(), "Compact physical-coordinate FE fields for the selected MPI comparison.");
          prm.declare_entry("Scheduled field output", "false", Patterns::Bool(), "Use BP3 heavy-output time/slip intervals for dense authoritative tensors, coefficients and normal cuts. Force initial/event milestones and graceful stop; retain per-step checks, profiles and root weak loads. False preserves all bounded-comparison exports.");
          prm.declare_entry("Inject nonlinear failure", "false", Patterns::Bool(), "Throw native nonlinear nonconvergence once inside step-two mechanics, for the actual outer cutback control.");
          prm.declare_entry("Mature lifecycle checks", "false", Patterns::Bool(), "Permit same-mode mature DG restart/retry and verify accepted-state invariance across an actual outer cutback.");
          prm.declare_entry("Inject candidate failure", "false", Patterns::Bool(), "Arm the collective fixed-sample registry guard on the last rank at step two, verify rejection after native history candidates and retry unchanged dt.");
          prm.leave_subsection();
          prm.leave_subsection();
        }
        void parse_parameters(ParameterHandler &prm) override
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("BP3 coupled checks");
          population_event_audit = prm.get_bool("Population event audit");
          moving_particles = prm.get_bool("Moving particle check");
          material_only = prm.get_bool("Transition material check");
          profile_only = prm.get_bool("Profile initialization check");
          prescribed_schedule = prm.get_bool("Prescribed schedule check");
          write_coefficients = prm.get_bool("Write FE coefficients");
          write_authoritative_stress = prm.get_bool("Write authoritative stress");
          scheduled_fields = prm.get_bool("Scheduled field output");
          common_points_file = prm.get("Common stress points file");
          inject_failure = prm.get_bool("Inject candidate failure");
          lifecycle = prm.get_bool("Mature lifecycle checks");
          nonlinear_failure = prm.get_bool("Inject nonlinear failure");
          AssertThrow(!nonlinear_failure || lifecycle,ExcMessage("Nonlinear failure injection requires mature lifecycle checks."));
          AssertThrow(!scheduled_fields || lifecycle,ExcMessage("Scheduled first-event fields require the restart-aware mature lifecycle observer."));
          prm.leave_subsection();
          prm.leave_subsection();
          if (scheduled_fields)
            {
              prm.enter_subsection("Postprocess");
              prm.enter_subsection("BP3");
              field_schedule.slip_interval = prm.get_double("Heavy output slip interval");
              field_schedule.time_interval = prm.get_double("Heavy output time interval");
              prm.leave_subsection();
              prm.leave_subsection();
            }
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("BP3 restored monitor");
          audit_filter_mode = prm.get("Friction normal input");
          audit_filter_length = prm.get_double("Normal filter length");
          prm.leave_subsection();
          prm.leave_subsection();
          prm.enter_subsection("Particles");
          minimum_particles=prm.get_integer("Minimum particles per cell");
          maximum_particles=prm.get_integer("Maximum particles per cell");
          prm.leave_subsection();
        }
        std::list<std::string> required_other_postprocessors() const override
        {
          if (material_only || profile_only) return {};
          std::list<std::string> required{"reconstructed fault BP3", "BP3 restored monitor"};
          if (scheduled_fields) required.push_back("BP3 output complete");
          return required;
        }

        void initialize() override
        {
          const auto &parameters = this->get_parameters();
          if (!common_points_file.empty())
            {
              AssertThrow(Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1,
                          ExcMessage("Common stress export requires serial ownership."));
              std::ifstream input(common_points_file);
              std::string line;
              std::getline(input,line);
              AssertThrow(input && line=="point,cell,x,y,JxW",ExcMessage("Invalid common stress point header."));
              while(std::getline(input,line))
                {
                  for(auto &c:line) if(c==',') c=' ';
                  std::istringstream row(line);
                  unsigned int id;std::string cell;Point<dim> x;double weight;
                  row>>id>>cell>>x[0]>>x[1]>>weight;
                  AssertThrow(row && weight>0. && std::isfinite(weight),ExcMessage("Malformed common stress point."));
                  auto &points=common_points[cell];
                  points.physical.push_back(x);points.weights.push_back(weight);points.ids.push_back(id);
                }
              AssertThrow(!common_points.empty(),ExcMessage("Empty common stress points."));
            }
          if (material_only || profile_only)
            {
              initial_composition=this->get_initial_composition_manager_pointer();
              AssertThrow(bool(initial_composition),ExcMessage("Transition observer must retain the initial-composition manager before its native release."));
              AssertThrow(Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1
                          && parameters.nonlinear_solver==Parameters<dim>::NonlinearSolver::single_Advection_no_Stokes,
                          ExcMessage("Transition check is a serial initialization-only control."));
              return;
            }
          AssertThrow(dim == 2
                      && (!parameters.resume_computation || lifecycle) && !parameters.mesh_deformation_enabled
                      && parameters.adaptive_refinement_interval == 0 && parameters.additional_refinement_times.empty()
                      && parameters.nonlinear_solver == Parameters<dim>::NonlinearSolver::single_Advection_iterated_Newton_Stokes,
                      ExcMessage("7B requires fresh fixed stationary coupled Newton mechanics."));
          this->get_signals().post_simulator_initialization.connect([this](const SimulatorAccess<dim> &)
          {
            // All weak rows are captured. A small spatial window retains QP
            // recurrence controls without dense QP or particle CSV output.
            this->get_reconstructed_fault_surface_system().set_normal_traction_diagnostic([](const Point<dim> &p)
            {
              const double s = BP3::down_dip(p[0],p[1]);
              const double spacing = BP3::geometry().length/8.;
              return std::abs(s-spacing*std::round(s/spacing)) < .25;
            });
          });
          this->get_signals().post_advection_solver.connect([this](const SimulatorAccess<dim> &,bool temperature,unsigned int,const SolverControl &)
          {
            if (moving_particles) { if (temperature) moving_attempt=true; }
            else capture_incoming(temperature);
            if (temperature && inject_failure && !failure_done && this->get_timestep_number()==2)
              {
                failure_done=true;
                verify_candidate_rejection();
              }
          });
          if (moving_particles)
            {
              this->get_signals().post_simulator_initialization.connect([this](const SimulatorAccess<dim> &)
              {
                auto &pm=const_cast<Particle::Manager<dim> &>(this->get_phase_field_handler().get_associated_particle_manager());
                if (population_event_audit)
                  {
                    AssertThrow(Utilities::MPI::n_mpi_processes(this->get_mpi_communicator()) == 1, ExcMessage("The event observer is serial; independent bounded cases can run in parallel."));
                    const auto *native = dynamic_cast<const Particle::Integrator::RK2<dim> *>(pm.integrator.get());
                    AssertThrow(native != nullptr, ExcMessage("Event observation requires the native RK2 integrator."));
                    pm.integrator = std::make_unique<PopulationAuditRK2>(*native, [this]()
                    {
                      if (this->get_timestep_number() == 44) audit_fixed_fields("population-44-before", true);
                    });
                  }

                this->get_signals().post_particle_backup.connect([this,&pm](Particle::Manager<dim> &manager)
                {
                  if (&manager==&pm) backup_moving_particles();
                });
                pm.get_particle_handler().signals.particle_lost.connect([this](const auto &,const auto &)
                {
                  ++managed_lost;
                });
                pm.post_particle_creation.connect([this,&pm](const auto &p)
                {
                  const auto &probe=dynamic_cast<const Particle::Interpolator::ManagedHistoryProbe<dim> &>(pm.get_interpolator());
                  const auto stress=pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
                  const auto H=pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
                  AssertThrow(probe.last_point==p->get_location(),ExcMessage("Birth did not use the captured native interpolation."));
                  AssertThrow(std::isfinite(probe.last_values.at(H)) && p->get_properties()[H]==probe.last_values[H],
                              ExcMessage("Moving birth H differs from its transported-history fit."));
                  for (unsigned int c=0; c<3; ++c)
                    {
                      AssertThrow(std::isfinite(probe.last_values.at(stress+c)) && p->get_properties()[stress+c]==probe.last_values[stress+c],ExcMessage("Birth stress differs from its old-history fit."));
                      maximum_birth_stress=std::max(maximum_birth_stress,std::abs(probe.last_values[stress+c]));
                    }
                  managed_births.insert(p->get_id());
                });
                this->get_signals().post_particle_management.connect([this,&pm](Particle::Manager<dim> &manager)
                {
                  if (&manager == &pm && moving_attempt)
                    {
                      if (population_event_audit && this->get_timestep_number() == 44) audit_fixed_fields("population-44-after", true);
                      verify_moving_management();
                    }
                });
                this->get_signals().post_reconstructed_fault_linear_solver.connect(
                  [this](const auto &,const auto &,const auto &,const auto &,const auto &,const auto &,double,unsigned int,double)
                {
                  verify_moving_tangent();
                });
              });
            }
          this->get_signals().post_reconstructed_fault_linear_solver.connect(
            [this](const auto &,const auto &,const auto &,const auto &,const auto &,const auto &,double,unsigned int,double)
            {
              if(nonlinear_failure && !failure_done && this->get_timestep_number()==2)
                {failure_done=true;throw ExcNonlinearSolverNoConvergence();}
            });
          this->get_signals().post_reconstructed_fault_solver.connect([this](unsigned int,unsigned int,double,const std::vector<std::vector<bool> > &)
          {
            ++solver_publications;
          });
        }

        void save(std::map<std::string,std::string> &status) const override
        {
          if(!lifecycle)return;
          std::ostringstream out;aspect::oarchive archive(out);
          archive &last_output &previous_slip &failure_done;
          status["BP3 mature diagnostic progress"]=out.str();
          if (scheduled_fields)
            {
              std::ostringstream schedule;aspect::oarchive saved(schedule);
              saved &field_schedule &field_event;
              status["BP3 scheduled diagnostic output"]=schedule.str();
            }
        }
        void load(const std::map<std::string,std::string> &status) override
        {
          if(!lifecycle)return;
          std::istringstream in(status.at("BP3 mature diagnostic progress"));aspect::iarchive archive(in);
          archive &last_output &previous_slip &failure_done;
          if (scheduled_fields)
            {
              const auto entry=status.find("BP3 scheduled diagnostic output");
              AssertThrow(entry!=status.end(),ExcMessage("Checkpoint lacks the selected BP3 diagnostic output schedule."));
              const double slip=field_schedule.slip_interval,time=field_schedule.time_interval;
              std::istringstream schedule(entry->second);aspect::iarchive saved(schedule);
              saved &field_schedule &field_event;
              field_schedule.slip_interval=slip;field_schedule.time_interval=time;
            }
        }
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          if (profile_only)
            return write_initial_profile();
          if (material_only)
            return verify_transition();
          const auto &model = Plugins::get_plugin_as_type<const Model>(this->get_material_model());
          const auto &manager = this->get_reconstructed_fault_manager();
          const auto &fault = manager.get_fault(0);
          const auto &surface = this->get_reconstructed_fault_surface_system();
          const auto &weak = surface.get_linearization_residual();
          const auto &diagnostic = surface.get_normal_traction_diagnostic();
          const auto step = this->get_timestep_number();
          AssertThrow(incoming_step == step && solver_publications == 1 && diagnostic.step == step
                      && (last_output == numbers::invalid_unsigned_int || step == last_output+1),
                      ExcMessage("7B requires one accepted mechanical solve and one output per physical step."));
          last_output = step;
          if(lifecycle)
            AssertThrow(model.fe_strain_state->committed_step==step && model.fe_strain_state->accepted_publications==step,
                        ExcMessage("Mature DG did not publish exactly once per accepted positive step."));
          const auto &intro = this->introspection();
          const auto phase_block = intro.variable("phase_field").block_index;
          AssertThrow(this->get_solution().block(phase_block) == initial_phase.block(phase_block)
                      && Access::current_normalization_integrals(model).at(0) == initial_I,
                      ExcMessage("7B accepted profile changed."));
          const auto project = [&weak](const std::vector<double> &load)
          {
            return ReconstructedFaultUtilities::solve_tridiagonal_system(weak.mass_diagonal[0],weak.mass_off_diagonal[0],load);
          };
          const auto raw = project(weak.raw_normal_traction.empty() ? weak.normal_traction.at(0) : weak.raw_normal_traction.at(0));
          const auto filtered = project(weak.normal_traction.at(0));
          const auto shear = project(weak.shear_traction.at(0));
          const auto pressure = project(diagnostic.pressure_load.at(0));
          const auto deviatoric = project(diagnostic.deviatoric_load.at(0));
          const auto background = project(diagnostic.background_load.at(0));
          const auto property = [&manager](const std::string &name)
          {
            return manager.get_property_information()[manager.get_property_index(name)].position;
          };
          const auto theta_position = property("phase field fault state");
          const auto slip_position = property("cumulative_signed_slip_m");
          const auto cohesive_position = property("phase field fault cohesive traction");
          const auto &V = manager.get_timestep_committed_slip_rate(0);
          bool emit_fields=true;
          std::vector<double> accepted_slip;
          if (scheduled_fields)
            {
              for(unsigned int j=0;j<V.size();++j) accepted_slip.push_back(fault.get_properties(j)[slip_position]);
              const auto maximum=std::max_element(V.begin(),V.end());
              const bool started=field_event.started,complete=field_event.complete;
              const auto below=field_event.below;
              field_event.observe(this->get_time(),*maximum,
                                  BP3::down_dip(fault.vertex(maximum-V.begin())[0],fault.vertex(maximum-V.begin())[1]));
              const bool milestone=(!started && field_event.started) || (!complete && field_event.complete)
                                   || (below==0 && field_event.below==1);
              emit_fields=field_schedule.due(this->get_time(),accepted_slip,milestone || BP3Benchmark::long_run_stop);
              // The native output dependency has already reduced the complete
              // stop request. Keep every rank in the same export/index branch.
              emit_fields=Utilities::MPI::max(static_cast<unsigned int>(emit_fields),this->get_mpi_communicator());
            }
          const double dt = step ? this->get_timestep() : model.initial_time_step;
          const auto coefficients = Model::compute_maxwell_coefficients(model.reference_viscosities[0],model.elastic_shear_moduli[0],dt);
          const double Dc = model.characteristic_fault_slip_distance();
          double Z_error = 0.,Z_scale = 1.,Z_max = 0.,theta_min = std::numeric_limits<double>::infinity(),normal_split_error = 0.;
          std::ofstream profile((Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==0
                                 ? this->get_output_directory()+"coupled-profile-"+std::to_string(step)+".csv" : "/dev/null"));
          profile << std::setprecision(17) << "node,s_m,time_s,V_m_s,theta_s,Omega,slip_m,shear_Pa,raw_normal_Pa,filtered_normal_Pa,pressure_Pa,deviatoric_Pa,background_Pa,I_h_m,Z_Pa_m\n";
          for (unsigned int j=0; j<V.size(); ++j)
            {
              const auto values = fault.get_properties(j);
              const double theta = values[theta_position],slip = values[slip_position];
              const double expected_slip = previous_slip[j]+(step ? this->get_timestep()*V[j] : 0.);
              AssertThrow(std::isfinite(theta) && theta > 0. && initial_I[j] > 0. && values[cohesive_position] == 0.
                          && std::abs(slip-expected_slip) <= 64*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(expected_slip)),
                          ExcMessage("7B positivity, mature traction or once-only slip publication failed."));
              if (moving_particles)
                {
                  const double expected_theta=step ? BP3::aging_state_reference(V[j],incoming_theta[j],this->get_timestep(),Dc) : incoming_theta[j];
                  AssertThrow(std::abs(theta-expected_theta)<256*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(expected_theta)),ExcMessage("Moving RSF state was not updated exactly once."));
                }
              previous_slip[j] = slip;
              theta_min = std::min(theta_min,theta);
              normal_split_error = std::max(normal_split_error,std::abs(raw[j]-(pressure[j]+deviatoric[j]+background[j])));
              double Z = 0.;
              if (model.uses_independent_bulk_history())
                {
                  const auto &s = *model.fe_strain_state;
                  AssertThrow(s.committed_step == step, ExcMessage("7B FE history was not committed at this physical step."));
                  Z = s.Z[j];
                  const double expected_Z = step ? coefficients.beta*incoming_Z[j]+coefficients.eta_ve*V[j] : 0.;
                  Z_error = std::max(Z_error,std::abs(Z-expected_Z));
                  Z_scale = std::max(Z_scale,std::abs(expected_Z));
                  Z_max = std::max(Z_max,Z);
                }
              profile << j << ',' << BP3::down_dip(fault.vertex(j)[0],fault.vertex(j)[1]) << ',' << this->get_time()
                      << ',' << V[j] << ',' << theta << ',' << V[j]*theta/Dc << ',' << slip << ',' << shear[j]
                      << ',' << raw[j] << ',' << filtered[j] << ',' << pressure[j] << ',' << deviatoric[j]
                      << ',' << background[j] << ',' << initial_I[j] << ',' << Z << '\n';
            }
          AssertThrow(Z_error <= 64*std::numeric_limits<double>::epsilon()*Z_scale
                      && (!model.uses_independent_bulk_history() || step == 0 || Z_max > 0.)
                      && normal_split_error <= 1.e-6+1.e-11*BP3::sigma0,
                      ExcMessage("7B FE memory or normal-traction decomposition failed."));

          if (write_authoritative_stress)
            {
              std::ofstream loads(!scheduled_fields || Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==0
                                  ? local_file("surface-weak-"+std::to_string(step)+".csv") : "/dev/null");
              loads<<std::setprecision(17)<<"node,s_m,mass_diagonal,mass_edge,shear,raw_normal,filtered_normal,friction,residual,pressure,deviatoric,background\n";
              for(unsigned int j=0;j<V.size();++j)
                loads<<j<<','<<BP3::down_dip(fault.vertex(j)[0],fault.vertex(j)[1])
                     <<','<<weak.mass_diagonal[0][j]<<','<<(j+1<V.size()?weak.mass_off_diagonal[0][j]:0.)
                     <<','<<weak.shear_traction[0][j]<<','<<weak.raw_normal_traction[0][j]
                     <<','<<weak.normal_traction[0][j]<<','<<weak.friction_traction[0][j]<<','<<weak.values[0][j]
                     <<','<<diagnostic.pressure_load[0][j]<<','<<diagnostic.deviatoric_load[0][j]
                     <<','<<diagnostic.background_load[0][j]<<'\n';
              AssertThrow(bool(loads),ExcMessage("Cannot export actual surface weak loads."));
            }

          double stress_error = 0.,stress_fraction = 0.,stress_scale = 0.,maximum_divergence = 0.,trace_increment = 0.;
          // Bound FE differentiation before cancellation, as in the production
          // FE publication check. The small net strain is not its operand scale.
          const auto gradient_operand = [this,&intro](const CellId &id,const Point<dim> &unit)
          {
            const auto cell = this->get_triangulation().create_cell_iterator(id);
            const typename DoFHandler<dim>::active_cell_iterator dof_cell(&this->get_triangulation(),cell->level(),cell->index(),&this->get_dof_handler());
            FEValues<dim> fe(this->get_mapping(),this->get_fe(),Quadrature<dim>({unit}),update_gradients);
            aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::native_fevalues);
            fe.reinit(dof_cell);
            Vector<double> values(this->get_fe().dofs_per_cell);
            dof_cell->get_dof_values(this->get_solution(),values);
            double sum = 0.;
            for (unsigned int i=0; i<values.size(); ++i)
              if (this->get_fe().system_to_component_index(i).first < dim)
                sum += std::abs(values[i])*fe[intro.extractors.velocities].symmetric_gradient(i,0).norm();
            return sum;
          };
          std::vector<double> batched_operand;
          if(model.uses_stationary_mature_dg_optimizations())
            {
              batched_operand.resize(diagnostic.samples.size());
              std::map<CellId,std::vector<unsigned int>> groups;
              for(unsigned int q=0;q<diagnostic.samples.size();++q)groups[CellId(diagnostic.samples[q].cell)].push_back(q);
              for(const auto &group:groups)
                {
                  std::vector<Point<dim>> units,physical;
                  for(const auto i:group.second){units.push_back(diagnostic.samples[i].reference_position);physical.push_back(diagnostic.samples[i].position);}
                  model.prepare_dg_point_samples(group.first,physical);
                  const auto cell=model.fe_strain_state->evaluation_cells.at(group.first);
                  FEValues<dim> fe(this->get_mapping(),this->get_fe(),Quadrature<dim>(units),update_gradients);
                  aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::native_fevalues);
                  fe.reinit(cell);Vector<double> values(this->get_fe().dofs_per_cell);cell->get_dof_values(this->get_solution(),values);
                  for(unsigned int q=0;q<units.size();++q)
                    for(unsigned int i=0;i<values.size();++i)
                      if(this->get_fe().system_to_component_index(i).first<dim)
                        batched_operand[group.second[q]]+=std::abs(values[i])*fe[intro.extractors.velocities].symmetric_gradient(i,q).norm();
                }
            }
          // Incoming QP snapshots use exactly the mechanical measure/time level.
          // At step zero their trial stress is kinematic; published history is zero.
          unsigned int diagnostic_index=0;
          for (const auto &q : diagnostic.samples)
            {
              if (weak.raw_normal_traction.empty())
                AssertThrow(q.friction_normal==q.total,
                            ExcMessage("Raw-normal control is not using the unfiltered quadrature-point input."));
              const auto strain = symmetrize(q.velocity_gradient);
              const auto effective = model.uses_deviatoric_strain_rate() ? Utilities::Tensors::consistent_deviator(strain) : strain;
              Tensor<1,dim> tangent;
              tangent[0] = q.normal[1];
              tangent[1] = -q.normal[0];
              const auto S = manager.get_shear_sense(0)*symmetrize(outer_product(tangent,q.normal));
              const double rate = manager.interpolate_slip_rate(0,q.segment,q.xi);
              const auto expected = coefficients.beta*q.incoming_stress+2*coefficients.eta_ve*(effective-q.chi*rate*S);
              SymmetricTensor<2,dim> actual = q.stress;
              double scale = coefficients.beta*q.incoming_stress.norm()+2*coefficients.eta_ve*(strain.norm()+std::abs(q.chi*rate)*S.norm());
              scale = std::max(scale,coefficients.beta*q.incoming_stress.norm()
                               +2*coefficients.eta_ve*((batched_operand.empty() ? gradient_operand(CellId(q.cell),q.reference_position)
                                                      : batched_operand[diagnostic_index])+std::abs(q.chi*rate)*S.norm()));
              if (step && model.uses_independent_bulk_history())
                {
                  const auto &s = *model.fe_strain_state;
                  double new_scale = 0.,old_scale = 0.;
                  if (model.uses_dg_trial_stress_history())
                    {
                      actual=model.evaluate_dg_history_stress(*s.dg_history,s.Z,s.I,CellId(q.cell),q.position,&new_scale);
                      model.evaluate_dg_history_stress(*incoming_dg,incoming_Z,initial_I,CellId(q.cell),q.position,&old_scale);
                    }
                  else
                    {
                      actual = model.evaluate_fe_strain_stress(s.displacement_history,s.Z,s.I,CellId(q.cell),q.position,&new_scale);
                      model.evaluate_fe_strain_stress(incoming_fe,incoming_Z,initial_I,CellId(q.cell),q.position,&old_scale);
                    }
                  scale = std::max(scale,2*(new_scale+coefficients.beta*old_scale));
                }
              ++diagnostic_index;
              const double error = (actual-expected).norm();
              const double bound = 128*std::numeric_limits<double>::epsilon()*std::max(1.,scale)
                                   +(step && model.uses_dg_trial_stress_history()
                                     ? model.dg_source_publication_allowance(*model.fe_strain_state->dg_history,
                                                                            CellId(q.cell),q.position) : 0.);
              if (error > bound)
                this->get_pcout() << std::setprecision(17) << "7B QP mismatch: step=" << step
                                  << ", x=" << q.position << ", error=" << error << ", bound=" << bound
                                  << ", actual=" << actual << ", expected=" << expected << ", old=" << q.incoming_stress
                                  << ", strain=" << strain << ", S=" << S << ", chi=" << q.chi << ", V=" << rate
                                  << ", beta=" << coefficients.beta << ", eta_ve=" << coefficients.eta_ve
                                  << ", sample eta_ve=" << q.eta_ve << ", sample dt=" << q.stress_time_step << std::endl;
              AssertThrow(error <= bound, ExcMessage("7B accepted QP stress differs from direct Maxwell evaluation."));
              stress_error = std::max(stress_error,error);
              stress_fraction = std::max(stress_fraction,error/bound);
              stress_scale = std::max(stress_scale,expected.norm());
              maximum_divergence = std::max(maximum_divergence,std::abs(trace(strain)));
              trace_increment = std::max(trace_increment,std::abs(2*coefficients.eta_ve*trace(strain)/3.));
            }
          const auto &pm = this->get_phase_field_handler().get_associated_particle_manager();
          const auto &info = pm.get_property_manager().get_data_info();
          const auto stress_position = info.get_position_by_field_name("maxwell stress");
          const auto H_position = info.get_position_by_field_name("crack_driving_force");
          std::map<types::particle_index, SymmetricTensor<2,dim> > published;
          for (const auto &a : anchors)
            published.emplace(a.id,SymmetricTensor<2,dim>());
          for (const auto &p : pm.get_particle_handler())
            {
              AssertThrow(fixed_particles.at(p.get_id()).first == p.get_location()
                          && fixed_particles.at(p.get_id()).second == p.get_properties()[H_position],
                          ExcMessage("7B particles moved or inert history changed during mechanics."));
              const auto selected = published.find(p.get_id());
              if (selected != published.end())
                {
                  SymmetricTensor<2,dim> stress;
                  for (unsigned int c=0; c<SymmetricTensor<2,dim>::n_independent_components; ++c)
                    stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)] = p.get_properties()[stress_position+c];
                  selected->second = stress;
                }
            }
          AssertThrow(fixed_particles.size() == pm.get_particle_handler().n_locally_owned_particles(), ExcMessage("7B particle population changed."));
          // Terminal history uses deal.II's physical-point average. Near a
          // face its default 1e-6 unit-cell search tolerance can include both
          // traces. Check that recurrence, and report the host-cell difference
          // separately rather than treating it as a constitutive defect.
          Utilities::MPI::RemotePointEvaluation<dim> particle_cache;
          std::vector<Tensor<2,dim>> sampled_gradients;
          std::vector<double> sampled_phase;
          if (moving_particles)
            {
              std::vector<Point<dim>> points;
              for (const auto &a : anchors) points.push_back(a.position);
              particle_cache.reinit(this->get_phase_field_handler().get_grid_cache(),points);
              sampled_gradients=VectorTools::point_gradients<dim>(particle_cache,this->get_dof_handler(),this->get_solution(),
                                                                  VectorTools::EvaluationFlags::avg,intro.component_indices.velocities[0]);
              sampled_phase=VectorTools::point_values<1>(particle_cache,this->get_dof_handler(),this->get_solution(),
                                                         VectorTools::EvaluationFlags::avg,intro.variable("phase_field").first_component_index);
            }
          unsigned int particle_index=0, multiple_traces=0;
          double host_difference=0, host_difference_sum=0;
          for (const auto &a : anchors)
            {
              const auto cell = this->get_dof_handler().get_triangulation().create_cell_iterator(a.cell);
              const typename DoFHandler<dim>::active_cell_iterator dof_cell(&this->get_triangulation(),cell->level(),cell->index(),&this->get_dof_handler());
              const auto unit = this->get_mapping().transform_real_to_unit_cell(dof_cell,a.position);
              FEValues<dim> fe(this->get_mapping(),this->get_fe(),Quadrature<dim>({unit}),update_values|update_gradients);
              fe.reinit(dof_cell);
              std::vector<SymmetricTensor<2,dim> > strain(1);
              std::vector<double> phi(1);
              fe[intro.extractors.velocities].get_function_symmetric_gradients(this->get_solution(),strain);
              fe[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)].get_function_values(this->get_solution(),phi);
              const auto source = manager.project_to_bulk_source(a.position);
              double chiV = 0.;
              SymmetricTensor<2,dim> S;
              if (source.active)
                {
                  auto t = fault.vertex(source.segment_index+1)-fault.vertex(source.segment_index);
                  t /= t.norm();
                  Tensor<1,dim> n;
                  n[0] = -t[1];
                  n[1] = t[0];
                  S = manager.get_shear_sense(0)*symmetrize(outer_product(t,n));
                  const auto l = model.evaluate_reconstructed_fault_localization(0,source.segment_index,source.xi,phi[0],phi[0],"7B point control");
                  chiV = l.current_h/l.current_I_h*manager.interpolate_slip_rate(0,source.segment_index,source.xi);
                }
              const auto effective = model.uses_deviatoric_strain_rate() ? Utilities::Tensors::consistent_deviator(strain[0]) : strain[0];
              auto expected = step ? coefficients.beta*a.stress+2*coefficients.eta_ve*(effective-chiV*S) : SymmetricTensor<2,dim>();
              double scale = coefficients.beta*a.stress.norm()+2*coefficients.eta_ve*(strain[0].norm()+std::abs(chiV)*S.norm());
              scale = std::max(scale,coefficients.beta*a.stress.norm()
                               +2*coefficients.eta_ve*(gradient_operand(a.cell,unit)+std::abs(chiV)*S.norm()));
              if (step && model.uses_independent_bulk_history())
                {
                  const auto &s = *model.fe_strain_state;
                  double new_scale = 0.,old_scale = 0.;
                  if (model.uses_dg_trial_stress_history())
                    {
                      model.evaluate_dg_history_stress(*s.dg_history,s.Z,s.I,a.cell,a.position,&new_scale);
                      model.evaluate_dg_history_stress(*incoming_dg,incoming_Z,initial_I,a.cell,a.position,&old_scale);
                    }
                  else
                    {
                      model.evaluate_fe_strain_stress(s.displacement_history,s.Z,s.I,a.cell,a.position,&new_scale);
                      model.evaluate_fe_strain_stress(incoming_fe,incoming_Z,initial_I,a.cell,a.position,&old_scale);
                    }
                  scale = std::max(scale,2*(new_scale+coefficients.beta*old_scale));
                }
              if (moving_particles)
                {
                  auto rate=symmetrize(sampled_gradients[particle_index]);
                  if (model.uses_deviatoric_strain_rate()) rate=Utilities::Tensors::consistent_deviator(rate);
                  double sampled_chiV=0;
                  if (source.active)
                    {
                      const auto l=model.evaluate_reconstructed_fault_localization(0,source.segment_index,source.xi,
                                                                                   sampled_phase[particle_index],sampled_phase[particle_index],"moving point control");
                      sampled_chiV=l.current_h/l.current_I_h*manager.interpolate_slip_rate(0,source.segment_index,source.xi);
                    }
                  const auto sampled_expected = step ? coefficients.beta * a.stress + 2 * coefficients.eta_ve * (rate - sampled_chiV * S) : SymmetricTensor<2, dim>();
                  const double difference = (sampled_expected - expected).norm();
                  if (population_event_audit && step == 34 && difference > host_difference)
                    face_probe = std::make_unique<ParticleAnchor>(a);
                  host_difference = std::max(host_difference, difference);
                  host_difference_sum += difference;
                  const auto &ptrs = particle_cache.get_point_ptrs();
                  if (ptrs[particle_index + 1] - ptrs[particle_index] > 1) ++multiple_traces;
                  expected = sampled_expected;
                  ++particle_index;
                }
              const double error = (published.at(a.id)-expected).norm();
              const double bound = 128*std::numeric_limits<double>::epsilon()*std::max(1.,scale)
                                   +(step && model.uses_dg_trial_stress_history()
                                     ? model.dg_source_publication_allowance(*model.fe_strain_state->dg_history,a.cell,a.position) : 0.);
              AssertThrow(error <= bound, ExcMessage("7B published particle sample has an incorrect Maxwell increment."));
              stress_error = std::max(stress_error,error);
              stress_fraction = std::max(stress_fraction,error/bound);
              stress_scale = std::max(stress_scale,expected.norm());
            }
          if (moving_particles)
            {
              std::ofstream sampling(local_file("moving-sampling.csv"),step ? std::ios::app : std::ios::out);
              if (!step) sampling<<"step,time_s,multiple_traces,max_host_difference_Pa,mean_host_difference_Pa\n";
              sampling<<std::setprecision(17)<<step<<','<<this->get_time()<<','<<multiple_traces<<','<<host_difference
                      <<','<<host_difference_sum/anchors.size()<<'\n';
            }
          // Three short normal cuts cross the graded interfaces. Retain both
          // cell traces at a shared face: a stress jump is not a point average.
          std::ofstream bulk(emit_fields ? local_file("bulk-normal-cuts-"+std::to_string(step)+".csv") : "/dev/null");
          bulk << std::setprecision(17) << "station,r_m,cell,edge_m,pressure_Pa,deviatoric_Pa,normal_Pa,phi,divergence_per_s,instant_tensor_normal_difference_Pa,tau_xx,tau_yy,tau_zz,tau_xy\n";
          Tensor<1,dim> normal,tangent;
          for (unsigned int d=0; d<2; ++d)
            {
              normal[d] = BP3::geometry().normal[d];
              tangent[d] = BP3::geometry().tangent[d];
            }
          const auto N = symmetrize(outer_product(normal,normal));
          const auto S = manager.get_shear_sense(0)*symmetrize(outer_product(tangent,normal));
          std::array<unsigned int,SymmetricTensor<2,dim>::n_independent_components> mapped_stress;
          mapped_stress.fill(numbers::invalid_unsigned_int);
          for (const auto &entry : this->get_parameters().mapped_particle_properties)
            if (entry.second.first == "maxwell stress") mapped_stress[entry.second.second] = entry.first;
          if (model.uses_homogeneous_mesh_data())
            {
              const auto fields=intro.get_indices_for_fields_of_type(CompositionalFieldDescription::stress);
              AssertThrow(fields.size()==mapped_stress.size(),ExcMessage("Mature diagnostic needs three stress labels."));
              std::copy(fields.begin(),fields.end(),mapped_stress.begin());
            }
          const auto &working = this->get_current_linearization_point();
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              for (unsigned int station=1; station<=3; ++station)
                {
                  Point<dim> foot;
                  for (unsigned int d=0; d<2; ++d)
                    foot[d] = BP3::geometry().upper[d]+station*BP3::geometry().length/4.*tangent[d];
                  const auto bounds = cell->bounding_box().get_boundary_points();
                  double lo = -150.,hi = 150.;
                  for (unsigned int d=0; d<dim; ++d)
                    if (normal[d] != 0.)
                      {
                        double first = (bounds.first[d]-foot[d])/normal[d],last = (bounds.second[d]-foot[d])/normal[d];
                        if (first > last) std::swap(first,last);
                        lo = std::max(lo,first);
                        hi = std::min(hi,last);
                      }
                  if (hi <= lo) continue;
                  std::vector<double> radii {lo,hi};
                  for (int i=std::ceil(lo/2.); 2*i<hi; ++i) radii.push_back(2*i);
                  std::sort(radii.begin(),radii.end());
                  radii.erase(std::unique(radii.begin(),radii.end()),radii.end());
                  std::vector<Point<dim> > units;
                  for (const double r : radii)
                    {
                      auto unit = this->get_mapping().transform_real_to_unit_cell(cell,foot+r*normal);
                      AssertThrow(GeometryInfo<dim>::distance_to_unit_cell(unit)<1.e-12, ExcMessage("7B clipped bulk cut left its cell."));
                      for (unsigned int d=0; d<dim; ++d) unit[d] = std::clamp(unit[d],0.,1.);
                      units.push_back(unit);
                    }
                  FEValues<dim> fe(this->get_mapping(),this->get_fe(),Quadrature<dim>(units),update_values|update_gradients|update_quadrature_points);
                  fe.reinit(cell);
                  std::vector<double> p(radii.size()),phi(radii.size());
                  std::vector<SymmetricTensor<2,dim> > strain(radii.size());
                  std::array<std::vector<double>,SymmetricTensor<2,dim>::n_independent_components> old;
                  fe[intro.extractors.pressure].get_function_values(working,p);
                  fe[intro.extractors.velocities].get_function_symmetric_gradients(working,strain);
                  fe[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)].get_function_values(working,phi);
                  if (!model.uses_independent_bulk_history())
                    for (unsigned int c=0; c<old.size(); ++c)
                      {
                        old[c].resize(radii.size());
                        fe[intro.extractors.compositional_fields[mapped_stress[c]]].get_function_values(working,old[c]);
                      }
                  for (unsigned int q=0; q<radii.size(); ++q)
                    {
                      const auto x = fe.quadrature_point(q);
                      SymmetricTensor<2,dim> stress;
                      if (step && model.uses_independent_bulk_history())
                        stress = model.retained_history_stress(cell->id(),x, {});
                      else
                        {
                          if (!model.uses_independent_bulk_history())
                            for (unsigned int c=0; c<old.size(); ++c)
                              stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)] = coefficients.beta*old[c][q];
                          double chiV = 0.;
                          const auto source = manager.project_to_bulk_source(x);
                          if (source.active)
                            {
                              const auto l = model.evaluate_reconstructed_fault_localization(0,source.segment_index,source.xi,phi[q],phi[q],"7B bulk cut");
                              chiV = l.current_h/l.current_I_h*manager.interpolate_slip_rate(0,source.segment_index,source.xi);
                            }
                          const auto effective = model.uses_deviatoric_strain_rate() ? Utilities::Tensors::consistent_deviator(strain[q]) : strain[q];
                          stress += 2*coefficients.eta_ve*(effective-chiV*S);
                        }
                      const double deviatoric = -(stress*N);
                      bulk << station << ',' << radii[q] << ',' << cell->id() << ',' << cell->diameter()/std::sqrt(double(dim))
                           << ',' << p[q] << ',' << deviatoric << ',' << BP3::sigma0+p[q]+deviatoric << ',' << phi[q]
                           << ',' << trace(strain[q]) << ',' << 2*coefficients.eta_ve *trace(strain[q])/3.
                           << ',' << stress[0][0] << ',' << stress[1][1] << ',' << -stress[0][0]-stress[1][1]
                           << ',' << stress[0][1] << '\n';
                    }
                }
          if (write_authoritative_stress && emit_fields)
            {
              TimerOutput::Scope timer(this->get_computing_timer(),"BP3 dense stress output");
              // Same physical points for all backends. Particle history follows
              // its actual frozen mapped FE input and native Maxwell operator;
              // independent history is read directly after accepted publication.
              for(unsigned int dataset=0;dataset<(common_points.empty()?1u:2u);++dataset)
                {
                  const bool common=dataset==1;
                  std::ofstream out(local_file((common?"common-stress-":"authoritative-stress-")+std::to_string(step)+".csv"));
                  out<<std::setprecision(17)<<"cell,qp,x,y,JxW,phi,chi,Z,T_xx,T_yy,T_xy,tau_xx,tau_yy,tau_zz,tau_xy,p\n";
                  for (const auto &cell:this->get_dof_handler().active_cell_iterators())
                    if (cell->is_locally_owned())
                      {
                        std::vector<Point<dim>> units;
                        const CommonPoints *physical=nullptr;
                        if(common)
                          {
                            const auto found=common_points.find(cell->id().to_string());
                            AssertThrow(found!=common_points.end(),ExcMessage("Common points omit a native cell."));
                            physical=&found->second;
                            for(const auto &x:physical->physical)
                              {
                                const auto unit=this->get_mapping().transform_real_to_unit_cell(cell,x);
                                AssertThrow(GeometryInfo<dim>::is_inside_unit_cell(unit),ExcMessage("Wrong common-point cell trace."));
                                units.push_back(unit);
                              }
                          }
                        const Quadrature<dim> points=common ? Quadrature<dim>(units) : Quadrature<dim>(QGauss<dim>(3));
                        FEValues<dim> fe(this->get_mapping(),this->get_fe(),points,
                          update_values|update_gradients|update_quadrature_points|update_JxW_values);
                        std::vector<double> phase(points.size()),pressure(points.size());
                        std::vector<SymmetricTensor<2,dim>> strain(points.size());
                        std::array<std::vector<double>,3> old;
                    fe.reinit(cell);
                    fe[intro.extractors.pressure].get_function_values(working,pressure);
                    fe[intro.extractors.velocities].get_function_symmetric_gradients(working,strain);
                    fe[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)].get_function_values(working,phase);
                    if(model.uses_stationary_mature_dg_optimizations())
                      model.prepare_dg_point_samples(cell->id(),common ? physical->physical : fe.get_quadrature_points());
                    if (!model.uses_independent_bulk_history())
                      for (unsigned int c=0;c<old.size();++c)
                        {
                          old[c].resize(points.size());
                          fe[intro.extractors.compositional_fields[mapped_stress[c]]].get_function_values(working,old[c]);
                        }
                    for (unsigned int q=0;q<points.size();++q)
                      {
                        const auto x=common ? physical->physical[q] : fe.quadrature_point(q);
                        const auto source=manager.project_to_bulk_source(x);
                        double chi=0.,Z=0.,rate=0.;
                        if (source.active)
                          {
                            const auto l=model.evaluate_reconstructed_fault_localization(0,source.segment_index,source.xi,phase[q],phase[q],"authoritative export");
                            chi=l.current_h/l.current_I_h;
                            rate=manager.interpolate_slip_rate(0,source.segment_index,source.xi);
                            if (model.uses_independent_bulk_history())
                              Z=(1-source.xi)*model.fe_strain_state->Z[source.segment_index]+source.xi*model.fe_strain_state->Z[source.segment_index+1];
                          }
                        SymmetricTensor<2,dim> tau,T;
                        if (model.uses_independent_bulk_history())
                          {
                            tau=model.retained_history_stress(cell->id(),x,{});
                            T=tau+2*chi*Z*S;
                          }
                        else if (step)
                          {
                            for (unsigned int c=0;c<old.size();++c)
                              tau[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]=coefficients.beta*old[c][q];
                            tau+=2*coefficients.eta_ve*(Utilities::Tensors::consistent_deviator(strain[q])-chi*rate*S);
                            // Particle path does not own independent T/Z.
                            Z=std::numeric_limits<double>::quiet_NaN();T=tau;
                          }
                        out<<cell->id()<<','<<(common?physical->ids[q]:q)<<','<<x[0]<<','<<x[1]<<','<<(common?physical->weights[q]:fe.JxW(q))<<','<<phase[q]<<','<<chi<<','<<Z
                           <<','<<T[0][0]<<','<<T[1][1]<<','<<T[0][1]<<','<<tau[0][0]<<','<<tau[1][1]
                           <<','<<-tau[0][0]-tau[1][1]<<','<<tau[0][1]<<','<<pressure[q]<<'\n';
                      }
                  }
                  AssertThrow(bool(out),ExcMessage("Cannot export authoritative stress."));
                }
              if (model.uses_dg_trial_stress_history())
                {
                  const auto &history=*model.fe_strain_state->dg_history;
                  if(step==0)
                    {
                      std::ofstream support(local_file("dg-support.csv"));
                      support<<std::setprecision(17)<<"mode,xi,eta\n";
                      const auto &nodes=history.space->element.get_unit_support_points();
                      for(unsigned int j=0;j<nodes.size();++j)
                        support<<j<<','<<nodes[j][0]<<','<<nodes[j][1]<<'\n';
                    }
                  std::ofstream coefficients_file(local_file("dg-coefficients-"+std::to_string(step)+".csv"));
                  coefficients_file<<std::setprecision(17)<<"cell,mode,xx,yy,xy\n";
                  std::vector<types::global_dof_index> indices(history.space->element.n_dofs_per_cell());
                  for (const auto &cell:history.space->dofs.active_cell_iterators())
                    if (cell->is_locally_owned())
                      {
                        cell->get_dof_indices(indices);
                        for (unsigned int j=0;j<indices.size();++j)
                          coefficients_file<<cell->id()<<','<<j<<','<<history.trial_stress[0][indices[j]]<<','
                            <<history.trial_stress[1][indices[j]]<<','<<history.trial_stress[2][indices[j]]<<'\n';
                      }
                }
            }
          if (write_coefficients && emit_fields)
            {
              TimerOutput::Scope timer(this->get_computing_timer(),"BP3 dense coefficient output");
              std::ofstream coefficients_file(local_file("physical-fe-"+std::to_string(step)+".csv"));
              coefficients_file<<std::setprecision(17)<<"component,x,y,accepted,history\n";
              const auto &fe=this->get_fe();
              const auto &history=model.uses_independent_bulk_history() ? model.accepted_fe_strain_state() : this->get_solution();
              std::set<types::global_dof_index> written;
              std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
              for (const auto &cell:this->get_dof_handler().active_cell_iterators())
                if (cell->is_locally_owned())
                  {
                    cell->get_dof_indices(indices);
                    for (unsigned int j=0; j<indices.size(); ++j)
                      {
                        const auto c=fe.system_to_component_index(j).first;
                        if ((c<dim || c==intro.component_indices.pressure || c==intro.variable("phase_field").first_component_index)
                            && this->get_dof_handler().locally_owned_dofs().is_element(indices[j])
                            && written.insert(indices[j]).second)
                          {
                            const auto x=this->get_mapping().transform_unit_to_real_cell(cell,fe.get_unit_support_points()[j]);
                            coefficients_file<<c<<','<<x[0]<<','<<x[1]<<','<<this->get_solution()[indices[j]]<<','
                                             <<(c<dim ? (model.uses_independent_bulk_history() ? history[indices[j]]
                                                  : std::numeric_limits<double>::quiet_NaN()) : 0.)<<'\n';
                          }
                      }
                  }
              AssertThrow(bool(coefficients_file),ExcMessage("Cannot write physical-coordinate FE comparison."));
            }
          const auto path = local_file("coupled-checks.csv");
          const bool header = BP3::needs_header(path);
          std::ofstream out(path,std::ios::app);
          out << std::setprecision(17);
          if (header)
            out << "step,time_s,dt_s,mechanical_qps,qp_controls,particle_controls,particles,min_I_m,min_theta_s,Z_max_Pa_m,Z_error_Pa_m,stress_error_Pa,error_fraction_of_allowance,stress_scale_Pa,normal_split_error_Pa,max_divergence_per_s,instant_tensor_normal_difference_Pa,publications\n";
          out << step << ',' << this->get_time() << ',' << this->get_timestep() << ',' << fault_mechanical_quadrature(*this).size()
              << ',' << diagnostic.samples.size() << ',' << anchors.size() << ',' << fixed_particles.size()
              << ',' << *std::min_element(initial_I.begin(),initial_I.end()) << ',' << theta_min << ',' << Z_max << ',' << Z_error
              << ',' << stress_error << ',' << stress_fraction << ',' << stress_scale << ',' << normal_split_error
              << ',' << maximum_divergence << ',' << trace_increment << ',' << solver_publications << '\n';
          AssertThrow(out && profile && bulk, ExcMessage("Cannot write compact 7B results."));
          if (scheduled_fields && emit_fields)
            {
              BP3::collective_root_write(this->get_mpi_communicator(),[&]()
                {
                  const auto path=this->get_output_directory()+"field-outputs.csv";
                  const bool header=BP3::needs_header(path);
                  std::ofstream fields(path,std::ios::app);fields.exceptions(std::ios::failbit|std::ios::badbit);
                  if(header)fields<<"step,time_s,max_V_m_s\n";
                  fields<<std::setprecision(17)<<step<<','<<this->get_time()<<','<<*std::max_element(V.begin(),V.end())<<'\n';
                });
              field_schedule.written(this->get_time(),accepted_slip);
            }
          this->get_pcout() << "   BP3 7B accepted check: step=" << step << ", stress error=" << stress_error
                            << ", allowance fraction=" << stress_fraction << ", stress scale=" << stress_scale << " Pa." << std::endl;
          if (moving_particles) write_moving_check();
          if (population_event_audit && step == 34)
            {
              AssertThrow(bool(face_probe), ExcMessage("The retained near-face event was not reproduced."));
              audit_near_face(*face_probe);
            }
          if(std::getenv("ASPECT_FAULT_COMPARE_COUPLING"))verify_stationary_operators();
          if(std::getenv("ASPECT_FAULT_PERFORMANCE"))
            this->get_pcout()<<"Fault sampling counters: step="<<step<<aspect::internal::FaultSamplingCounters::report()<<std::endl;
          return {"BP3 serial paired history",std::to_string(step)};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"BP3 serial coupled checks",
                                  "Bounded serial stationary-history paired BP3 publication and compact response checks.")
  }
}
