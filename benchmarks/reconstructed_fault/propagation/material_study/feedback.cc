// Read-only P2 observer. Loading is a native boundary function. H is never set
// here.
#include "../../../../tests/phase_field_fault_stage_i.cc"
#include "../../../../tests/phase_field_fault_test_access.h"
#include "../../../../tests/phase_field_test_access.h"
#include <aspect/particle/manager.h>
#include <aspect/particle/particle_domain.h>
#include <aspect/plugins.h>
#include <aspect/postprocess/interface.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/simulator_signals.h>
#include <cstdlib>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools_evaluate.h>
#include <fstream>
#include <iomanip>
#include <map>

namespace aspect {
namespace Postprocess {
template <int dim>
class CoupledFeedback : public Interface<dim>, public SimulatorAccess<dim> {
  using V = LinearAlgebra::BlockVector;
  using M = LinearAlgebra::BlockSparseMatrix;
  using A = internal::PhaseFieldTestAccess<dim>;
  using F = MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
  std::map<types::particle_index, double> incoming, initial_H, initial_phi;
  std::map<types::particle_index, Point<dim>> positions;
  std::map<types::particle_index, std::vector<double>> incoming_stress;
  std::vector<double> previous_q, previous_I;
  double phase_kkt = 0, mechanical_residual = 0, min_alpha = 1;
  unsigned int newton = 0, krylov = 0, observations = 0;
  bool phase_checked = false;
  unsigned int b() const {
    return this->introspection().variable("phase_field").block_index;
  }
  auto &pm() const {
    return this->get_phase_field_handler().get_associated_particle_manager();
  }
  auto &material() const {
    return Plugins::get_plugin_as_type<
        const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
  }
  unsigned int hi() const {
    return pm()
        .get_property_manager()
        .get_data_info()
        .get_position_by_field_name("crack_driving_force");
  }
  std::vector<double> property(const std::string &name) const {
    auto &manager = this->get_reconstructed_fault_manager();
    const auto pos =
        manager.get_property_information()[manager.get_property_index(name)]
            .position;
    std::vector<double> values;
    const auto &fault = manager.get_fault(0);
    for (unsigned int v = 0; v < fault.n_vertices(); ++v)
      values.push_back(fault.get_properties(v)[pos]);
    return values;
  }
  #include "diagnostics.h"
  void start() {
    incoming.clear();
    incoming_stress.clear();
    const auto stress_position =
        pm().get_property_manager().get_data_info().get_position_by_field_name(
            "maxwell stress");
    for (const auto &p : pm().get_particle_handler()) {
      incoming[p.get_id()] = p.get_properties()[hi()];
      for (unsigned int i = 0;
           i < SymmetricTensor<2, dim>::n_independent_components; ++i)
        incoming_stress[p.get_id()].push_back(
            p.get_properties()[stress_position + i]);
      if (this->get_timestep_number() == 0)
        positions[p.get_id()] = p.get_location();
      else
        AssertThrow(positions.at(p.get_id()) == p.get_location(),
                    ExcMessage("P2 isolation moved a particle."));
    }
    if (this->get_timestep_number() > 0) {
      previous_q = property("phase field fault cohesive traction");
      previous_I = property("phase field fault previous I h");
    }
    phase_checked = false;
    observations = 0;
  }
  void diagnose_normalization() {
    auto &model = const_cast<MaterialModel::PhaseFieldFault<dim> &>(material());
    const auto saved = F::current_normalization_integrals(model);
    // Small serial frozen-field artifact: no history or phase is modified.
    std::ofstream phase_nodes(this->get_output_directory() + "frozen-phase-" +
                              std::to_string(this->get_timestep_number()) + ".csv");
    phase_nodes << std::setprecision(17) << "x,y,phi,g\n";
    const auto &vertex_dofs = A::vertex_dofs(this->get_phase_field_handler());
    for (unsigned int v = 0; v < vertex_dofs.size(); ++v)
      if (vertex_dofs[v] != numbers::invalid_dof_index &&
          this->get_solution().locally_owned_elements().is_element(vertex_dofs[v])) {
        const auto &point = this->get_triangulation().get_vertices()[v];
        const double phi = this->get_solution()[vertex_dofs[v]];
        phase_nodes << point[0] << ',' << point[1] << ',' << phi << ','
                    << this->get_phase_field_handler().energetic_degradation({.5,.5}, phi)
                    << '\n';
      }
    phase_nodes.close();

    M matrix;
    const auto &system = this->get_system_matrix();
    matrix.reinit(system.n_block_rows(), system.n_block_cols());
    for (unsigned int i = 0; i < system.n_block_rows(); ++i)
      for (unsigned int j = 0; j < system.n_block_cols(); ++j)
        matrix.block(i, j).copy_from(system.block(i, j));
    matrix.collect_sizes();
    V rhs(this->introspection().index_sets.system_partitioning,
          this->get_mpi_communicator());
    auto &phase_handler = this->get_phase_field_handler();
    A::assemble(phase_handler, matrix, rhs, this->get_solution(), true);
    A::constrain(phase_handler, matrix, rhs, this->get_solution());
    V correction(rhs);
    correction = 0;
    A::solve(phase_handler, matrix, rhs, correction);
    const double estimate = correction.block(b()).linfty_norm();
    V owned(rhs), repeat(this->get_solution());
    owned = repeat;
    const auto &dofs = A::vertex_dofs(phase_handler);
    for (unsigned int v = 0; v < dofs.size(); ++v)
      if (dofs[v] != numbers::invalid_dof_index &&
          owned.locally_owned_elements().is_element(dofs[v]))
        owned[dofs[v]] = std::clamp(
            owned[dofs[v]] +
                1e-6 * std::sin(
                           23 * this->get_triangulation().get_vertices()[v][0] +
                           5 * this->get_triangulation().get_vertices()[v][1]),
            0.0, 1.0);
    repeat = owned;
    A::bounded(phase_handler, matrix, rhs, repeat);
    owned = repeat;
    V accepted(rhs);
    accepted = this->get_solution();
    owned -= accepted;
    std::ofstream uncertainty(
        this->get_output_directory() + "failed-phase-uncertainty-" +
        std::to_string(this->get_timestep_number()) + ".csv");
    uncertainty << std::setprecision(17)
                << "jacobian_correction,identical_H_perturbed_repeat\n"
                << estimate << ',' << owned.block(b()).linfty_norm() << '\n';

    double offband_phi = 0, active_distance = 0;
    const auto initial_profiles = phase_handler.get_phase_field_profiles(study_core());
    for (unsigned int v = 0; v < dofs.size(); ++v)
      if (dofs[v] != numbers::invalid_dof_index &&
          accepted.locally_owned_elements().is_element(dofs[v])) {
        const double distance =
            std::abs(this->get_triangulation().get_vertices()[v][1] - .5);
        if (initial_profiles[0]->value(distance) <= 1e-6)
          offband_phi = std::max(offband_phi, static_cast<double>(accepted[dofs[v]]));
        if (accepted[dofs[v]] > 1e-6)
          active_distance = std::max(active_distance, distance);
      }
    std::ofstream spread(this->get_output_directory() + "failed-phase-spread-" +
                         std::to_string(this->get_timestep_number()) + ".csv");
    spread << std::setprecision(17)
           << "offband_nodal_phi_max,active_nodal_normal_distance_max\n"
           << offband_phi << ',' << active_distance << '\n';
    const auto profiles = F::normalization_profiles(model);
    AssertThrow(this->get_triangulation().n_global_active_cells() == 1024,
                ExcMessage("Independent normal integration assumes the "
                           "selected 32 by 32 mesh."));
    std::vector<std::vector<double>> integrals, projected;
    for (const unsigned int order : {16u, 32u}) {
      QGauss<1> rule(order);
      std::vector<Point<dim>> points;
      for (const auto &profile : profiles)
        for (unsigned int cell = 0; cell < 32; ++cell)
          for (unsigned int j = 0; j < order; ++j) {
            auto p = profile.origin;
            p[1] = (cell + rule.point(j)[0]) / 32.;
            points.push_back(p);
          }
      Utilities::MPI::RemotePointEvaluation<dim> cache;
      cache.reinit(this->get_phase_field_handler().get_grid_cache(), points);
      const auto phi = VectorTools::point_values<1>(
          cache, this->get_dof_handler(), this->get_solution(),
          VectorTools::EvaluationFlags::avg,
          this->introspection().variable("phase_field").first_component_index);
      std::vector<double> values(profiles.size(), 0.);
      unsigned int index = 0;
      for (unsigned int i = 0; i < profiles.size(); ++i)
        for (unsigned int cell = 0; cell < 32; ++cell)
          for (unsigned int j = 0; j < order; ++j, ++index)
            values[i] +=
                rule.weight(j) / 32. *
                F::normalization_integrand(
                    phi[index],
                    this->get_phase_field_handler().energetic_degradation(
                        profiles[i].material_fractions, phi[index]));
      projected.push_back(F::probe_normalization_projection(model, values));
      integrals.push_back(values);
    }
    AssertThrow(F::current_normalization_integrals(model) == saved,
                ExcMessage("Normalization diagnostic changed native values."));
    const auto suffix = std::to_string(this->get_timestep_number()) + ".csv";
    std::ofstream raw(this->get_output_directory() +
                      "normalization-positive-profiles-" + suffix);
    raw << std::setprecision(17) << "x,Ih_Q16,Ih_Q32\n";
    for (unsigned int i = 0; i < profiles.size(); ++i)
      raw << profiles[i].origin[0] << ',' << integrals[0][i] << ','
          << integrals[1][i] << '\n';
    std::ofstream nodes(this->get_output_directory() +
                        "normalization-projection-diagnosis-" + suffix);
    nodes << std::setprecision(17)
          << "x,native_Ih,Q16_projected,Q32_projected\n";
    const auto &fault = this->get_reconstructed_fault_manager().get_fault(0);
    for (unsigned int i = 0; i < saved[0].size(); ++i)
      nodes << fault.vertex(i)[0] << ',' << saved[0][i] << ','
            << projected[0][i] << ',' << projected[1][i] << '\n';
  }
  void phase_check() {
    const auto stress_position =
        pm().get_property_manager().get_data_info().get_position_by_field_name(
            "maxwell stress");
    for (const auto &p : pm().get_particle_handler()) {
      AssertThrow(p.get_properties()[hi()] == incoming.at(p.get_id()),
                  ExcMessage("Mechanical trial accumulated H."));
      for (unsigned int i = 0;
           i < SymmetricTensor<2, dim>::n_independent_components; ++i)
        AssertThrow(incoming_stress.at(p.get_id())[i] ==
                        p.get_properties()[stress_position + i],
                    ExcMessage("Mechanical trial accumulated Maxwell stress."));
    }
    if (this->get_timestep_number() > 0)
      AssertThrow(property("phase field fault cohesive traction") ==
                          previous_q &&
                      property("phase field fault previous I h") == previous_I,
                  ExcMessage("Mechanical trial accumulated cohesive history."));
    ++observations;
    if (phase_checked)
      return;
    const auto &original = this->get_system_matrix();
    M matrix;
    matrix.reinit(original.n_block_rows(), original.n_block_cols());
    for (unsigned int i = 0; i < original.n_block_rows(); ++i)
      for (unsigned int j = 0; j < original.n_block_cols(); ++j)
        matrix.block(i, j).copy_from(original.block(i, j));
    matrix.collect_sizes();
    V rhs(this->introspection().index_sets.system_partitioning,
          this->get_mpi_communicator());
    A::assemble(this->get_phase_field_handler(), matrix, rhs,
                this->get_solution(), false);
    phase_kkt = 0;
    for (auto i : rhs.block(b()).locally_owned_elements()) {
      const double x = this->get_solution().block(b())[i],
                   r = rhs.block(b())[i];
      phase_kkt = std::max(phase_kkt,
                           std::abs(x == 0 ? std::max(r, 0.0)
                                           : (x == 1 ? std::min(r, 0.0) : r)));
    }
    AssertThrow(
        phase_kkt <= 1e-11,
        ExcMessage("P2 phase not equilibrated against incoming mechanical H."));
    phase_checked = true;
  }

public:
  void initialize() override {
    AssertThrow(dim == 2 && Utilities::MPI::n_mpi_processes(
                                this->get_mpi_communicator()) == 1,
                ExcMessage("Initial P2 observer is bounded to serial 2D."));
    AssertThrow(
        std::getenv("ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION"),
        ExcMessage("P2 uses the documented existing frozen RK2 control."));
    this->get_signals().start_timestep.connect(
        [this](const SimulatorAccess<dim> &) { start(); });
    this->get_signals().post_particle_management.connect(
        [this](Particle::Manager<dim> &) {
          if (this->get_timestep_number() == 0 ||
              this->get_reconstructed_fault_manager().get_faults().empty())
            return;
          phase_check();
          // Capture the phase consumption even if later mechanical preparation
          // fails. Native normalization evaluation only refreshes transient
          // numerical data; this fixed-mixture control does not alter any
          // committed history.
          auto &model =
              const_cast<MaterialModel::PhaseFieldFault<dim> &>(material());
          const auto values = F::compute_normalization_integrals(model).at(0);
          AssertThrow(
              property("phase field fault cohesive traction") == previous_q &&
                  property("phase field fault previous I h") == previous_I,
              ExcMessage(
                  "Normalization observation changed committed history."));
          double lo = 1, hi = 0, healing = 0, change = 0;
          for (auto i :
               this->get_solution().block(b()).locally_owned_elements()) {
            const double x = this->get_solution().block(b())[i],
                         old = this->get_old_solution().block(b())[i];
            lo = std::min(lo, x);
            hi = std::max(hi, x);
            healing = std::max(healing, old - x);
            change = std::max(change, std::abs(old - x));
          }
          std::ofstream out(this->get_output_directory() + "phase-entry-" +
                            std::to_string(this->get_timestep_number()) +
                            ".csv");
          out << std::setprecision(17)
              << "time,phase_kkt,phi_min,phi_max,change,healing,Ih_min,Ih_max\n"
              << this->get_time() << ',' << phase_kkt << ',' << lo << ',' << hi
              << ',' << change << ',' << healing << ','
              << *std::min_element(values.begin(), values.end()) << ','
              << *std::max_element(values.begin(), values.end()) << '\n';
          std::ofstream nodes(
              this->get_output_directory() + "normalization-entry-" +
              std::to_string(this->get_timestep_number()) + ".csv");
          nodes << std::setprecision(17) << "x,Ih\n";
          for (unsigned int i = 0; i < values.size(); ++i)
            nodes
                << this->get_reconstructed_fault_manager().get_fault(0).vertex(
                       i)[0]
                << ',' << values[i] << '\n';
          out.close();
          nodes.close();
          study_phase("entry", values);
          AssertThrow(lo >= 0 && hi <= 1 && healing <= 4.922e-12,
                      ExcMessage("P2 material-study phase admissibility failed; entry retained."));
          if (*std::min_element(values.begin(), values.end()) <= 0)
            diagnose_normalization();
        });
    this->get_signals().post_reconstructed_fault_linear_solver.connect(
        [this](const auto &, const auto &, const auto &, const auto &,
               const auto &, const auto &, double, unsigned int,
               double) { phase_check(); });
    this->get_signals().post_reconstructed_fault_solver.connect(
        [this](unsigned int n, unsigned int k, double a, const auto &) {
          newton = n;
          krylov = k;
          min_alpha = a;
        });
    this->get_signals().post_nonlinear_solver.connect(
        [this](const SolverControl &s) {
          mechanical_residual = s.last_value();
        });
  }
  std::pair<std::string, std::string> execute(TableHandler &) override {
    const unsigned int step = this->get_timestep_number();
    const double dt = this->get_timestep(), time = this->get_time();
    auto &manager = this->get_reconstructed_fault_manager();
    auto &h = this->get_phase_field_handler();
    const auto &state = this->get_solution();
    if (!phase_checked)
      phase_check(); // Valid only if no history changed (asserted).
    const auto q = property("phase field fault cohesive traction");
    const auto Ih = F::current_normalization_integrals(material()).at(0);
    const auto previous_after = property("phase field fault previous I h");
    AssertThrow(
        Ih == previous_after,
        ExcMessage("Committed normalization does not match current phase."));
    if (step == 0) {
      previous_q = q;
      previous_I = Ih;
      initial_H = incoming;
    }
    study_begin(step);
    study_phase("accepted", Ih);
    double min_phi = 1, max_phi = 0, healing = 0, change = 0;
    for (auto i : state.block(b()).locally_owned_elements()) {
      const double x = state.block(b())[i];
      min_phi = std::min(min_phi, x);
      max_phi = std::max(max_phi, x);
      if (step > 0) {
        healing = std::max(healing, this->get_old_solution().block(b())[i] - x);
        change = std::max(change,
                          std::abs(x - this->get_old_solution().block(b())[i]));
      }
    }
    AssertThrow(min_phi >= 0 && max_phi <= 1,
                ExcMessage("P2 phase bounds violated."));
    struct Bin {
      double volume = 0, H0 = 0, Hin = 0, Hnew = 0, candidate = 0,
             cmax = -1e300, phi0 = 0, phi = 0, phimax = 0, dHmax = 0;
      unsigned int increases = 0;
    };
    std::vector<Bin> bins(16);
    double candidate_error = 0, H_change = 0, H_total = 0,
           phase_volume_change = 0, offband_H = 0, offband_phi = 0;
    unsigned int increases = 0, offband_increases = 0;
    const auto profiles = h.get_phase_field_profiles(study_core());
    std::vector<Point<dim>> sample_points;
    for (const auto &a :
         manager.get_locally_owned_particle_fault_associations())
      sample_points.push_back(a.position);
    Utilities::MPI::RemotePointEvaluation<dim> cache;
    cache.reinit(h.get_grid_cache(), sample_points);
    const auto component =
        this->introspection().variable("phase_field").first_component_index;
    const auto native_phi = VectorTools::point_values<1>(
        cache, this->get_dof_handler(), state,
        VectorTools::EvaluationFlags::avg, component);
    const auto native_old = VectorTools::point_values<1>(
        cache, this->get_dof_handler(), this->get_old_solution(),
        VectorTools::EvaluationFlags::avg, component);
    std::map<types::particle_index, std::pair<double, double>> samples;
    unsigned int sample_index = 0;
    for (const auto &p : pm().get_particle_handler()) {
      samples[p.get_id()] = {native_phi[sample_index],
                             native_old[sample_index]};
      ++sample_index;
    }

    const FEValuesExtractors::Scalar phase(
        this->introspection().variable("phase_field").first_component_index);
    for (const auto &cell : this->get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned()) {
        std::vector<Point<dim>> points;
        for (const auto &p :
             pm().get_particle_handler().particles_in_cell(cell))
          points.push_back(p.get_reference_location());
        if (points.empty())
          continue;
        FEValues<dim> fe(this->get_mapping(), this->get_fe(),
                         Quadrature<dim>(points), update_values);
        fe.reinit(cell);
        std::vector<double> phi(points.size()), old_phi(points.size());
        fe[phase].get_function_values(state, phi);
        fe[phase].get_function_values(this->get_old_solution(), old_phi);
        unsigned int j = 0;
        for (const auto &p :
             pm().get_particle_handler().particles_in_cell(cell)) {
          AssertThrow(
              positions.at(p.get_id()) == p.get_location(),
              ExcMessage(
                  "P2 transport/birth control changed positions or IDs."));
          phi[j] = samples.at(p.get_id()).first;
          old_phi[j] = samples.at(p.get_id()).second;
          const auto a = manager.project_to_normal_profiles(p.get_location());
          AssertThrow(a.active,
                      ExcMessage("P2 particle outside full association."));
          const double H = p.get_properties()[hi()],
                       old = incoming.at(p.get_id());
          const auto surface = F::surface_material_state_at_projection(
              material(), a.fault_index, a.segment_index, a.xi);
          const double g = h.energetic_degradation(surface.first, phi[j]);
          const double old_g = h.energetic_degradation(
              surface.first, step == 0 ? phi[j] : old_phi[j]);
          const double old_h = F::normalization_integrand(
              step == 0 ? phi[j] : old_phi[j], old_g);
          double candidate = old;
          if (step > 0) {
            const auto c = F::surface_maxwell_coefficients(
                material(), a.fault_index, a.segment_index, a.xi, dt);
            const auto interpolate = [&](const auto &v) {
              return (1 - a.xi) * v[a.segment_index] +
                     a.xi * v[a.segment_index + 1];
            };
            candidate = F::compute_crack_driving_force_candidate(
                dt, c.first, c.second, g, old_h, interpolate(q),
                interpolate(previous_q));
            const double expected = std::max(old, candidate);
            candidate_error = std::max(candidate_error, std::abs(expected - H));
            AssertThrow(std::abs(expected - H) <=
                            128 * std::numeric_limits<double>::epsilon() *
                                std::max(1.0, std::abs(expected)),
                        ExcMessage("Native mechanical candidate does not "
                                   "explain committed H."));
          } else
            AssertThrow(H == old, ExcMessage("Timestep zero evolved H."));
          AssertThrow(H >= old, ExcMessage("Mechanical H decreased."));
          if (step == 0)
            initial_phi[p.get_id()] = phi[j];
          const double distance = std::abs(p.get_location()[1] - .5);
          const auto index =
              std::min(15u, static_cast<unsigned int>(distance * 32));
          const double volume = pm().get_particle_domain_handler()
                                    .get_particle_domain(p.get_local_index())
                                    .volume();
          auto &bin = bins[index];
          bin.volume += volume;
          bin.H0 += volume * initial_H.at(p.get_id());
          bin.Hin += volume * old;
          bin.Hnew += volume * H;
          bin.candidate += volume * candidate;
          bin.cmax = std::max(bin.cmax, candidate);
          bin.phi0 += volume * initial_phi.at(p.get_id());
          bin.phi += volume * phi[j];
          bin.phimax = std::max(bin.phimax, phi[j]);
          bin.dHmax = std::max(bin.dHmax, H - initial_H.at(p.get_id()));
          const bool increased = H - old > 1e-10;
          bin.increases += increased;
          increases += increased;
          H_change += volume * (H - old);
          H_total += volume * (H - initial_H.at(p.get_id()));
          phase_volume_change += volume * (phi[j] - initial_phi.at(p.get_id()));
          if (profiles[0]->value(distance) <= 1e-6) {
            offband_increases += increased;
            offband_H = std::max(offband_H, H - initial_H.at(p.get_id()));
            offband_phi = std::max(offband_phi, phi[j]);
          }
          study_particle(p, phi[j], old_phi[j], old, H, candidate, volume);
          ++j;
        }
      }
    std::ofstream spatial(this->get_output_directory() + "feedback-bins-" +
                          std::to_string(step) + ".csv");
    spatial << std::setprecision(17)
            << "r_min,r_max,volume,H_initial,H_incoming,H_committed,candidate_"
               "mean,candidate_max,phi_initial,phi,phi_max,H_cumulative_max,"
               "count_increase\n";
    for (unsigned int i = 0; i < bins.size(); ++i) {
      const auto &z = bins[i];
      spatial << i / 32.0 << ',' << (i + 1) / 32.0 << ',' << z.volume << ','
              << z.H0 / z.volume << ',' << z.Hin / z.volume << ','
              << z.Hnew / z.volume << ',' << z.candidate / z.volume << ','
              << z.cmax << ',' << z.phi0 / z.volume << ',' << z.phi / z.volume
              << ',' << z.phimax << ',' << z.dHmax << ',' << z.increases
              << '\n';
    }
    std::ofstream faults(this->get_output_directory() + "feedback-fault-" +
                         std::to_string(step) + ".csv");
    faults << std::setprecision(17)
           << "x,q_incoming,q_committed,Ih_incoming,Ih_current,V\n";
    const auto &fault = manager.get_fault(0);
    const auto &velocity = manager.get_timestep_committed_slip_rate(0);
    for (unsigned int i = 0; i < q.size(); ++i)
      faults << fault.vertex(i)[0] << ',' << previous_q[i] << ',' << q[i] << ','
             << previous_I[i] << ',' << Ih[i] << ',' << velocity[i] << '\n';
    std::ofstream out(this->get_output_directory() + "feedback.csv",
                      step == 0 ? std::ios::out : std::ios::app);
    if (step == 0)
      out << "step,time,dt,phase_kkt,mechanical_residual,newton,krylov,min_"
             "alpha,trial_observations,phi_min,phi_max,phase_change,phase_"
             "healing,H_increment_integral,H_cumulative_integral,phase_"
             "cumulative_integral,H_increases,offband_H_increases,offband_H_"
             "cumulative_max,offband_phi_max,candidate_commit_error,Ih_min,Ih_"
             "max,q_min,q_max,V_max\n";
    out << std::setprecision(17) << step << ',' << time << ',' << dt << ','
        << phase_kkt << ',' << mechanical_residual << ',' << newton << ','
        << krylov << ',' << min_alpha << ',' << observations << ',' << min_phi
        << ',' << max_phi << ',' << change << ',' << healing << ',' << H_change
        << ',' << H_total << ',' << phase_volume_change << ',' << increases
        << ',' << offband_increases << ',' << offband_H << ',' << offband_phi
        << ',' << candidate_error << ','
        << *std::min_element(Ih.begin(), Ih.end()) << ','
        << *std::max_element(Ih.begin(), Ih.end()) << ','
        << *std::min_element(q.begin(), q.end()) << ','
        << *std::max_element(q.begin(), q.end()) << ','
        << *std::max_element(velocity.begin(), velocity.end()) << '\n';
    study_end();
    this->get_pcout() << "P2 feedback: t=" << time
                      << " H cumulative=" << H_total
                      << " phase cumulative=" << phase_volume_change
                      << " off-band phi=" << offband_phi
                      << " healing=" << healing << std::endl;
    return {"P2 native mechanical feedback", "observed"};
  }
};
ASPECT_REGISTER_POSTPROCESSOR(CoupledFeedback, "P2 coupled feedback",
                              "Observe native H feedback with fixed particles "
                              "and geometry; never prescribe H.")
} // namespace Postprocess
} // namespace aspect
