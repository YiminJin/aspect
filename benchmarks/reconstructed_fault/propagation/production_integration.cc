// P1 observer and prescribed-H schedule; all trajectory solves use the
// simulator.
#include "../../../tests/phase_field_controlled_history.cc"
#include "../../../tests/phase_field_fault_test_access.h"
#include <aspect/particle/manager.h>
#include <aspect/particle/particle_domain.h>
#include <aspect/plugins.h>
#include <aspect/postprocess/interface.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/simulator_signals.h>
#include <aspect/utilities.h>
#include <fstream>
#include <iomanip>

namespace aspect {
namespace Postprocess {
template <int dim>
class ProductionIntegration : public Interface<dim>,
                              public SimulatorAccess<dim> {
  using V = LinearAlgebra::BlockVector;
  using M = LinearAlgebra::BlockSparseMatrix;
  using A = internal::PhaseFieldTestAccess<dim>;
  using F = MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
  std::map<types::particle_index, std::pair<double, bool>> baseline;
  std::map<types::particle_index,double> initialized_H;
  std::vector<std::vector<double>> restored;
  std::vector<double> initial_properties;
  unsigned int b() const {
    return this->introspection().variable("phase_field").block_index;
  }
  auto &particles() const {
    return this->get_phase_field_handler()
        .get_associated_particle_manager()
        .get_particle_handler();
  }
  unsigned int hi() const {
    return this->get_phase_field_handler()
        .get_associated_particle_manager()
        .get_property_manager()
        .get_data_info()
        .get_position_by_field_name("crack_driving_force");
  }
  auto &material() const {
    return const_cast<MaterialModel::PhaseFieldFault<dim> &>(
        Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
            this->get_material_model()));
  }
  std::vector<double> properties() const {
    std::vector<double> result;
    for (const auto &f : this->get_reconstructed_fault_manager().get_faults())
      for (unsigned int v = 0; v < f.n_vertices(); ++v)
        for (double p : f.get_properties(v)) {
          result.push_back(std::isfinite(p) ? 1 : 0);
          result.push_back(std::isfinite(p) ? p : 0);
        }
    return result;
  }
  std::vector<double> fingerprint() const {
    std::vector<double> result = properties();
    for (const V *v : {&this->get_solution(), &this->get_old_solution(),
                       &this->get_old_old_solution()})
      for (unsigned int j = 0; j < v->n_blocks(); ++j)
        for (auto i : v->block(j).locally_owned_elements()) {
          result.push_back(i);
          result.push_back(v->block(j)[i]);
        }
    for (const auto &p : particles()) {
      result.push_back(p.get_id());
      for (unsigned int d = 0; d < dim; ++d)
        result.push_back(p.get_location()[d]);
      for (double x : p.get_properties())
        result.push_back(x);
    }
    return result;
  }
  double cpdi(const V &state,
              const typename Particles::ParticleHandler<dim>::particle_iterator
                  &p) const {
    const auto &h = this->get_phase_field_handler();
    const auto domain = h.get_associated_particle_manager()
                            .get_particle_domain_handler()
                            .get_particle_domain(p->get_local_index());
    double value = 0;
    for (unsigned int i = 0; i < domain.n_relevant_vertices(); ++i)
      value += state[A::vertex_dofs(h)[domain.relevant_vertex_index(i)]] *
               domain.weighting_function_value(i);
    return value;
  }
  double difference_norm(const V &a, const V &other,
                         const bool phase_only = true) const {
    V owned(this->introspection().index_sets.system_partitioning,
            this->get_mpi_communicator());
    V right(owned);
    owned = a;
    right = other;
    owned -= right;
    return phase_only ? owned.block(b()).linfty_norm() : owned.l2_norm();
  }
  void start() {
    if (!restored.empty()) {
      const auto rank =
          Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
      AssertThrow(
          fingerprint() == restored.at(rank),
          ExcMessage(
              "P1 restart changed accepted vectors/particles/properties."));
      AssertThrow(difference_norm(this->get_old_solution(),
                                  this->get_old_old_solution()) > 1e-4,
                  ExcMessage("Checkpoint erased phase-history distinction."));
      this->get_pcout()
          << "P1 restart: all stored histories exact and distinct\n";
      restored.clear();
    }
    const auto step = this->get_timestep_number();
    if (step == 0)
      for (const auto &p : particles()) initialized_H.emplace(p.get_id(),p.get_properties()[hi()]);
    if (step == 2 || step == 4)
      for (auto &p : particles()) {
        const auto &entry = baseline.at(p.get_id());
        if (entry.second)
          p.get_properties()[hi()] =
              std::max(p.get_properties()[hi()],
                       entry.first * (step == 2 ? 1.05 : 1.10));
      }
  }
  void matrix_copy(M &m) const {
    const auto &src = this->get_system_matrix();
    m.reinit(src.n_block_rows(), src.n_block_cols());
    for (unsigned int r = 0; r < m.n_block_rows(); ++r)
      for (unsigned int c = 0; c < m.n_block_cols(); ++c)
        m.block(r, c).copy_from(src.block(r, c));
    m.collect_sizes();
  }
  void scratch_checks(M &m, V &rhs) {
    const auto original_state=fingerprint();
    const auto original_domains=this->get_phase_field_handler().get_associated_particle_manager()
      .get_particle_domain_handler().geometry_version();
    auto &h = this->get_phase_field_handler();
    const V explicit_state(this->get_solution());
    std::vector<V *> targets = {
        &const_cast<V &>(this->get_solution()),
        &const_cast<V &>(this->get_current_linearization_point()),
        &const_cast<V &>(this->get_old_solution()),
        &const_cast<V &>(this->get_old_old_solution())};
    std::vector<V> saved;
    for (auto *v : targets)
      saved.push_back(*v);
    const auto entries = [&]() {
      std::vector<double> a;
      for (auto i : rhs.block(b()).locally_owned_elements())
        for (auto e = m.block(b(), b()).begin(i); e != m.block(b(), b()).end(i);
             ++e)
          a.push_back(e->value());
      return a;
    };
    double rounding = 0, energy = 0;
    A::assemble(h, m, rhs, explicit_state, true, &rounding, &energy);
    const V clean_rhs(rhs);
    const auto clean_matrix = entries();
    const auto restore = [&]() {
      for (unsigned int i = 0; i < targets.size(); ++i)
        *targets[i] = saved[i];
    };
    try {
      for (unsigned int k = 0; k < targets.size(); ++k) {
        V owned(rhs);
        owned = *targets[k];
        for (unsigned int j = 0; j < owned.n_blocks(); ++j)
          owned.block(j) = 1.25 + k + j;
        *targets[k] = owned;
      }
      double r2 = 0, e2 = 0;
      A::assemble(h, m, rhs, explicit_state, true, &r2, &e2);
      V diff(rhs);
      diff -= clean_rhs;
      AssertThrow(
          diff.block(b()).l2_norm() == 0 && entries() == clean_matrix &&
              rounding == r2 && energy == e2,
          ExcMessage(
              "Phase assembly read unrelated live/predictor/history state."));
      A::assemble(h, m, rhs, explicit_state, false, nullptr, &e2);
      diff = rhs;
      diff -= clean_rhs;
      AssertThrow(diff.block(b()).l2_norm() == 0 && energy == e2,
                  ExcMessage("Trial residual/energy differs."));
      restore();
    } catch (...) {
      restore();
      throw;
    }
    try {
      for (auto *v : targets) {
        V owned(rhs);
        owned = *v;
        owned.block(b()) = 1.25;
        *v = owned;
      }
      throw std::runtime_error("restoration probe");
    } catch (const std::runtime_error &) {
      restore();
    }
    for (unsigned int k = 0; k < targets.size(); ++k) {
      AssertThrow(difference_norm(*targets[k], saved[k], false) == 0,
                  ExcInternalError());
    }
    AssertThrow(fingerprint()==original_state && original_domains==this->get_phase_field_handler()
      .get_associated_particle_manager().get_particle_domain_handler().geometry_version(),
      ExcMessage("Scratch check changed particles, H, domains or committed properties."));
    this->get_pcout() << "P1 scratch: residual/J/energy/roundoff independent; "
                         "normal/exception restoration exact\n";
  }
  void initialization_failure_checks(M &matrix, V &rhs) {
    auto &h = this->get_phase_field_handler();
    const V accepted(this->get_solution());
    auto &parameters =
        const_cast<PhaseField::SolverParameters &>(A::parameters(h));
    const auto saved_iterations = parameters.max_nonlinear_iterations;
    parameters.max_nonlinear_iterations = 1;
    V candidate(accepted);
    bool threw = false;
    try {
      h.evolve_phase_field(matrix, rhs, candidate);
    } catch (const std::exception &) {
      threw = true;
    }
    parameters.max_nonlinear_iterations = saved_iterations;
    AssertThrow(threw && difference_norm(candidate, accepted) == 0,
                ExcMessage("Failed phase solve published an iterate."));

    auto &manager = this->get_reconstructed_fault_manager();
    std::vector<std::vector<std::vector<double>>> saved;
    for (const auto &f : manager.get_faults()) {
      saved.emplace_back();
      for (unsigned int v = 0; v < f.n_vertices(); ++v)
        saved.back().emplace_back(f.get_properties(v).begin(),
                                  f.get_properties(v).end());
    }
    const auto &info = manager.get_property_information();
    const auto q =
        info[manager.get_property_index("phase field fault cohesive traction")]
            .position;
    const auto ih =
        info[manager.get_property_index("phase field fault previous I h")]
            .position;
    for (unsigned int f = 0; f < saved.size(); ++f)
      for (unsigned int v = 0; v < saved[f].size(); ++v) {
        auto values = manager.get_fault(f).get_properties(v);
        values[q] = values[ih] = numbers::signaling_nan<double>();
      }
    const auto uninitialized = properties();
    auto &live = const_cast<V &>(this->get_solution());
    V owned(rhs);
    owned = live;
    owned.block(b()) = 1;
    live = owned;
    F::invalidate_normalization_cache(material());
    threw = false;
    try {
      material().prepare_reconstructed_fault_phase_state();
    } catch (const std::exception &) {
      threw = true;
    }
    const bool unchanged = properties() == uninitialized;
    live = accepted;
    for (unsigned int f = 0; f < saved.size(); ++f)
      for (unsigned int v = 0; v < saved[f].size(); ++v) {
        auto values = manager.get_fault(f).get_properties(v);
        std::copy(saved[f][v].begin(), saved[f][v].end(), values.begin());
      }
    F::invalidate_normalization_cache(material());
    F::compute_normalization_integrals(material());
    AssertThrow(
        threw && unchanged,
        ExcMessage(
            "Failed dependent initialization partially committed history."));
    this->get_pcout() << "P1 initialization failure: phase unpublished, paired "
                         "cohesive history unpublished; restored"
                      << std::endl;
  }
  void support_check() const {
    const auto &h = this->get_phase_field_handler();
    auto &manager = this->get_reconstructed_fault_manager();
    const auto &fault = manager.get_prescribed_faults().at(0);
    const auto profiles =
        h.get_phase_field_profiles(fault.core_phase_field_values.at(0));
    double width = 0;
    for (const auto &profile : profiles)
      width = std::max(width, profile->get_coordinate_values().back());
    const double x0 = fault.vertices.front()[0], x1 = fault.vertices.back()[0],
                 y0 = fault.vertices.front()[1];
    Point<dim> inside, outside;
    inside[0] = outside[0] = 0.5 * (x0 + x1);
    inside[1] = y0 + width * (1 - 1e-6);
    outside[1] = y0 + width * (1 + 1e-6);
    Point<dim> boundary(inside);
    boundary[1] = 1;
    const bool full = manager.project_to_normal_profiles(boundary).active;
    if (full) {
      for (double y : {0.0, 1.0, -1e-15, 1.0+1e-15}) {
        boundary[1] = y;
        AssertThrow(manager.project_to_normal_profiles(boundary).active,
                    ExcMessage("Full cross-section misses a boundary sample."));
      }
    } else
      AssertThrow(manager.project_to_normal_profiles(inside).active &&
                      !manager.project_to_normal_profiles(outside).active,
                  ExcMessage("Unexpected retained support width."));
    manager.prepare_stokes_qp_projection_cache();
    const auto builds = manager.get_stokes_qp_cache_diagnostics().rebuild_count;
    manager.prepare_stokes_qp_projection_cache();
    AssertThrow(builds == manager.get_stokes_qp_cache_diagnostics().rebuild_count,
                ExcMessage("Unchanged association cache rebuilt."));
    FEValues<dim> fe(this->get_mapping(), this->get_fe(),
                     this->introspection().quadratures.velocities,
                     update_values | update_quadrature_points |
                         update_JxW_values);
    const FEValuesExtractors::Scalar phase(
        this->introspection().variable("phase_field").first_component_index);
    double missed_qp = 0, missed_particle = 0;
    double whole_source = 0, omitted_source = 0;
    const auto fractions =
        F::surface_material_fractions_at_vertex(material(), manager, 0, 0);
    const auto &normalization = F::current_normalization_integrals(material());
    unsigned int diffuse_cap_qp = 0;
    std::vector<double> values(fe.n_quadrature_points);
    for (const auto &cell : this->get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned()) {
        fe.reinit(cell);
        fe[phase].get_function_values(this->get_solution(), values);
        const auto &associations = manager.get_stokes_qp_fault_associations(
          cell->id(), this->introspection().quadratures.velocities, fe.get_quadrature_points());
        for (unsigned int q = 0; q < values.size(); ++q) {
          const auto &point = fe.quadrature_point(q);
          AssertThrow(associations[q].active == manager.project_to_normal_profiles(point).active,
                      ExcMessage("Mechanical cache and normal association disagree."));
          if (full)
            AssertThrow(associations[q].active, ExcMessage("Full support misses a mechanical QP."));
          if (point[0] >= x0 && point[0] <= x1) {
            Point<dim> axis(point);
            axis[1] = y0;
            const auto foot = manager.project_to_normal_profiles(axis);
            AssertThrow(foot.active, ExcInternalError());
            const auto &I = normalization[foot.fault_index];
            const double Ih = (1 - foot.xi) * I[foot.segment_index] +
                              foot.xi * I[foot.segment_index + 1];
            const double localization =
                F::normalization_integrand(
                    values[q], h.energetic_degradation(fractions, values[q])) /
                Ih;
            const double contribution = localization * fe.JxW(q);
            whole_source += contribution;
            if (!manager.project_to_normal_profiles(point).active)
              omitted_source += contribution;
          }

          if (!manager.project_to_normal_profiles(point).active) {
            if (point[0] >= x0 && point[0] <= x1)
              missed_qp = std::max(missed_qp, values[q]);
            else if (values[q] > 1e-6)
              ++diffuse_cap_qp;
          }
        }
      }
    const auto &particle_data =
        manager.get_locally_owned_particle_fault_associations();
    unsigned int index = 0;
    for (auto p = particles().begin(); p != particles().end(); ++p, ++index) {
      if (full)
        AssertThrow(particle_data[index].active, ExcMessage("Full support misses a particle."));
      if (p->get_location()[0] >= x0 && p->get_location()[0] <= x1 &&
          !particle_data[index].active)
        missed_particle =
            std::max(missed_particle, cpdi(this->get_solution(), p));
    }
    missed_qp = Utilities::MPI::max(missed_qp, this->get_mpi_communicator());
    missed_particle =
        Utilities::MPI::max(missed_particle, this->get_mpi_communicator());
    diffuse_cap_qp =
        Utilities::MPI::sum(diffuse_cap_qp, this->get_mpi_communicator());
    whole_source =
        Utilities::MPI::sum(whole_source, this->get_mpi_communicator());
    omitted_source =
        Utilities::MPI::sum(omitted_source, this->get_mpi_communicator());
    this->get_pcout() << std::setprecision(17)
                      << "P1 support: full_cross_section=" << full << " prescribed_width=" << width
                      << " missed_strip_qp_phi=" << missed_qp
                      << " missed_strip_particle_phi=" << missed_particle
                      << " cap_qp_outside_sharp_tip=" << diffuse_cap_qp
                      << " normalized_source_total=" << whole_source
                      << " omitted_source=" << omitted_source
                      << " omitted_fraction=" << omitted_source/whole_source << std::endl;
    if (full)
      AssertThrow(omitted_source == 0, ExcMessage("Full association omitted a mechanical source."));
    AssertThrow(missed_qp <= 1e-6 && missed_particle <= 1e-6,
                ExcMessage("Retained support misses active normal-strip "
                           "integration samples."));
  }
  void contact_check(M &matrix, V &rhs) {
    auto &h = this->get_phase_field_handler();
    V owned(rhs), probe(this->get_solution());
    owned = probe;
    for (auto i : rhs.block(b()).locally_owned_elements()) {
      owned.block(b())[i] = (i % 3 == 0 ? 0 : (i % 3 == 1 ? 1 : 0.5));
      rhs.block(b())[i] = (i % 3 == 0 ? -1 : 1);
    }
    probe = owned;
    M original;
    matrix_copy(original);
    A::assemble(h, original, rhs, this->get_solution(), true);
    for (auto i : rhs.block(b()).locally_owned_elements())
      rhs.block(b())[i] = (i % 3 == 0 ? -1 : 1);
    matrix.block(b(), b()).copy_from(original.block(b(), b()));
    A::constrain(h, matrix, rhs, probe);
    unsigned int remote = 0;
    for (auto i : rhs.block(b()).locally_owned_elements())
      for (auto e = original.block(b(), b()).begin(i);
           e != original.block(b(), b()).end(i); ++e) {
        const auto j = e->column();
        const bool ai = i % 3 != 2, aj = j % 3 != 2;
        const double original_value = e->value();
        const double expected =
            ai ? (i == j ? 1.0 : 0.0) : (aj ? 0.0 : original_value);
        AssertThrow(matrix.block(b(), b()).el(i, j) == expected,
                    ExcMessage("Contact symmetric elimination failed: row=" +
                               std::to_string(i) +
                               " column=" + std::to_string(j) + " actual=" +
                               std::to_string(matrix.block(b(), b()).el(i, j)) +
                               " expected=" + std::to_string(expected) +
                               " phi=" + std::to_string(probe.block(b())[i])));
        if (!ai && aj &&
            !rhs.block(b()).locally_owned_elements().is_element(j) &&
            e->value() != 0)
          ++remote;
      }
    // Reverse both contact signs: both endpoints must release, without
    // clipping.
    matrix.block(b(), b()).copy_from(original.block(b(), b()));
    for (auto i : rhs.block(b()).locally_owned_elements())
      rhs.block(b())[i] = (i % 3 == 0 ? 1 : -1);
    A::constrain(h, matrix, rhs, probe);
    for (auto i : rhs.block(b()).locally_owned_elements())
      for (auto e = original.block(b(), b()).begin(i);
           e != original.block(b(), b()).end(i); ++e)
        AssertThrow(matrix.block(b(), b()).el(i, e->column()) == e->value(),
                    ExcMessage("Contact release failed."));
    remote = Utilities::MPI::sum(remote, this->get_mpi_communicator());
    if (Utilities::MPI::n_mpi_processes(this->get_mpi_communicator()) > 1)
      AssertThrow(
          remote > 0,
          ExcMessage("MPI contact test exercised no off-rank columns."));
    this->get_pcout() << "P1 contact signs/release/symmetric elimination: "
                         "passed, remote columns="
                      << remote << std::endl;
  }
  double correction_and_repeat(M &matrix, V &rhs) {
    auto &h = this->get_phase_field_handler();
    const auto &state = this->get_solution();
    A::assemble(h, matrix, rhs, state, true);
    A::constrain(h, matrix, rhs, state);
    V correction(rhs);
    correction = 0;
    A::solve(h, matrix, rhs, correction);
    const double estimate = correction.block(b()).linfty_norm();
    V owned(rhs), repeat(state);
    owned = state;
    const auto &dofs = A::vertex_dofs(h);
    const auto &vertices = this->get_triangulation().get_vertices();
    for (unsigned int v = 0; v < dofs.size(); ++v)
      if (dofs[v] != numbers::invalid_dof_index &&
          owned.locally_owned_elements().is_element(dofs[v]))
        owned[dofs[v]] =
            std::clamp(state[dofs[v]] + 1e-6 * std::sin(23 * vertices[v][0] +
                                                        5 * vertices[v][1]),
                       0.0, 1.0);
    repeat = owned;
    A::bounded(h, matrix, rhs, repeat);
    const double difference = difference_norm(repeat, state);
    const double allowance =
        std::max(4 * estimate, 64 * std::numeric_limits<double>::epsilon());
    AssertThrow(difference <= 1.298e-11,
                ExcMessage("Identical-H repeat exceeds prior allowance."));
    this->get_pcout() << std::setprecision(17)
                      << "P1 phase uncertainty: correction=" << estimate
                      << " directional_allowance=" << allowance
                      << " identical_H_repeat=" << difference << std::endl;
    return estimate;
  }

public:
  void initialize() override {
    this->get_signals().start_timestep.connect(
        [this](const SimulatorAccess<dim> &) { start(); });
  }
  std::pair<std::string, std::string> execute(TableHandler &) override {
    const auto step = this->get_timestep_number(),
               rank = Utilities::MPI::this_mpi_process(
                   this->get_mpi_communicator());
    const auto comm = this->get_mpi_communicator();
    const auto &state = this->get_solution();
    auto &h = this->get_phase_field_handler();
    M matrix;
    matrix_copy(matrix);
    V rhs(this->introspection().index_sets.system_partitioning, comm);
    this->get_pcout() << "P1 observer step=" << step << std::endl;
    if (step == 0) {
      std::ofstream initial_H(this->get_output_directory()+"initial-H-"+std::to_string(rank)+".csv");
      initial_H << std::setprecision(17) << "id,x,y,H,native_H\n";
      for (const auto &p : particles()) {
        const double expected = initialized_H.at(p.get_id());
        AssertThrow(expected == p.get_properties()[hi()],
                    ExcMessage("Relaxation/association selection changed initialized native H."));
        initial_H << p.get_id() << ',' << p.get_location()[0] << ',' << p.get_location()[1] << ','
                  << p.get_properties()[hi()] << ',' << expected << '\n';
      }
      this->get_pcout() << "P1 initial H: native pre-phase/pre-association snapshot preserved exactly" << std::endl;
      scratch_checks(matrix, rhs);
      contact_check(matrix, rhs);
      initialization_failure_checks(matrix, rhs);
      for (auto p = particles().begin(); p != particles().end(); ++p) {
        const double phi = cpdi(state, p);
        baseline[p->get_id()] = {p->get_properties()[hi()],
                                 phi > 0.1 && phi < 0.7};
      }
      initial_properties = properties();
    }
    AssertThrow(
        difference_norm(state, this->get_current_linearization_point()) == 0,
        ExcMessage("Accepted phase not published to predictor."));
    AssertThrow(properties() == initial_properties,
                ExcMessage("Phase-only trajectory advanced cohesive history."));
    double rounding = 0, energy = 0;
    A::assemble(h, matrix, rhs, state, true, &rounding, &energy);
    double minimum = 1, maximum = 0, free = 0, dual = 0, complementarity = 0,
           change = 0, healing = 0, cpmin = 1, cpmax = 0;
    for (auto i : rhs.block(b()).locally_owned_elements()) {
      const double x = state.block(b())[i], r = rhs.block(b())[i];
      minimum = std::min(minimum, x);
      maximum = std::max(maximum, x);
      if (x == 0)
        dual = std::max(dual, std::max(r, 0.0));
      else if (x == 1)
        dual = std::max(dual, std::max(-r, 0.0));
      else
        free = std::max(free, std::abs(r));
      complementarity =
          std::max(complementarity,
                   std::max(x * std::max(-r, 0.0), (1 - x) * std::max(r, 0.0)));
      if (step > 0) {
        change = std::max(change,
                          std::abs(x - this->get_old_solution().block(b())[i]));
        healing = std::max(healing, this->get_old_solution().block(b())[i] - x);
      }
    }
    for (auto p = particles().begin(); p != particles().end(); ++p) {
      double x = cpdi(state, p);
      cpmin = std::min(cpmin, x);
      cpmax = std::max(cpmax, x);
    }
    minimum = Utilities::MPI::min(minimum, comm);
    maximum = Utilities::MPI::max(maximum, comm);
    free = Utilities::MPI::max(free, comm);
    dual = Utilities::MPI::max(dual, comm);
    complementarity = Utilities::MPI::max(complementarity, comm);
    change = Utilities::MPI::max(change, comm);
    healing = Utilities::MPI::max(healing, comm);
    cpmin = Utilities::MPI::min(cpmin, comm);
    cpmax = Utilities::MPI::max(cpmax, comm);
    AssertThrow(minimum >= 0 && maximum <= 1 && cpmin >= -3e-14 &&
                    cpmax <= 1 + 3e-14,
                ExcMessage("P1 raw phase bounds failed."));
    AssertThrow(std::max({free, dual, complementarity}) <= 1e-11,
                ExcMessage("P1 box equilibrium gate failed."));
    AssertThrow(healing <= 1.298e-11,
                ExcMessage("P1 resolved incremental healing."));
    if (step == 2 || step == 4)
      AssertThrow(change > 1e-4, ExcMessage("P1 unresolved phase response."));
    if (step == 1 || step == 3)
      AssertThrow(change <= 1.298e-11, ExcMessage("P1 hold drift."));
    if (step == 2)
      correction_and_repeat(matrix, rhs);
    support_check();
    const auto ih = F::current_normalization_integrals(material());
    // Test the actual I_h-only projection on distributed profile owners.
    const auto profiles = F::normalization_profiles(material());
    std::vector<double> constant(profiles.size(), 7.0), samples;
    double local_integral = 0;
    for (const auto &profile : profiles) {
      AssertThrow(profile.xi >= 0 && profile.xi <= 1 && profile.surface_weight > 0,
                  ExcMessage("Nonpositive normalization quadrature/basis."));
      samples.push_back(.1 + std::exp(-1000 * profile.origin[0] * profile.origin[0]));
      local_integral += profile.surface_weight * samples.back();
    }
    const double integral = Utilities::MPI::sum(local_integral, comm);
    const auto &fault = this->get_reconstructed_fault_manager().get_fault(0);
    double constant_error = 0, conservation_error = 0, minimum_lumped = 1e100;
    for (const bool lumped : {false, true}) {
      const auto c = F::probe_normalization_projection(material(), constant, lumped);
      const auto values = F::probe_normalization_projection(material(), samples, lumped);
      double represented = 0;
      for (unsigned int i = 0; i < values.size(); ++i) {
        constant_error = std::max(constant_error, std::abs(c[i]-7));
        const double measure = .5 * ((i > 0 ? fault.vertex(i).distance(fault.vertex(i-1)) : 0)
                       + (i+1 < values.size() ? fault.vertex(i+1).distance(fault.vertex(i)) : 0));
        AssertThrow(measure > 0, ExcMessage("Nonpositive surface nodal measure."));
        represented += measure * values[i];
        if (lumped) minimum_lumped = std::min(minimum_lumped, values[i]);
      }
      conservation_error = std::max(conservation_error, std::abs(represented-integral));
    }
    AssertThrow(constant_error < 2e-14 && conservation_error < 2e-14 && minimum_lumped > 0
                && F::current_normalization_integrals(material()) == ih,
                ExcMessage("I_h projection constant/conservation/restoration check failed."));
    if (rank == 0) {
      std::ofstream check(this->get_output_directory() + "projection-check-" + std::to_string(step) + ".csv");
      check << std::setprecision(17) << "constant_error,conservation_error,minimum_lumped\n"
            << constant_error << ',' << conservation_error << ',' << minimum_lumped << '\n';
    }
    // Frozen P2 coarse-field samples exercise the native projector at both
    // authorized fault spacings. These read-only data do not replace live I_h.
    if (step == 0 && this->get_reconstructed_fault_manager().has_property(
                          "row sum lumped I h representation") &&
        fault.vertex(0)[0] == 0 && fault.vertex(fault.n_vertices()-1)[0] == 1) {
      const auto filename = Utilities::expand_ASPECT_SOURCE_DIR(
          "$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/propagation/normalization_evidence/frozen-profiles-" +
          std::to_string(fault.n_cells()) + ".csv");
      std::ifstream input(filename);
      AssertThrow(input, ExcMessage("Missing frozen normalization fixture: " + filename));
      std::string line;
      std::getline(input, line);
      std::vector<double> x, data;
      while (std::getline(input, line)) {
        const auto comma = line.find(',');
        x.push_back(std::stod(line.substr(0, comma)));
        data.push_back(std::stod(line.substr(comma+1)));
      }
      std::vector<double> owned_samples;
      for (const auto &profile : profiles) {
        AssertThrow(profile.id < data.size() &&
                    std::abs(profile.origin[0]-x[profile.id]) < 3e-15,
                    ExcMessage("Frozen normalization profile identity mismatch."));
        owned_samples.push_back(data[profile.id]);
      }
      const auto lumped = F::probe_normalization_projection(material(), owned_samples, true);
      const auto consistent = F::probe_normalization_projection(material(), owned_samples, false);
      if (rank == 0) {
        std::ofstream out(this->get_output_directory() + "frozen-native-projection.csv");
        out << std::setprecision(17) << "x,lumped,consistent\n";
        for (unsigned int i=0; i<lumped.size(); ++i)
          out << fault.vertex(i)[0] << ',' << lumped[i] << ',' << consistent[i] << '\n';
      }
      AssertThrow(F::current_normalization_integrals(material()) == ih,
                  ExcMessage("Frozen projection diagnostic changed live normalization."));
    }
    const auto warm = F::compute_normalization_integrals(material());
    AssertThrow(ih == warm, ExcMessage("Warm I_h changed."));
    F::invalidate_normalization_cache(material());
    const auto cold = F::compute_normalization_integrals(material());
    double ihdiff = 0, ihmin = 1e100, ihmax = 0;
    for (unsigned int f = 0; f < ih.size(); ++f)
      for (unsigned int v = 0; v < ih[f].size(); ++v) {
        ihdiff = std::max(ihdiff, std::abs(ih[f][v] - cold[f][v]));
        ihmin = std::min(ihmin, ih[f][v]);
        ihmax = std::max(ihmax, ih[f][v]);
      }
    AssertThrow(ihdiff == 0 && ihmin > 0 && std::isfinite(ihmax),
                ExcMessage("Cold normalization mismatch."));
    const auto suffix =
        (this->get_parameters().resume_computation ? "resume-" : "") +
        std::to_string(step);
    std::ofstream nodes(this->get_output_directory() + "phase-" + suffix + "-" +
                        std::to_string(rank) + ".csv");
    nodes << std::setprecision(17) << "x,y,phi,old,older\n";
    const auto &dofs = A::vertex_dofs(h);
    const auto &vertices = this->get_triangulation().get_vertices();
    for (unsigned int v = 0; v < dofs.size(); ++v)
      if (dofs[v] != numbers::invalid_dof_index &&
          state.locally_owned_elements().is_element(dofs[v]))
        nodes << vertices[v][0] << ',' << vertices[v][1] << ','
              << state[dofs[v]] << ',' << this->get_old_solution()[dofs[v]]
              << ',' << this->get_old_old_solution()[dofs[v]] << '\n';
    if (rank == 0) {
      std::ofstream out(this->get_output_directory() + "metrics-" + suffix +
                        ".csv");
      out << std::setprecision(17)
          << "step,time,min,max,free,dual,complementarity,change,healing,cpdi_"
             "min,cpdi_max,roundoff,energy,Ih_min,Ih_max,Ih_cold_difference\n"
          << step << ',' << this->get_time() << ',' << minimum << ',' << maximum
          << ',' << free << ',' << dual << ',' << complementarity << ','
          << change << ',' << healing << ',' << cpmin << ',' << cpmax << ','
          << rounding << ',' << energy << ',' << ihmin << ',' << ihmax << ','
          << ihdiff << '\n';
    }
    return {"P1 production phase gate:", "passed"};
  }
  void save(std::map<std::string, std::string> &strings) const override {
    AssertThrow(this->get_timestep_number() == 3 &&
                    difference_norm(this->get_old_solution(),
                                    this->get_old_old_solution()) > 1e-4,
                ExcMessage("Wrong checkpoint stage or equal histories."));
    const auto states =
        Utilities::MPI::all_gather(this->get_mpi_communicator(), fingerprint());
    const auto maps =
        Utilities::MPI::all_gather(this->get_mpi_communicator(), baseline);
    std::ostringstream buffer;
    {
      aspect::oarchive archive(buffer);
      archive << states << maps << initial_properties;
    }
    strings["P1Integration"] = buffer.str();
  }
  void load(const std::map<std::string, std::string> &strings) override {
    std::vector<decltype(baseline)> maps;
    std::istringstream buffer(strings.at("P1Integration"));
    {
      aspect::iarchive archive(buffer);
      archive >> restored >> maps >> initial_properties;
    }
    baseline =
        maps.at(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()));
  }
};
ASPECT_REGISTER_POSTPROCESSOR(
    ProductionIntegration, "P1 production integration",
    "Controlled H through the production phase lifecycle.")
} // namespace Postprocess
} // namespace aspect
