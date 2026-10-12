// Deliberate P1 test-only extension. No production solver or history-law
// replacement.
#include "../../../tests/phase_field_controlled_history.cc"
#include "../../../tests/phase_field_test_access.h"
#include "reference_1d.h"
#include <aspect/particle/particle_domain.h>
#include <fstream>
#include <iomanip>
#include <map>

namespace aspect
{
namespace Postprocess
{
template <int dim>
class P1BoundDiagnostic : public Interface<dim>, public SimulatorAccess<dim>
{
  using V = LinearAlgebra::BlockVector;
  using M = LinearAlgebra::BlockSparseMatrix;
  using A = internal::PhaseFieldTestAccess<dim>;
  bool reference_only = true;
  std::vector<double> fractions;
  struct Restore
  {
    V &live;
    V saved;
    std::vector<std::pair<double *, double>> H;
    explicit Restore (V &v) : live (v), saved (v) {}
    ~Restore ()
    {
      live = saved;
      for (auto &p : H)
        *p.first = p.second;
    }
  };
  std::string
  fingerprint () const
  {
    std::string s;
    const auto add = [&] (const auto &v)
      { s.append (reinterpret_cast<const char *> (&v), sizeof (v)); };
    for (const V *v : { &this->get_solution (), &this->get_old_solution (),
                        &this->get_system_rhs () })
      for (unsigned int b = 0; b < v->n_blocks (); ++b)
        for (auto i : v->block (b).locally_owned_elements ())
          add (v->block (b)[i]);
    const auto &pm
        = this->get_phase_field_handler ().get_associated_particle_manager ();
    for (const auto &p : pm.get_particle_handler ())
      {
        add (p.get_id ());
        for (unsigned int d = 0; d < dim; ++d)
          add (p.get_location ()[d]);
        for (double v : p.get_properties ())
          add (v);
      }
    add (pm.get_particle_domain_handler ().geometry_version ());
    for (const auto &f :
         this->get_reconstructed_fault_manager ().get_faults ())
      {
        add (f.geometry_version ());
        for (unsigned int i = 0; i < f.n_vertices (); ++i)
          {
            for (unsigned int d = 0; d < dim; ++d)
              add (f.vertex (i)[d]);
            for (double v : f.get_properties (i))
              add (v);
          }
      }
    return s;
  }
  struct Metrics
  {
    double minimum = 1e100, maximum = -1e100, free = 0, dual = 0,
           complementarity = 0, kkt = 0, change = 0, healing = 0;
    unsigned int lower = 0, upper = 0;
  };
  Metrics
  metrics (const V &u, const V &rhs, const V &previous) const
  {
    Metrics m;
    const auto b = this->introspection ().variable ("phase_field").block_index;
    for (auto i : u.block (b).locally_owned_elements ())
      {
        const double x = u.block (b)[i], r = rhs.block (b)[i];
        m.minimum = std::min (m.minimum, x);
        m.maximum = std::max (m.maximum, x);
        m.change = std::max (m.change, std::abs (x - previous.block (b)[i]));
        m.healing = std::max (m.healing, previous.block (b)[i] - x);
        double defect = r;
        if (x == 0)
          {
            ++m.lower;
            defect = std::max (r, 0.0);
            m.dual = std::max (m.dual, defect);
          }
        else if (x == 1)
          {
            ++m.upper;
            defect = std::min (r, 0.0);
            m.dual = std::max (m.dual, -defect);
          }
        else
          m.free = std::max (m.free, std::abs (r));
        // Multipliers from gradient=-rhs; report both contact products.
        m.complementarity = std::max (
            m.complementarity,
            std::max (x * std::max (-r, 0.0), (1 - x) * std::max (r, 0.0)));
        m.kkt = std::max (m.kkt, std::abs (defect));
      }
    return m;
  }
  double
  energy (const V &u) const
  {
    const auto &h = this->get_phase_field_handler ();
    const auto &pm = h.get_associated_particle_manager ();
    const auto &dofs = A::vertex_dofs (h);
    const auto hi = pm.get_property_manager ()
                        .get_data_info ()
                        .get_position_by_field_name ("crack_driving_force");
    double result = 0;
    for (const auto &p : pm.get_particle_handler ())
      {
        const auto d = pm.get_particle_domain_handler ().get_particle_domain (
            p.get_local_index ());
        double phi = 0;
        Tensor<1, dim> grad;
        for (unsigned int i = 0; i < d.n_relevant_vertices (); ++i)
          {
            const double x = u[dofs[d.relevant_vertex_index (i)]];
            phi += x * d.weighting_function_value (i);
            grad += x * d.weighting_function_gradient (i);
          }
        const auto c
            = A::coefficients (h, fractions, phi, p.get_properties ()[hi]);
        AssertThrow (c.valid, ExcInternalError ());
        result += d.volume ()
                  * (c.energy + 0.5 * c.diffusion * grad.norm_square ());
      }
    return result;
  }
  unsigned int
  bounded (M &matrix, V &rhs, V &u, double &target) const
  {
    auto &handler = this->get_phase_field_handler ();
    const auto b = this->introspection ().variable ("phase_field").block_index;
    V owned (rhs), update (rhs), trial (u);
    owned = u;
    for (auto i : owned.block (b).locally_owned_elements ())
      owned.block (b)[i] = std::clamp (double (owned.block (b)[i]), 0.0, 1.0);
    u = owned;
    for (unsigned int it = 0; it < 100; ++it)
      {
        double roundoff = 0;
        AssertThrow (A::assemble (handler, matrix, rhs, u, true, &roundoff),
                     ExcInternalError ());
        const auto m = metrics (u, rhs, u);
        target = std::max (1e-11, roundoff);
        if (m.kkt <= target)
          return it;
        std::vector<bool> active (u.block (b).size (), false);
        for (auto i : u.block (b).locally_owned_elements ())
          active[i] = (u.block (b)[i] == 0 && rhs.block (b)[i] <= 0)
                      || (u.block (b)[i] == 1 && rhs.block (b)[i] >= 0);
        const V raw_rhs (rhs);
        // Symmetric free-subspace Newton system, same assembled tangent.
        for (auto i : rhs.block (b).locally_owned_elements ())
          {
            if (active[i])
              {
                matrix.block (b, b).clear_row (i, 1);
                rhs.block (b)[i] = 0;
              }
            else
              for (auto entry = matrix.block (b, b).begin (i);
                   entry != matrix.block (b, b).end (i); ++entry)
                if (active[entry->column ()])
                  entry->value () = 0;
          }
        matrix.block (b, b).compress (VectorOperation::insert);
        update = 0;
        A::solve (handler, matrix, rhs, update);
        const double old_energy = energy (u);
        bool accepted = false;
        for (unsigned int ls = 0; ls < 35; ++ls)
          {
            const double alpha = std::ldexp (1.0, -int (ls));
            owned = u;
            double slope = 0;
            for (auto i : owned.block (b).locally_owned_elements ())
              {
                const double x = std::clamp (double (u.block (b)[i])
                                                 + alpha * update.block (b)[i],
                                             0.0, 1.0);
                owned.block (b)[i] = x;
                slope -= raw_rhs.block (b)[i] * (x - u.block (b)[i]);
              }
            trial = owned;
            AssertThrow (A::assemble (handler, matrix, rhs, trial, false),
                         ExcInternalError ());
            const double new_kkt = metrics (trial, rhs, trial).kkt;
            if (new_kkt < m.kkt * (1 - 1e-4 * alpha)
                || (slope < 0 && energy (trial) < old_energy + 1e-4 * slope))
              {
                u = trial;
                accepted = true;
                break;
              }
          }
        AssertThrow (accepted,
                     ExcMessage ("P1 bound Newton line search stalled."));
      }
    AssertThrow (false, ExcMessage ("P1 bound Newton budget exhausted."));
    return 100;
  }
  void
  reference (M &matrix, V &rhs, V &state,
             const std::function<double (double)> &profile,
             const double core) const
  {
    auto &h = this->get_phase_field_handler ();
    const auto &pm = h.get_associated_particle_manager ();
    const auto hi = pm.get_property_manager ()
                        .get_data_info ()
                        .get_position_by_field_name ("crack_driving_force");
    const auto &fault
        = this->get_reconstructed_fault_manager ().get_prescribed_faults ()[0];
    AssertThrow (
        fault.vertices[0][0] == 0 && fault.vertices[1][0] == 1,
        ExcMessage ("Reference requires through-going unit-box band."));
    std::map<double, double> sample_map;
    for (const auto &p : pm.get_particle_handler ())
      {
        const double y = p.get_location ()[1], H = p.get_properties ()[hi];
        if (sample_map.count (y))
          AssertThrow (std::abs (sample_map[y] - H)
                           < 1e-10 * std::max (1.0, H),
                       ExcInternalError ());
        sample_map[y] = H;
      }
    const unsigned int strips = sample_map.size (), cells = strips / 3;
    AssertThrow (strips == 3 * cells, ExcInternalError ());
    std::vector<double> samples, breaks;
    for (const auto &p : sample_map)
      samples.push_back (p.second);
    for (unsigned int i = 1; i < strips; ++i)
      breaks.push_back (double (i) / strips);
    const double baseline = samples.front ();
    const auto *model
        = dynamic_cast<const MaterialModel::PhaseFieldModel<dim> *> (
            &this->get_material_model ());
    const double threshold = model->get_phase_field_activation_threshold ();
    double left = 0, right = 0.5;
    for (unsigned int i = 0; i < 60; ++i)
      {
        const double y = 0.5 * (left + right);
        if (profile (y) > threshold)
          right = y;
        else
          left = y;
      }
    const double transition = 0.5 * (left + right);
    const std::vector<double> edges = { transition, 1 - transition };
    const std::function<double (double)> native_H = [&] (double y)
      {
        const double p = profile (y);
        return p > threshold
                   ? h.stationary_crack_driving_force (fractions, p, core)
                   : baseline;
      };
    const std::function<double (double)> sampled_H = [&] (double y)
      { return samples[std::min<unsigned int> (strips - 1, y * strips)]; };
    for (const auto &p : sample_map)
      AssertThrow (std::abs (native_H (p.first) - p.second)
                       < 1e-9 * std::max (1.0, p.second),
                   ExcMessage ("Reference H differs from native particle H."));
    const auto coefficient = [&] (double p, double H)
      { return A::coefficients (h, fractions, p, H); };
    const auto c = coefficient (0, baseline);
    this->get_pcout () << std::setprecision (17)
                       << "P1 reference cells=" << cells
                       << " transition=" << transition
                       << " baseline_H=" << baseline
                       << " tail_diffusion=" << c.diffusion
                       << " tail_tangent=" << c.tangent
                       << " L_tail=" << std::sqrt (c.diffusion / c.tangent)
                       << std::endl;
    std::ofstream summary (this->get_output_directory ()
                           + "reference-summary.csv");
    summary << std::setprecision (17)
            << "name,nodes,minimum,residual,correction,iterations,cpdi_node_"
               "difference\n";
    const auto &dofs = A::vertex_dofs (h);
    const auto &vertices = this->get_triangulation ().get_vertices ();
    std::vector<std::pair<std::string, P1Reference::Result>> results;
    const auto run = [&] (std::string name, unsigned int n, unsigned int q,
                          bool sampled, bool midpoint)
      {
        this->get_pcout () << "P1 reference solve " << name << std::endl;
        auto r = P1Reference::solve (n, q, sampled ? breaks : edges,
                                     sampled ? sampled_H : native_H, profile,
                                     coefficient, midpoint);
        double difference = 0;
        for (unsigned int v = 0; v < dofs.size (); ++v)
          if (dofs[v] != numbers::invalid_dof_index
              && std::abs (vertices[v][0] - .5) < 1e-12)
            difference
                = std::max (difference, std::abs (state[dofs[v]]
                                                  - r.value (vertices[v][1])));
        summary << name << ',' << r.x.size () << ','
                << *std::min_element (r.phi.begin (), r.phi.end ()) << ','
                << r.residual << ',' << r.correction << ',' << r.iterations
                << ',' << difference << '\n';
        summary.flush ();
        results.emplace_back (name, std::move (r));
      };
    run ("cpdi_reduction", cells, 1, true, true);
    run ("coarse_sampled_gauss8", cells, 8, true, false);
    run ("coarse_native_gauss8", cells, 8, false, false);
    run ("fine_sampled_4096", 4096, 4, true, false);
    run ("fine_sampled_8192", 8192, 8, true, false);
    run ("fine_native_4096", 4096, 4, false, false);
    run ("fine_native_8192", 8192, 8, false, false);
    std::ofstream cuts (this->get_output_directory () + "reference-cuts.csv");
    cuts << std::setprecision (17) << "y,profile,H_native,H_sampled,cpdi";
    for (const auto &r : results)
      cuts << ',' << r.first;
    cuts << '\n';
    for (unsigned int i = 0; i <= 8192; ++i)
      {
        const double y = double (i) / 8192;
        // CPDI reduction is separately compared against every actual
        // center-line node.
        cuts << y << ',' << profile (y) << ',' << native_H (y) << ','
             << sampled_H (y) << ',' << results[0].second.value (y);
        for (const auto &r : results)
          cuts << ',' << r.second.value (y);
        cuts << '\n';
      }
    (void)matrix;
    (void)rhs;
  }

public:
  static void
  declare_parameters (ParameterHandler &prm)
  {
    prm.enter_subsection ("Postprocess");
    prm.enter_subsection ("P1 bound diagnostic");
    prm.declare_entry ("Reference only", "true", Patterns::Bool ());
    prm.leave_subsection ();
    prm.leave_subsection ();
  }
  void
  parse_parameters (ParameterHandler &prm) override
  {
    prm.enter_subsection ("Postprocess");
    prm.enter_subsection ("P1 bound diagnostic");
    reference_only = prm.get_bool ("Reference only");
    prm.leave_subsection ();
    prm.leave_subsection ();
  }
  std::pair<std::string, std::string>
  execute (TableHandler &) override
  {
    AssertThrow (
        dim == 2
            && Utilities::MPI::n_mpi_processes (this->get_mpi_communicator ())
                   == 1
            && this->get_timestep_number () == 0,
        ExcMessage ("Bounded P1 is serial initialization only."));
    auto &h = this->get_phase_field_handler ();
    auto &pm = h.get_associated_particle_manager ();
    auto &particles = pm.get_particle_handler ();
    const auto hi = pm.get_property_manager ()
                        .get_data_info ()
                        .get_position_by_field_name ("crack_driving_force");
    const auto &fields
        = this->introspection ().chemical_composition_field_indices ();
    AssertThrow (fields.size () == 1, ExcInternalError ());
    const auto map
        = this->get_parameters ().mapped_particle_properties.at (fields[0]);
    const auto ci = pm.get_property_manager ()
                        .get_data_info ()
                        .get_position_by_field_name (map.first)
                    + map.second;
    const double composition = particles.begin ()->get_properties ()[ci];
    fractions
        = MaterialModel::MaterialUtilities::compute_composition_fractions (
            std::vector<double>{ composition });
    for (const auto &p : particles)
      AssertThrow (p.get_properties ()[ci] == composition,
                   ExcMessage ("P1 reference requires uniform composition."));
    const auto b = this->introspection ().variable ("phase_field").block_index;
    const auto &dofs = A::vertex_dofs (h);
    const auto &vertices = this->get_triangulation ().get_vertices ();
    const auto &faults
        = this->get_reconstructed_fault_manager ().get_prescribed_faults ();
    AssertThrow (faults.size () == 1, ExcInternalError ());
    const auto &fault = faults[0];
    const double core = fault.core_phase_field_values[0];
    auto profiles = h.get_phase_field_profiles (core);
    const std::function<double (double)> profile
        = [&] (double y) { return profiles[0]->value (std::abs (y - 0.5)); };
    V rhs (this->introspection ().index_sets.system_partitioning,
           this->get_mpi_communicator ()),
        owned (rhs), state (this->get_solution ());
    M matrix;
    matrix.reinit (this->get_system_matrix ().n_block_rows (),
                   this->get_system_matrix ().n_block_cols ());
    for (unsigned int r = 0; r < matrix.n_block_rows (); ++r)
      for (unsigned int c = 0; c < matrix.n_block_cols (); ++c)
        matrix.block (r, c).copy_from (
            this->get_system_matrix ().block (r, c));
    matrix.collect_sizes ();
    const std::string before = fingerprint ();
    auto &live = const_cast<V &> (this->get_solution ());
    {
      Restore restore (live);
      for (auto &p : particles)
        restore.H.emplace_back (&p.get_properties ()[hi],
                                p.get_properties ()[hi]);
      // Verify assembly on supplied scratch state with poisoned live phase.
      double clean_allowance = 0, poisoned_allowance = 0;
      AssertThrow (A::assemble (h, matrix, rhs, state, true, &clean_allowance),
                   ExcInternalError ());
      const V clean_rhs (rhs);
      const auto entries = [&] ()
        {
          std::vector<double> values;
          for (auto row : rhs.block (b).locally_owned_elements ())
            for (auto entry = matrix.block (b, b).begin (row);
                 entry != matrix.block (b, b).end (row); ++entry)
              values.push_back (entry->value ());
          return values;
        };
      const auto clean_matrix = entries ();
      owned = live;
      owned.block (b) = 1.25;
      live = owned;
      AssertThrow (
          A::assemble (h, matrix, rhs, state, true, &poisoned_allowance),
          ExcInternalError ());
      V difference (rhs);
      difference -= clean_rhs;
      AssertThrow (difference.block (b).l2_norm () == 0
                       && entries () == clean_matrix
                       && clean_allowance == poisoned_allowance,
                   ExcMessage ("Bound diagnostic assembly read live phi."));
      double entry_allowance = 0;
      A::assemble (h, matrix, rhs, state, false, &entry_allowance);
      const double equality_target = std::max (
          entry_allowance, A::parameters (h).nonlinear_solver_tolerance
                               * rhs.block (b).l2_norm ());
      h.evolve_phase_field (matrix, rhs, state);
      A::assemble (h, matrix, rhs, state, true);
      AssertThrow (rhs.block (b).l2_norm () <= equality_target,
                   ExcMessage ("Reference equality residual failed."));
      this->get_pcout () << std::setprecision (17)
                         << "P1 equality residual=" << rhs.block (b).l2_norm ()
                         << " target=" << equality_target << std::endl;
      if (reference_only)
        reference (matrix, rhs, state, profile, core);
      else
        {
          V equality (state), equality_previous (state);
          double comparison_target = equality_target,
                 comparison_residual = rhs.block (b).l2_norm ();
          // The rounded native profile is the initial guess, not a pinned
          // seed.
          owned = state;
          for (unsigned int v = 0; v < dofs.size (); ++v)
            if (dofs[v] != numbers::invalid_dof_index)
              {
                const double distance
                    = ReconstructedFaultUtilities::
                          closest_point_distance_and_core_phase_field (
                              fault, vertices[v])
                              .first;
                const double p = profiles[0]->value (distance);
                for (const auto &material_profile : profiles)
                  AssertThrow (material_profile->value (distance) == p,
                               ExcInternalError ());
                owned[dofs[v]] = p;
              }
          state = owned;
          const V rounded (state);
          double target = 0;
          std::ofstream csv (this->get_output_directory () + "bounds.csv");
          csv << std::setprecision (17)
              << "stage,iterations,target,node_min,node_max,free_residual,"
                 "dual_violation,complementarity,kkt,lower,upper,change,"
                 "healing,H_increment_min,energy,equality_difference,"
                 "correction,cpdi_min,cpdi_max,equality_min,equality_healing,"
                 "equality_residual,equality_target,weight_min,weight_sum_"
                 "defect,cpdi_roundoff\n";
          const auto record
              = [&] (const std::string &stage, unsigned int iterations,
                     const V &old, double H_increment)
            {
              AssertThrow (A::assemble (h, matrix, rhs, state, true),
                           ExcInternalError ());
              const auto m = metrics (state, rhs, old);
              AssertThrow (m.minimum >= 0 && m.maximum <= 1 && m.free <= target
                               && m.dual <= target
                               && m.complementarity <= target,
                           ExcMessage ("P1 box KKT check failed."));
              double diff = 0;
              for (auto i : state.block (b).locally_owned_elements ())
                diff = std::max (diff, std::abs (state.block (b)[i]
                                                 - equality.block (b)[i]));
              double cpdi_min = 1e100, cpdi_max = -1e100;
              double weight_min = 1e100, weight_sum_defect = 0,
                     cpdi_roundoff = 0;
              for (const auto &p : particles)
                {
                  const auto d
                      = pm.get_particle_domain_handler ().get_particle_domain (
                          p.get_local_index ());
                  double phi = 0, sum = 0, negative_weight = 0,
                         absolute_weight = 0;
                  for (unsigned int i = 0; i < d.n_relevant_vertices (); ++i)
                    {
                      const double w = d.weighting_function_value (i);
                      weight_min = std::min (weight_min, w);
                      negative_weight += std::max (-w, 0.0);
                      absolute_weight += std::abs (w);
                      sum += w;
                      phi += state[dofs[d.relevant_vertex_index (i)]] * w;
                    }
                  const double rounding
                      = 16 * std::numeric_limits<double>::epsilon ()
                        * d.n_relevant_vertices ()
                        * std::max (1.0, absolute_weight);
                  // Only roundoff-sized deviations from a convex partition are
                  // allowed. Do not clip weights or the evaluated phase field.
                  AssertThrow (
                      negative_weight <= rounding
                          && std::abs (sum - 1) <= rounding,
                      ExcMessage (
                          "P1 CPDI weights significantly violate convexity."));
                  weight_sum_defect
                      = std::max (weight_sum_defect, std::abs (sum - 1));
                  cpdi_roundoff = std::max (
                      cpdi_roundoff,
                      negative_weight + std::abs (sum - 1) + rounding);
                  cpdi_min = std::min (cpdi_min, phi);
                  cpdi_max = std::max (cpdi_max, phi);
                }
              AssertThrow (cpdi_min >= -cpdi_roundoff
                               && cpdi_max <= 1 + cpdi_roundoff,
                           ExcMessage ("P1 CPDI bounds failed."));
              double equality_minimum = 1e100, equality_healing = 0;
              for (auto i : equality.block (b).locally_owned_elements ())
                {
                  equality_minimum = std::min (equality_minimum,
                                               double (equality.block (b)[i]));
                  equality_healing = std::max (equality_healing,
                                               equality_previous.block (b)[i]
                                                   - equality.block (b)[i]);
                }
              // Directional KKT correction estimate, allowing contact nodes
              // with a wrong multiplier sign to move off the bound.
              std::vector<bool> active (state.block (b).size (), false);
              for (auto i : rhs.block (b).locally_owned_elements ())
                active[i]
                    = (state.block (b)[i] == 0 && rhs.block (b)[i] <= 0)
                      || (state.block (b)[i] == 1 && rhs.block (b)[i] >= 0);
              for (auto i : rhs.block (b).locally_owned_elements ())
                if (active[i])
                  {
                    matrix.block (b, b).clear_row (i, 1);
                    rhs.block (b)[i] = 0;
                  }
                else
                  for (auto entry = matrix.block (b, b).begin (i);
                       entry != matrix.block (b, b).end (i); ++entry)
                    if (active[entry->column ()])
                      entry->value () = 0;
              matrix.block (b, b).compress (VectorOperation::insert);
              V correction (rhs);
              correction = 0;
              if (rhs.block (b).l2_norm () > 0)
                A::solve (h, matrix, rhs, correction);
              csv << stage << ',' << iterations << ',' << target << ','
                  << m.minimum << ',' << m.maximum << ',' << m.free << ','
                  << m.dual << ',' << m.complementarity << ',' << m.kkt << ','
                  << m.lower << ',' << m.upper << ',' << m.change << ','
                  << m.healing << ',' << H_increment << ',' << energy (state)
                  << ',' << diff << ',' << correction.block (b).linfty_norm ()
                  << ',' << cpdi_min << ',' << cpdi_max << ','
                  << equality_minimum << ',' << equality_healing << ','
                  << comparison_residual << ',' << comparison_target << ','
                  << weight_min << ',' << weight_sum_defect << ','
                  << cpdi_roundoff << '\n';
              csv.flush ();
              this->get_pcout ()
                  << "P1 bounds " << stage << " iterations=" << iterations
                  << " kkt=" << m.kkt << " min=" << m.minimum
                  << " healing=" << m.healing << std::endl;
            };
          const unsigned int relax_iterations
              = bounded (matrix, rhs, state, target);
          record ("relax_commit", relax_iterations, rounded, 0);
          // Accept this equilibrium as the diagnostic committed phase
          // baseline.
          const V relaxed (state);
          V committed (state);
          std::vector<bool> mask;
          for (const auto &p : particles)
            {
              const auto d
                  = pm.get_particle_domain_handler ().get_particle_domain (
                      p.get_local_index ());
              double phi = 0;
              for (unsigned int i = 0; i < d.n_relevant_vertices (); ++i)
                phi += state[dofs[d.relevant_vertex_index (i)]]
                       * d.weighting_function_value (i);
              mask.push_back (phi > 0.1 && phi < 0.7);
            }
          const std::vector<std::string> labels
              = { "hold",           "hold_repeat", "ramp_05",
                  "ramp_05_repeat", "ramp_10",     "unload" };
          const std::vector<double> factors
              = { .9, .9, 1.05, 1.05, 1.10, 1.0 };
          for (unsigned int k = 0; k < labels.size (); ++k)
            {
              double H_increment = 1e100;
              for (unsigned int i = 0; i < restore.H.size (); ++i)
                {
                  auto &p = restore.H[i];
                  const double old = *p.first;
                  if (mask[i])
                    *p.first = std::max (old, factors[k] * p.second);
                  H_increment = std::min (H_increment, *p.first - old);
                }
              AssertThrow (H_increment >= 0, ExcInternalError ());
              state = committed;
              if (labels[k] == "ramp_05_repeat")
                {
                  // Solver starting-guess perturbation only: H stays
                  // identical.
                  owned = state;
                  for (auto i : owned.block (b).locally_owned_elements ())
                    owned.block (b)[i]
                        = std::clamp (double (state.block (b)[i])
                                          + 1e-6 * std::sin (double (i) + 1),
                                      0.0, 1.0);
                  state = owned;
                }
              equality_previous = equality;
              double allowance = 0;
              AssertThrow (
                  A::assemble (h, matrix, rhs, equality, false, &allowance),
                  ExcInternalError ());
              comparison_target = std::max (
                  allowance, A::parameters (h).nonlinear_solver_tolerance
                                 * rhs.block (b).l2_norm ());
              h.evolve_phase_field (matrix, rhs, equality);
              AssertThrow (A::assemble (h, matrix, rhs, equality, true),
                           ExcInternalError ());
              comparison_residual = rhs.block (b).l2_norm ();
              AssertThrow (comparison_residual <= comparison_target,
                           ExcMessage ("Matched-H equality solve failed."));
              const unsigned int iterations
                  = bounded (matrix, rhs, state, target);
              record (labels[k], iterations, committed, H_increment);
              committed = state;
            }
          std::ofstream cut (this->get_output_directory ()
                             + "bound-nodes.csv");
          cut << std::setprecision (17)
              << "x,y,rounded,relaxed,equality_final,bounded_final\n";
          for (unsigned int v = 0; v < dofs.size (); ++v)
            if (dofs[v] != numbers::invalid_dof_index
                && (std::abs (vertices[v][0] - .5) < 1e-12
                    || std::abs (vertices[v][1] - .5) < 1e-12))
              cut << vertices[v][0] << ',' << vertices[v][1] << ','
                  << rounded[dofs[v]] << ',' << relaxed[dofs[v]] << ','
                  << equality[dofs[v]] << ',' << state[dofs[v]] << '\n';
        }
    }
    AssertThrow (fingerprint () == before,
                 ExcMessage ("P1 normal restoration failed."));
    bool threw = false;
    try
      {
        Restore restore (live);
        for (auto &p : particles)
          {
            restore.H.emplace_back (&p.get_properties ()[hi],
                                    p.get_properties ()[hi]);
            p.get_properties ()[hi] *= 1.05;
          }
        owned = live;
        owned.block (b) = .3;
        live = owned;
        owned.block (b) = 1.25;
        state = owned;
        h.evolve_phase_field (matrix, rhs, state);
      }
    catch (const std::exception &e)
      {
        threw = std::string (e.what ()).find ("outside the nonsingular")
                != std::string::npos;
      }
    AssertThrow (threw && fingerprint () == before,
                 ExcMessage ("P1 exceptional restoration failed."));
    this->get_pcout () << "P1 supplied scratch state and normal/exception "
                          "restoration verified\n";
    return { "P1 diagnostic:",
             reference_only
                 ? "reference complete"
                 : "box constraints verified; healing assessed separately" };
  }
};
ASPECT_REGISTER_POSTPROCESSOR (
    P1BoundDiagnostic, "P1 bound diagnostic",
    "Test-only box-constrained phase operator and one-dimensional reference.")
}
}
