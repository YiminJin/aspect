/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/reconstructed_fault/cohesive_system_internal.h>
#include <aspect/reconstructed_fault/endpoint_mpi.h>
#include <deal.II/fe/fe_values.h>
#include <chrono>
#include <set>
#include <sstream>
#include <iomanip>

namespace aspect
{
  namespace CohesiveInterface
  {
    namespace
    {
      void
      sum_matrix (FullMatrix<double> &matrix, const MPI_Comm communicator)
      {
        std::vector<double> values (matrix.m () * matrix.n ());
        for (unsigned int i = 0; i < matrix.m (); ++i)
          for (unsigned int j = 0; j < matrix.n (); ++j)
            values[i * matrix.n () + j] = matrix (i, j);
        internal::endpoint_sum (values, communicator);
        for (unsigned int i = 0; i < matrix.m (); ++i)
          for (unsigned int j = 0; j < matrix.n (); ++j)
            matrix (i, j) = values[i * matrix.n () + j];
      }
      void
      sum_vector (Vector<double> &vector, const MPI_Comm communicator)
      {
        std::vector<double> values (vector.begin (), vector.end ());
        internal::endpoint_sum (values, communicator);
        std::copy (values.begin (), values.end (), vector.begin ());
      }
      // Only owned maps are written. Ghosted native fields are read inputs.
      void
      add_owned (System::Field &to, const System::Field &from, const double weight)
      {
        for (unsigned int b = 0; b < to.n_blocks (); ++b)
          for (const auto i : to.block (b).locally_owned_elements ())
            to.block (b)[i] += weight * from.block (b)[i];
      }
    }

    std::shared_ptr<System::Implementation::Prepared>
    System::Implementation::prepare (const State &s, Geometry geometry, const BulkSolve &inverse)
    {
      const auto begin = std::chrono::steady_clock::now ();
      auto p = std::make_shared<Prepared> ();
      p->geometry = std::move (geometry);
      p->points = p->geometry.surface_points ();
      const auto &g = p->geometry;
      const unsigned int n = g.modes (), nc = g.vertices ().size () - 1, nq = p->points.size ();
      std::ostringstream key;
      key << std::setprecision (17) << support << ';' << law.G << ';' << pressure_scaling << ';';
      for (const auto &piece : g.pieces ())
        key << piece.x << ';' << piece.y << ';' << piece.end << ';' << piece.a << ';' << piece.b
            << ';' << piece.c1 << ';' << piece.c2 << ';' << piece.c3 << ';' << piece.cells << ';'
            << piece.hermite << ';';
      if (cached_prepared && prepared_signature == key.str ())
        {
          *p = *cached_prepared;
          p->incoming_points = points (s, g);
          p->incoming_memory.assign (n, 0.);
          for (unsigned int j = 0; j < formed_cells (s) - 1; ++j)
            p->incoming_memory[j] = s.memory.at (j);
          ++cache_hits;
          return p;
        }
      auto &op = p->surface;
      op.law = law;
      op.affine_modes = 1;
      op.structural_q1_ties = true;
      op.K.reinit (n + 1, n + 1);
      op.Q.reinit (nq, n);
      op.Nm.reinit (nq, n + 1);
      op.Np.reinit (nq, n + 1);
      op.weights.resize (nq);
      op.mass.assign (n, 0.);
      p->normal_mass.assign (nc, 0.);
      for (unsigned int q = 0; q < nq; ++q)
        {
          const auto &point = p->points[q];
          op.weights[q] = point.weight;
          if (point.left > 0)
            op.Q (q, point.left - 1) = 1 - point.xi;
          if (point.right + 1 < g.vertices ().size ())
            op.Q (q, point.right - 1) = point.xi;
          for (unsigned int j = 0; j < n; ++j)
            op.mass[j] += point.weight * op.Q (q, j);
          p->normal_mass[point.cell] += point.weight * 4 * point.xi * (1 - point.xi);
        }
      p->incoming_points = points (s, g);
      p->incoming_memory.assign (n, 0.);
      const unsigned int persistent = formed_cells (s) - 1;
      for (unsigned int j = 0; j < persistent; ++j)
        p->incoming_memory[j] = s.memory.at (j);
      for (unsigned int j = persistent; j < s.memory.size (); ++j)
        AssertThrow (s.memory[j] == 0. && s.jump[j] == 0. && s.rate[j] == 0.,
                     ExcMessage ("Virgin candidate carries material memory."));
      p->columns.resize (n);
      p->normals.resize (nc);
      std::vector<std::string> keys (n), normal_keys (nc);
      std::vector<bool> new_column (n), new_normal (nc);
      for (unsigned int j = 0; j < n; ++j)
        {
          keys[j] = g.mode_signature (j);
          const auto found = cached_columns.find (keys[j]);
          if (found != cached_columns.end ())
            {
              p->columns[j] = found->second;
              ++cache_hits;
            }
          else
            {
              p->columns[j].load = owned ();
              *p->columns[j].load = 0.;
              new_column[j] = true;
            }
        }
      for (unsigned int j = 0; j < nc; ++j)
        {
          normal_keys[j] = std::to_string (j) + ':' + g.mode_signature (std::min (j, n - 1));
          const auto found = cached_normals.find (normal_keys[j]);
          if (found != cached_normals.end ())
            p->normals[j] = found->second;
          else
            {
              p->normals[j].load = owned ();
              *p->normals[j].load = 0.;
              new_normal[j] = true;
            }
        }
      FullMatrix<double> D (n, n), Dn (nc, n);
      Vector<double> normal_measure (nc);
      const auto &fe = system.get_fe ();
      const auto &intro = system.introspection ();
      const auto &constraints = system.get_current_constraints ();
      std::vector<unsigned int> local;
      for (unsigned int i = 0; i < fe.n_dofs_per_cell (); ++i)
        if (intro.is_stokes_component (fe.system_to_component_index (i).first))
          local.push_back (i);
      std::vector<types::global_dof_index> indices (fe.n_dofs_per_cell ()), ids (local.size ());
      for (const auto &cell : system.get_dof_handler ().active_cell_iterators ())
        if (cell->is_locally_owned ())
          {
            const auto rule = g.quadrature (cell->vertex (0), cell->vertex (3), bulk_order);
            FEValues<2> values (system.get_mapping (), fe, rule,
                                update_values | update_gradients | update_quadrature_points
                                    | update_JxW_values);
            values.reinit (cell);
            cell->get_dof_indices (indices);
            for (unsigned int i = 0; i < local.size (); ++i)
              ids[i] = indices[local[i]];
            std::map<unsigned int, Vector<double>> B, Bn;
            std::vector<Tensor> E (local.size ());
            std::vector<double> pressure (local.size ());
            for (unsigned int q = 0; q < rule.size (); ++q)
              {
                const auto x = values.quadrature_point (q);
                const auto modes = g.lifting (x);
                if (modes.empty () && !g.normal_active (x))
                  continue;
                const double w = values.JxW (q);
                for (unsigned int i = 0; i < local.size (); ++i)
                  {
                    E[i] = dev3 (
                        values[intro.extractors.velocities].symmetric_gradient (local[i], q));
                    pressure[i] = values[intro.extractors.pressure].value (local[i], q);
                  }
                for (const auto &m : modes)
                  {
                    const auto e = dev3 (m.gradient);
                    const double div = trace (m.gradient);
                    if (new_column[m.index])
                      {
                        auto &load = B[m.index];
                        if (load.size () == 0)
                          load.reinit (local.size ());
                        for (unsigned int i = 0; i < local.size (); ++i)
                          load[i]
                              += w
                                 * (2 * law.G * (E[i] * e) - pressure_scaling * pressure[i] * div);
                      }
                    for (const auto &other : modes)
                      D (m.index, other.index) += w * 2 * law.G * (e * dev3 (other.gradient));
                  }
                if (g.normal_active (x))
                  {
                    const auto normal = g.normal_test (x);
                    const auto en = dev3 (normal.gradient);
                    const double div = trace (normal.gradient);
                    normal_measure[normal.index] -= w * div;
                    if (new_normal[normal.index])
                      {
                        auto &load = Bn[normal.index];
                        if (load.size () == 0)
                          load.reinit (local.size ());
                        for (unsigned int i = 0; i < local.size (); ++i)
                          load[i]
                              += w
                                 * (2 * law.G * (E[i] * en) - pressure_scaling * pressure[i] * div);
                      }
                    for (const auto &m : modes)
                      Dn (normal.index, m.index) += w * 2 * law.G * (en * dev3 (m.gradient));
                  }
              }
            for (const auto &entry : B)
              constraints.distribute_local_to_global (entry.second, ids,
                                                      *p->columns[entry.first].load);
            for (const auto &entry : Bn)
              constraints.distribute_local_to_global (entry.second, ids,
                                                      *p->normals[entry.first].load);
          }
      sum_matrix (D, system.get_mpi_communicator ());
      sum_matrix (Dn, system.get_mpi_communicator ());
      sum_vector (normal_measure, system.get_mpi_communicator ());
      // The pressure multiplier is dual to the native bulk work measure.
      // Surface arclength quadrature alone is not its discrete normalization.
      for (unsigned int j = 0; j < nc; ++j)
        {
          AssertThrow (normal_measure[j] > 0., ExcMessage ("Invalid native normal work measure."));
          p->normal_mass[j] = normal_measure[j];
        }
      for (unsigned int j = 0; j < n; ++j)
        if (new_column[j])
          {
            p->columns[j].load->compress (VectorOperation::add);
            auto rhs = owned ();
            *rhs = *p->columns[j].load;
            *rhs *= -1.;
            p->columns[j].response = inverse (*rhs);
            ++bulk_solves;
            p->columns[j].scaled = scaled (*p->columns[j].response);
          }
      for (unsigned int j = 0; j < nc; ++j)
        if (new_normal[j])
          p->normals[j].load->compress (VectorOperation::add);
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
          op.K (i + 1, j + 1) = D (i, j) + (*p->columns[i].load) * (*p->columns[j].scaled);
      for (unsigned int i = 0; i < nc; ++i)
        for (unsigned int j = 0; j < n; ++j)
          {
            const double N
                = (Dn (i, j) + (*p->normals[i].load) * (*p->columns[j].scaled)) / p->normal_mass[i];
            for (unsigned int q = 0; q < 3; ++q)
              op.Nm (3 * i + q, j + 1) = op.Np (3 * i + q, j + 1) = N;
          }
      // Derived operator caches only; rejection may rebuild them. Accepted
      // tensors and formed records never depend on a retired candidate cache.
      cached_columns.clear ();
      cached_normals.clear ();
      for (unsigned int j = 0; j < n; ++j)
        cached_columns[keys[j]] = p->columns[j];
      for (unsigned int j = 0; j < nc; ++j)
        cached_normals[normal_keys[j]] = p->normals[j];
      cached_prepared = p;
      prepared_signature = key.str ();
      setup_seconds
          += std::chrono::duration<double> (std::chrono::steady_clock::now () - begin).count ();
      return p;
    }

    void
    System::Implementation::forcing (Prepared &p, const State &s, const Field &particular,
                                     const BulkSolve &inverse)
    {
      const unsigned int n = p.geometry.modes (), nc = p.normal_mass.size ();
      auto rhs = owned ();
      *rhs = 0.;
      p.particular = ghosted ();
      *p.particular = particular;
      Vector<double> shear (n), normal (nc);
      const auto &fe = system.get_fe ();
      const auto &intro = system.introspection ();
      const auto &constraints = system.get_current_constraints ();
      std::vector<unsigned int> local;
      for (unsigned int i = 0; i < fe.n_dofs_per_cell (); ++i)
        if (intro.is_stokes_component (fe.system_to_component_index (i).first))
          local.push_back (i);
      std::vector<types::global_dof_index> indices (fe.n_dofs_per_cell ()), ids (local.size ());
      double transfer = 0.;
      for (const auto &cell : system.get_dof_handler ().active_cell_iterators ())
        if (cell->is_locally_owned ())
          {
            const auto rule
                = p.geometry.quadrature (cell->vertex (0), cell->vertex (3), bulk_order);
            FEValues<2> values (system.get_mapping (), fe, rule,
                                update_values | update_gradients | update_quadrature_points
                                    | update_JxW_values);
            values.reinit (cell);
            cell->get_dof_indices (indices);
            for (unsigned int i = 0; i < local.size (); ++i)
              ids[i] = indices[local[i]];
            Vector<double> load (local.size ());
            const auto incoming = physical_stress (s, cell, rule);
            const auto smooth = s.bulk->evaluate (cell, rule);
            std::vector<SymmetricTensor<2, 2>> strain (rule.size ());
            std::vector<double> div (rule.size ());
            values[intro.extractors.velocities].get_function_symmetric_gradients (particular,
                                                                                  strain);
            values[intro.extractors.velocities].get_function_divergences (particular, div);
            for (unsigned int q = 0; q < rule.size (); ++q)
              {
                const auto x = values.quadrature_point (q);
                const auto modes = p.geometry.lifting (x);
                Tensor reconstructed = smooth[q];
                for (const auto &m : modes)
                  reconstructed += 2 * law.G * p.incoming_memory[m.index] * dev3 (m.gradient);
                transfer = std::max (transfer, (reconstructed - incoming[q]).norm ());
                const Tensor source = incoming[q] + 2 * law.G * dev3 (strain[q]);
                const double w = values.JxW (q);
                for (unsigned int i = 0; i < local.size (); ++i)
                  load[i]
                      += w
                         * (-source
                                * dev3 (values[intro.extractors.velocities].symmetric_gradient (
                                    local[i], q))
                            + pressure_scaling
                                  * values[intro.extractors.pressure].value (local[i], q) * div[q]);
                for (const auto &m : modes)
                  shear[m.index] += w * (source * dev3 (m.gradient));
                if (p.geometry.normal_active (x))
                  {
                    const auto normal_test = p.geometry.normal_test (x);
                    normal[normal_test.index] += w * (source * dev3 (normal_test.gradient));
                  }
              }
            constraints.distribute_local_to_global (load, ids, *rhs);
          }
      p.transfer_stress = Utilities::MPI::max (transfer, system.get_mpi_communicator ());
      AssertThrow (p.transfer_stress < 1e-7,
                   ExcMessage ("Candidate geometry changes an accepted physical Maxwell tensor."));
      rhs->compress (VectorOperation::add);
      sum_vector (shear, system.get_mpi_communicator ());
      sum_vector (normal, system.get_mpi_communicator ());
      p.correction = inverse (*rhs);
      ++bulk_solves;
      auto base = owned ();
      *base = 0.;
      add_owned (*base, particular, 1.);
      add_owned (*base, *p.correction, 1.);
      base->compress (VectorOperation::insert);
      p.base = ghosted ();
      *p.base = *base;
      const auto reference = scaled (*p.correction);
      const double dt = system.get_timestep (), beta = 1 / (1 + dt * law.G / law.eta);
      p.surface.force_offset.reinit (n);
      p.surface.normal_offset.reinit (p.points.size ());
      for (unsigned int j = 0; j < n; ++j)
        p.surface.force_offset[j] = beta * (shear[j] + (*p.columns[j].load) * (*reference));
      for (unsigned int j = 0; j < nc; ++j)
        {
          const double value
              = beta * (normal[j] + (*p.normals[j].load) * (*reference)) / p.normal_mass[j];
          for (unsigned int q = 0; q < 3; ++q)
            p.surface.normal_offset[3 * j + q] = value;
        }
    }
  }
}
