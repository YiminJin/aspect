/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/reconstructed_fault/cohesive_evolution.h>
#include <set>
namespace aspect
{
  namespace CohesiveEvolution
  {
    std::pair<Vector, std::vector<bool>>
    continuation_predictor (const Operators &op, const Incoming &old, const double dt,
                            const Vector &rate)
    {
      const unsigned n = op.Q.n (), nq = op.Q.m ();
      AssertDimension (rate.size (), n);
      const double beta = 1 / (1 + op.law.G / op.law.eta * dt);
      auto evaluate = [&] (const Vector &v, const std::vector<bool> &ties, Vector &force, Matrix &J)
        {
          force = op.force_offset;
          J.reinit (n, n);
          for (unsigned i = 0; i < n; ++i)
            for (unsigned j = 0; j < n; ++j)
              {
                force[i] += beta * op.K (i + 1, j + 1) * v[j];
                J (i, j) = beta * op.K (i + 1, j + 1);
              }
          for (unsigned q = 0; q < nq; ++q)
            {
              double inc = 0., pn = op.law.p0 + op.normal_offset[q];
              for (unsigned j = 0; j < n; ++j)
                {
                  inc += op.Q (q, j) * v[j];
                  pn += beta * op.Nm (q, j + 1) * v[j];
                }
              if (ties[q])
                inc = 0.;
              const auto r = CohesiveInterface::evaluate (op.law, old.history[q], inc, pn, dt,
                                                          inc == 0. ? 0. : std::copysign (1., inc));
              if (ties[q])
                continue;
              for (unsigned i = 0; i < n; ++i)
                {
                  const double w = op.weights[q] * op.Q (q, i);
                  force[i] += w * r.resistance;
                  for (unsigned j = 0; j < n; ++j)
                    J (i, j) += w
                                * (r.derivative * op.Q (q, j)
                                   + r.pressure_derivative * beta * op.Nm (q, j + 1));
                }
            }
        };
      std::vector<bool> initial (nq);
      for (unsigned q = 0; q < nq; ++q)
        initial[q] = old.history[q].damage == 0.;
      auto proposal = [&] (const std::vector<bool> &ties)
        {
          Matrix basis;
          const unsigned nf = structural_q1_basis (op.Q, ties, basis);
          Matrix C (n, nf);
          for (unsigned i = 0; i < n; ++i)
            for (unsigned j = 0; j < nf; ++j)
              C (i, j) = basis (i, j);
          Vector inc (n);
          for (unsigned k = 0; k < nf; ++k)
            {
              double a = 0.;
              for (unsigned j = 0; j < n; ++j)
                a += C (j, k) * rate[j] * dt;
              for (unsigned j = 0; j < n; ++j)
                inc[j] += C (j, k) * a;
            }
          if (!nf)
            return inc;
          const double tolerance = 1e-9 * *std::min_element (op.mass.begin (), op.mass.end ());
          for (unsigned it = 0; it < 30; ++it)
            {
              Vector force, f (nf);
              Matrix J;
              evaluate (inc, ties, force, J);
              C.Tvmult (f, force);
              if (f.linfty_norm () <= tolerance)
                break;
              Matrix JC (n, nf), A (nf, nf);
              J.mmult (JC, C);
              C.Tmmult (A, JC);
              f *= -1.;
              Vector z = equilibrated_step (A, f), step (n);
              C.vmult (step, z);
              bool found = false;
              double scale = 1.;
              for (unsigned ls = 0; ls < 30; ++ls)
                {
                  Vector v (inc), ff (nf), F;
                  Matrix unused;
                  v.add (scale, step);
                  try
                    {
                      evaluate (v, ties, F, unused);
                      C.Tvmult (ff, F);
                    }
                  catch (const std::exception &)
                    {
                      scale *= .5;
                      continue;
                    }
                  if (ff.l2_norm () < f.l2_norm () || ff.linfty_norm () <= tolerance)
                    {
                      inc = v;
                      found = true;
                      break;
                    }
                  scale *= .5;
                }
              if (!found)
                break;
            }
          return inc;
        };
      Vector best = proposal (initial), force, lambda;
      Matrix J;
      evaluate (best, initial, force, J);
      unsigned iterations = 0;
      double error = certificate (op, force, initial, lambda, iterations);
      if (error <= 1e-8)
        return { best, initial };
      std::vector<unsigned> tied;
      for (unsigned q = 0; q < nq; ++q)
        if (initial[q])
          tied.push_back (q);
      if (tied.empty ())
        return { best, initial };
      std::set<unsigned> cells{ tied.front () / 3 };
      std::sort (tied.begin (), tied.end (), [&] (unsigned a, unsigned b)
                   { return std::abs (lambda[a]) > std::abs (lambda[b]); });
      for (unsigned k = 0;
           k < std::min (3u, static_cast<unsigned> (tied.size ())) && cells.size () < 2; ++k)
        cells.insert (tied[k] / 3);
      auto chosen = initial;
      double distance = std::numeric_limits<double>::infinity ();
      for (unsigned cell : cells)
        for (unsigned mask = 1; mask < 8; ++mask)
          {
            auto ties = initial;
            for (unsigned j = 0; j < 3; ++j)
              if (mask & (1u << j))
                ties[3 * cell + j] = false;
            try
              {
                auto v = proposal (ties);
                evaluate (v, ties, force, J);
                error = certificate (op, force, ties, lambda, iterations);
                if (error > 1e-8)
                  continue;
                bool released = true;
                for (unsigned q = 0; q < nq; ++q)
                  if (initial[q] && !ties[q])
                    {
                      double inc = 0.;
                      for (unsigned j = 0; j < n; ++j)
                        inc += op.Q (q, j) * v[j];
                      released &= inc != 0.;
                    }
                if (!released)
                  continue;
                double d = 0.;
                for (unsigned j = 0; j < n; ++j)
                  d += op.mass[j] * std::pow (v[j] - dt * rate[j], 2);
                if (d < distance)
                  {
                    distance = d;
                    best = v;
                    chosen = ties;
                  }
              }
            catch (const std::exception &)
              { /* Full solver diagnoses an unsuccessful proposal. */
              }
          }
      return { best, chosen };
    }
  }
}
