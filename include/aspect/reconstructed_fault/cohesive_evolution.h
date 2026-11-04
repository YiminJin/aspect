/* Copyright (C) 2026 by the authors of ASPECT. GPL v2 or later. */
#ifndef aspect_cohesive_evolution_h
#define aspect_cohesive_evolution_h
// Stateless signed solve, shared by the native owner and retained references.
// The native mixed solve may supply independent force/normal offsets from
// general tensor history. Those offsets are loads, never physical histories.
#include <aspect/reconstructed_fault/cohesive_interface.h>
#include <aspect/utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <cstdlib>

namespace aspect
{
  namespace CohesiveEvolution
  {
    DeclException1(ExcStepFailure, std::string, << arg1);
    constexpr double weak_tolerance = 1e-8;
    using Matrix = dealii::FullMatrix<double>;
    using Vector = dealii::Vector<double>;
    using namespace dealii;
    struct Operators
    {
      Matrix K, Q, Nm, Np;
      CohesiveInterface::Parameters law;
      Vector force_offset, normal_offset;
      unsigned affine_modes = 1;
      bool structural_q1_ties = false;
      std::vector<double> weights, mass;
    };
    struct Incoming
    {
      double affine = 0.;
      std::vector<double> affine_memory;
      std::vector<double> memory;
      std::vector<CohesiveInterface::PointState> history;
    };
    struct Result
    {
      Vector increment, coefficients, lambda;
      Matrix tangent, basis;
      std::vector<CohesiveInterface::PointTrial> point;
      std::vector<bool> tied;
      unsigned free = 0, iterations = 0, certificate_iterations = 0;
      double weak = 0., curvature = 0., contact = 1e100;
    };
    // Accepted-active-set Newton predictor. This is a search strategy; solve()
    // still certifies the full signed weak law and admissible current tangent.
    std::pair<Vector, std::vector<bool>> continuation_predictor (const Operators &,
                                                                 const Incoming &, double dt,
                                                                 const Vector &accepted_rate);
    inline Vector
    least_squares (const Matrix &A, const Vector &b)
    {
      LAPACKFullMatrix<double> svd (A.m (), A.n ());
      svd = A;
      svd.compute_svd ();
      const auto &U = svd.get_svd_u (), &V = svd.get_svd_vt ();
      Vector x (A.n ());
      const double cutoff = std::numeric_limits<double>::epsilon () * std::max (A.m (), A.n ())
                            * svd.singular_value (0);
      for (unsigned k = 0; k < std::min (A.m (), A.n ()); ++k)
        if (svd.singular_value (k) > cutoff)
          {
            double p = 0.;
            for (unsigned i = 0; i < A.m (); ++i)
              p += U (i, k) * b[i];
            p /= svd.singular_value (k);
            for (unsigned j = 0; j < A.n (); ++j)
              x[j] += V (k, j) * p;
          }
      return x;
    }
    // Algebraic equilibration only: D_r A D_c z=D_r b, step=D_c z.
    // RSF's regularized zero-rate tangent and intact constraints differ by many
    // orders of magnitude. Scale both before SVD without changing the residual.
    inline Vector
    equilibrated_step (const Matrix &A, const Vector &b)
    {
      Matrix scaled (A);
      Vector rhs (b), columns (A.n ());
      for (unsigned j = 0; j < A.n (); ++j)
        {
          double norm = 0.;
          for (unsigned i = 0; i < A.m (); ++i)
            norm = std::max (norm, std::abs (A (i, j)));
          columns[j] = norm > 0 ? 1. / norm : 1.;
          for (unsigned i = 0; i < A.m (); ++i)
            scaled (i, j) *= columns[j];
        }
      for (unsigned i = 0; i < A.m (); ++i)
        {
          double norm = 0.;
          for (unsigned j = 0; j < A.n (); ++j)
            norm = std::max (norm, std::abs (scaled (i, j)));
          const double scale = norm > 0 ? 1. / norm : 1.;
          rhs[i] *= scale;
          for (unsigned j = 0; j < A.n (); ++j)
            scaled (i, j) *= scale;
        }
      auto x = least_squares (scaled, rhs);
      for (unsigned j = 0; j < A.n (); ++j)
        x[j] *= columns[j];
      return x;
    }
    inline unsigned
    null_basis (const Matrix &Q, const std::vector<bool> &tied, Matrix &C)
    {
      const unsigned n = Q.n (), m = std::count (tied.begin (), tied.end (), true);
      C.reinit (n, n);
      if (!m)
        {
          for (unsigned j = 0; j < n; ++j)
            C (j, j) = 1.;
          return n;
        }
      LAPACKFullMatrix<double> A (m, n);
      unsigned row = 0;
      for (unsigned q = 0; q < Q.m (); ++q)
        if (tied[q])
          {
            for (unsigned j = 0; j < n; ++j)
              A (row, j) = Q (q, j);
            ++row;
          }
      A.compute_svd ();
      const auto &V = A.get_svd_vt ();
      unsigned rank = 0;
      const double cutoff
          = std::numeric_limits<double>::epsilon () * std::max (m, n) * A.singular_value (0);
      for (unsigned k = 0; k < std::min (m, n); ++k)
        if (A.singular_value (k) > cutoff)
          ++rank;
      for (unsigned j = rank; j < n; ++j)
        for (unsigned i = 0; i < n; ++i)
          C (i, j - rank) = V (j, i);
      return n - rank;
    }
    // Three tied Gauss values of a Q1 polynomial imply zero at both of its nodes.
    // Eliminate these coordinates BEFORE the remaining SVD. This optional exact
    // master space prevents numerical SVD shadows becoming candidate history;
    // no increment, damage, or accepted memory is clipped after solving.
    inline unsigned
    structural_q1_basis (const Matrix &Q, const std::vector<bool> &tied, Matrix &C)
    {
      AssertThrow (Q.m () % 3 == 0,
                   ExcMessage ("Structural Q1 ties require Gauss3 cell ownership"));
      std::vector<bool> zero (Q.n (), false);
      for (unsigned cell = 0; cell < Q.m () / 3; ++cell)
        {
          if (!(tied[3 * cell] && tied[3 * cell + 1] && tied[3 * cell + 2]))
            continue;
          std::vector<unsigned> support;
          for (unsigned j = 0; j < Q.n (); ++j)
            if (Q (3 * cell, j) != 0 || Q (3 * cell + 1, j) != 0 || Q (3 * cell + 2, j) != 0)
              support.push_back (j);
          AssertThrow (support.size () <= 2,
                       ExcMessage ("Structural ties are restricted to independent Q1 jump cells"));
          if (support.size () == 2)
            AssertThrow (Q (3 * cell, support[0]) * Q (3 * cell + 1, support[1])
                             != Q (3 * cell, support[1]) * Q (3 * cell + 1, support[0]),
                         ExcMessage ("Rank deficient Q1 Gauss constraints"));
          for (auto j : support)
            zero[j] = true;
        }
      std::vector<unsigned> masters;
      for (unsigned j = 0; j < Q.n (); ++j)
        if (!zero[j])
          masters.push_back (j);
      C.reinit (Q.n (), Q.n ());
      if (masters.empty ())
        return 0;
      Matrix reduced (Q.m (), masters.size ()), basis;
      for (unsigned q = 0; q < Q.m (); ++q)
        for (unsigned j = 0; j < masters.size (); ++j)
          reduced (q, j) = Q (q, masters[j]);
      const unsigned nfree = null_basis (reduced, tied, basis);
      for (unsigned i = 0; i < masters.size (); ++i)
        for (unsigned j = 0; j < nfree; ++j)
          C (masters[i], j) = basis (i, j);
      return nfree;
    }
    // Box-constrained least squares certifies the complete signed weak law.
    // Use absolute row/column sums: Q2 shapes need not be positive.
    inline double
    certificate (const Operators &op, const Vector &force, const std::vector<bool> &tied,
                 Vector &lambda, unsigned &iterations, const unsigned limit = 20000)
    {
      const unsigned n = op.Q.n (), nq = op.Q.m ();
      std::vector<unsigned> ids;
      for (unsigned q = 0; q < nq; ++q)
        if (tied[q])
          ids.push_back (q);
      lambda.reinit (nq);
      Vector target (n);
      for (unsigned i = 0; i < n; ++i)
        target[i] = -force[i] / (op.law.peak * op.mass[i]);
      if (ids.empty ())
        return op.law.peak * target.linfty_norm ();
      Matrix L (n, ids.size ());
      double maxrow = 0., maxcol = 0.;
      for (unsigned i = 0; i < n; ++i)
        {
          double sum = 0.;
          for (unsigned k = 0; k < ids.size (); ++k)
            {
              L (i, k) = op.Q (ids[k], i) * op.weights[ids[k]] / op.mass[i];
              sum += std::abs (L (i, k));
            }
          maxrow = std::max (maxrow, sum);
        }
      for (unsigned k = 0; k < ids.size (); ++k)
        {
          double sum = 0.;
          for (unsigned i = 0; i < n; ++i)
            sum += std::abs (L (i, k));
          maxcol = std::max (maxcol, sum);
        }
      Vector x = least_squares (L, target), y (x), res (n), gradient (ids.size ());
      for (unsigned k = 0; k < x.size (); ++k)
        x[k] = std::max (-1., std::min (1., x[k]));
      y = x;
      auto error = [&] (const Vector &v)
        {
          L.vmult (res, v);
          res -= target;
          return op.law.peak * res.linfty_norm ();
        };
      double err = error (x), acc = 1.;
      const double lipschitz = maxrow * maxcol;
      if (lipschitz > 0)
        for (iterations = 0; err > weak_tolerance && iterations < limit; ++iterations)
          {
            error (y);
            L.Tvmult (gradient, res);
            Vector next (y);
            next.add (-1 / lipschitz, gradient);
            for (unsigned k = 0; k < next.size (); ++k)
              next[k] = std::max (-1., std::min (1., next[k]));
            const double a = .5 * (1 + std::sqrt (1 + 4 * acc * acc));
            y = next;
            y.add ((acc - 1) / a, next);
            y.add (-(acc - 1) / a, x);
            x = next;
            acc = a;
            err = error (x);
          }
      for (unsigned k = 0; k < ids.size (); ++k)
        lambda[ids[k]] = op.law.peak * x[k];
      return err;
    }
    inline Result
    solve (const Operators &op, const Incoming &old, const double dt, const double daffine,
           const Vector *predictor = nullptr, const std::vector<bool> *predictor_ties = nullptr,
           const std::vector<double> *affine_increments = nullptr)
    {
      AssertThrow (dt > 0, ExcMessage ("Native signed solve requires positive dt"));
      const unsigned n = op.Q.n (), nq = op.Q.m ();
      const unsigned na = op.affine_modes;
      AssertThrow (na == 1
                       || (old.affine_memory.size () == na && affine_increments
                           && affine_increments->size () == na),
                   ExcMessage ("Incomplete prescribed affine history"));
      const auto &law = op.law;
      const double peak = law.peak, df = 2. * law.Gc / peak,
                   beta = 1. / (1. + (law.G / law.eta) * dt);
      AssertThrow ((op.force_offset.size () == 0 || op.force_offset.size () == n)
                       && (op.normal_offset.size () == 0 || op.normal_offset.size () == nq),
                   ExcMessage ("Incomplete native condensed history loads."));
      Result out;
      out.increment.reinit (n);
      out.tied.resize (nq);
      out.coefficients.reinit (n + na);
      std::vector<unsigned> virgin;
      for (unsigned q = 0; q < nq; ++q)
        {
          double norm = 0.;
          for (unsigned j = 0; j < n; ++j)
            norm += std::abs (op.Q (q, j));
          if (old.history[q].damage == 0 && norm > 0)
            virgin.push_back (q);
        }
      const unsigned nv = virgin.size ();
      auto evaluate = [&] (const Vector &inc, const bool ties, Vector &force, Matrix &J)
        {
          if (na == 1)
            out.coefficients[0] = beta * (old.affine + daffine);
          else
            for (unsigned a = 0; a < na; ++a)
              out.coefficients[a] = beta * (old.affine_memory[a] + (*affine_increments)[a]);
          for (unsigned j = 0; j < n; ++j)
            out.coefficients[j + na] = beta * (old.memory[j] + inc[j]);
          Vector bulk (n + na);
          op.K.vmult (bulk, out.coefficients);
          force.reinit (n);
          J.reinit (n, n);
          out.point.resize (nq);
          out.contact = 1e100;
          for (unsigned i = 0; i < n; ++i)
            {
              force[i] = bulk[i + na];
              if (op.force_offset.size ())
                force[i] += op.force_offset[i];
              for (unsigned j = 0; j < n; ++j)
                J (i, j) = beta * op.K (i + na, j + na);
            }
          for (unsigned q = 0; q < nq; ++q)
            {
              double dq = 0., pm = law.p0, pp = law.p0;
              if (op.normal_offset.size ())
                {
                  pm += op.normal_offset[q];
                  pp += op.normal_offset[q];
                }
              for (unsigned j = 0; j < n; ++j)
                dq += op.Q (q, j) * inc[j];
              if (ties && out.tied[q])
                dq = 0.;
              for (unsigned j = 0; j < n + na; ++j)
                {
                  pm += op.Nm (q, j) * out.coefficients[j];
                  pp += op.Np (q, j) * out.coefficients[j];
                }
              out.contact = std::min ({ out.contact, pm, pp });
              AssertThrow (out.contact > 0, ExcMessage ("Native signed solve lost contact"));
              auto r = CohesiveInterface::evaluate (law, old.history[q], dq, .5 * (pm + pp), dt,
                                                    dq == 0 ? 0. : std::copysign (1., dq));
              out.point[q] = r;
              double smooth = r.resistance;
              if (!ties && old.history[q].damage == 0)
                smooth -= peak * (dq == 0 ? 0. : std::copysign (1., dq));
              if (ties && out.tied[q])
                smooth = 0.;
              for (unsigned i = 0; i < n; ++i)
                {
                  const double w = op.Q (q, i) * op.weights[q];
                  force[i] += w * smooth;
                  for (unsigned j = 0; j < n; ++j)
                    J (i, j) += w
                                * (r.derivative * op.Q (q, j)
                                   + r.pressure_derivative * beta * .5
                                         * (op.Nm (q, j + na) + op.Np (q, j + na)));
                }
            }
        };
      Vector force;
      Matrix J;
      evaluate (out.increment, false, force, J);
      bool supplied_root = false;
      if (predictor && predictor_ties)
        {
          out.increment = *predictor;
          out.tied = *predictor_ties;
          evaluate (out.increment, true, force, J);
          Vector lambda;
          unsigned it = 0;
          supplied_root = certificate (op, force, out.tied, lambda, it) <= 1e-8;
        }
      if (!supplied_root)
        {
          if (predictor)
            out.increment = *predictor;
          else
            out.increment = 0.;
          evaluate (out.increment, false, force, J);
        }
      bool all_tied = false;
      if (std::all_of (old.history.begin (), old.history.end (),
                       [] (const auto &p) { return p.damage == 0.; }))
        {
          Vector multiplier;
          unsigned iterations = 0;
          all_tied = certificate (op, force, std::vector<bool> (nq, true), multiplier, iterations)
                     <= 1e-8;
        }
      Vector target (n);
      for (unsigned i = 0; i < n; ++i)
        target[i] = -force[i] / (op.mass[i] * peak);
      Matrix L (n, std::max (1u, nv));
      for (unsigned i = 0; i < n; ++i)
        for (unsigned k = 0; k < nv; ++k)
          L (i, k) = op.Q (virgin[k], i) * op.weights[virgin[k]] / op.mass[i];
      Vector y (nv), x (n), z; // dimensionless jump x and nonsmooth multiplier y.
      if (predictor && !supplied_root)
        {
          x = *predictor;
          x /= df;
        }
      if (nv)
        {
          z = least_squares (L, target);
          for (unsigned k = 0; k < nv; ++k)
            y[k] = std::max (-1., std::min (1., z[k]));
        }
      auto residual = [&] (const Vector &xx, const Vector &yy, Vector &R, Matrix *A)
        {
          Vector inc (xx);
          inc *= df;
          evaluate (inc, false, force, J);
          R.reinit (n + nv);
          if (A)
            A->reinit (n + nv, n + nv);
          for (unsigned i = 0; i < n; ++i)
            {
              R[i] = force[i] / (op.mass[i] * peak);
              for (unsigned k = 0; k < nv; ++k)
                R[i] += L (i, k) * yy[k];
              if (A)
                {
                  for (unsigned j = 0; j < n; ++j)
                    (*A) (i, j) = J (i, j) * df / (op.mass[i] * peak);
                  for (unsigned k = 0; k < nv; ++k)
                    (*A) (i, n + k) = L (i, k);
                }
            }
          for (unsigned k = 0; k < nv; ++k)
            {
              double qx = 0.;
              for (unsigned j = 0; j < n; ++j)
                qx += op.Q (virgin[k], j) * xx[j];
              const double u = qx + yy[k];
              const bool outside = std::abs (u) >= 1.;
              R[n + k] = qx - std::copysign (std::max (0., std::abs (u) - 1.), u);
              if (A)
                {
                  if (!outside)
                    for (unsigned j = 0; j < n; ++j)
                      (*A) (n + k, j) = op.Q (virgin[k], j);
                  (*A) (n + k, n + k) = outside ? -1. : 0.;
                }
            }
        };
      for (unsigned it = 0; !all_tied && !supplied_root && it < 100; ++it)
        {
          Vector R;
          Matrix A;
          residual (x, y, R, &A);
          out.iterations = it + 1;
          double weak = 0., comp = 0.;
          for (unsigned i = 0; i < n; ++i)
            weak = std::max (weak, std::abs (R[i]) * peak);
          for (unsigned k = 0; k < nv; ++k)
            comp = std::max (comp, std::abs (R[n + k]));
          if (weak <= 2e-9 && comp < 2e-13)
            break;
          AssertThrow (it < 99, ExcStepFailure ("Native signed semismooth iteration limit"));
          R *= -1.;
          auto step = equilibrated_step (A, R);
          const double norm = R.l2_norm ();
          double scale = 1.;
          bool found = false;
          std::ofstream trace;
          if(const char *path=std::getenv("ASPECT_COHESIVE_SIGNED_TRACE"))
            if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
              {
                trace.open(path,std::ios::app);
                if(trace.tellp()==0)trace<<"dt_s,n,n_virgin,iteration,backtrack,alpha,residual_L2,residual_Linf,force_Linf,complementarity_Linf,minimum_contact_Pa,linear_step_defect_L2,directional_merit_derivative\n";
                trace<<std::setprecision(17);
              }
          Vector predicted(R.size());A.vmult(predicted,step);
          const double merit_slope=-(R*predicted);
          predicted-=R;
          const double step_defect=predicted.l2_norm();
          double last_norm=norm,last_force=0.,last_constraint=0.;
          for (unsigned ls = 0; ls < 35; ++ls)
            {
              Vector xx (x), yy (y), r;
              for (unsigned i = 0; i < n; ++i)
                xx[i] += scale * step[i];
              for (unsigned k = 0; k < nv; ++k)
                yy[k] += scale * step[n + k];
              residual (xx, yy, r, nullptr);
              last_norm=r.l2_norm();last_force=last_constraint=0.;
              for(unsigned i=0;i<n;++i)last_force=std::max(last_force,std::abs(r[i]));
              for(unsigned k=0;k<nv;++k)last_constraint=std::max(last_constraint,std::abs(r[n+k]));
              if(trace)trace<<dt<<','<<n<<','<<nv<<','<<it<<','<<ls<<','<<scale<<','<<last_norm<<','<<r.linfty_norm()<<','<<last_force<<','<<last_constraint<<','<<out.contact<<','<<step_defect<<','<<merit_slope<<'\n';
              if (r.l2_norm () < norm || r.linfty_norm () < 1e-13)
                {
                  x = xx;
                  y = yy;
                  found = true;
                  break;
                }
              scale *= .5;
            }
          AssertThrow (found, ExcStepFailure ("Native signed line search failed; iteration="+std::to_string(it)
            +", entry L2="+Utilities::to_string(norm,17)+", last L2="+Utilities::to_string(last_norm,17)
            +", force Linf="+Utilities::to_string(last_force,17)+", complementarity Linf="+Utilities::to_string(last_constraint,17)
            +", contact Pa="+Utilities::to_string(out.contact,17)+", linear defect="+Utilities::to_string(step_defect,17)
            +", merit derivative="+Utilities::to_string(merit_slope,17)));
        }
      if (!supplied_root)
        {
          out.increment = x;
          out.increment *= df;
        }
      Vector dq (nq);
      op.Q.vmult (dq, out.increment);
      const double roundoff = 256 * std::numeric_limits<double>::epsilon () * dq.linfty_norm ();
      if (!supplied_root)
        for (unsigned q = 0; q < nq; ++q)
          out.tied[q] = old.history[q].damage == 0.;
      // Use the solved primal-dual active set. Near a mature seed the slip can be
      // below 1e-17 m, so a relative-primal-only test misclassifies roundoff as
      // virgin release. Interior dual multipliers impose exact Gauss equalities.
      if (!all_tied && !supplied_root)
        for (unsigned k = 0; k < nv; ++k)
          {
            double qx = 0.;
            for (unsigned j = 0; j < n; ++j)
              qx += op.Q (virgin[k], j) * x[j];
            out.tied[virgin[k]]
                = std::abs (qx + y[k]) < 1.
                  || (std::abs (dq[virgin[k]]) <= roundoff && std::abs (y[k]) <= 1.);
          }

      out.free = op.structural_q1_ties ? structural_q1_basis (op.Q, out.tied, out.basis)
                                       : null_basis (op.Q, out.tied, out.basis);
      Vector projected (n);
      for (unsigned k = 0; k < out.free; ++k)
        {
          double p = 0.;
          for (unsigned j = 0; j < n; ++j)
            p += out.basis (j, k) * out.increment[j];
          for (unsigned i = 0; i < n; ++i)
            projected[i] += out.basis (i, k) * p;
        }
      out.increment = projected;
      for (unsigned it = 0; it < 12; ++it)
        {
          evaluate (out.increment, true, force, J);
          if (!out.free)
            break;
          Vector f (out.free);
          Matrix reduced (out.free);
          Matrix C (n, out.free), JC (n, out.free);
          for (unsigned j = 0; j < n; ++j)
            for (unsigned i = 0; i < out.free; ++i)
              {
                C (j, i) = out.basis (j, i);
                f[i] += C (j, i) * force[j];
              }
          J.mmult (JC, C);
          C.Tmmult (reduced, JC);
          if (f.linfty_norm () < 1e-12)
            break;
          f *= -1.;
          reduced.gauss_jordan ();
          Vector step (out.free);
          reduced.vmult (step, f);
          for (unsigned j = 0; j < n; ++j)
            for (unsigned i = 0; i < out.free; ++i)
              out.increment[j] += out.basis (j, i) * step[i];
        }
      evaluate (out.increment, true, force, J);
      out.tangent = J;
      out.weak = certificate (op, force, out.tied, out.lambda, out.certificate_iterations);
      if (out.weak > weak_tolerance)
        {
          const char *path = std::getenv ("CI_EVOLUTION_FAILURE");
          if (path)
            {
              std::ofstream f (path);
              f << std::setprecision (17) << dt << ' ' << daffine << ' ' << old.affine << ' '
                << out.weak << '\n';
              for (unsigned j = 0; j < n; ++j)
                f << old.memory[j] << ' ' << out.increment[j] << '\n';
              for (unsigned q = 0; q < nq; ++q)
                f << old.history[q].jump << ' ' << old.history[q].damage << ' '
                  << old.history[q].theta << ' ' << out.point[q].state.jump << ' '
                  << out.point[q].state.damage << ' ' << out.tied[q] << ' ' << out.lambda[q]
                  << '\n';
            }
        }
      AssertThrow (out.weak <= weak_tolerance, ExcMessage ("Native signed complete weak certificate failed: "
                                                 + std::to_string (out.weak)));
      for (unsigned q = 0; q < nq; ++q)
        if (out.tied[q])
          {
            out.point[q].q = out.lambda[q];
            out.point[q].resistance = out.lambda[q];
          }
      if (out.free)
        {
          Matrix C (op.Q.n (), out.free);
          // Orthonormalize in the physical nodal arclength mass metric.
          for (unsigned k = 0; k < out.free; ++k)
            {
              for (unsigned j = 0; j < n; ++j)
                C (j, k) = out.basis (j, k);
              for (unsigned p = 0; p < k; ++p)
                {
                  double dot = 0.;
                  for (unsigned j = 0; j < n; ++j)
                    dot += op.mass[j] * C (j, k) * C (j, p);
                  for (unsigned j = 0; j < n; ++j)
                    C (j, k) -= dot * C (j, p);
                }
              double norm = 0.;
              for (unsigned j = 0; j < n; ++j)
                norm += op.mass[j] * C (j, k) * C (j, k);
              for (unsigned j = 0; j < n; ++j)
                C (j, k) /= std::sqrt (norm);
            }
          Matrix JC (n, out.free);
          J.mmult (JC, C);
          LAPACKFullMatrix<double> symmetric (out.free);
          for (unsigned i = 0; i < out.free; ++i)
            for (unsigned k = 0; k < out.free; ++k)
              for (unsigned j = 0; j < n; ++j)
                symmetric (i, k) += .5 * (C (j, i) * JC (j, k) + C (j, k) * JC (j, i));
          symmetric.compute_eigenvalues ();
          out.curvature = 1e100;
          for (unsigned i = 0; i < out.free; ++i)
            out.curvature = std::min (out.curvature, symmetric.eigenvalue (i).real ());
          AssertThrow (
              out.curvature > 0.,
              ExcStepFailure ("Native signed candidate has negative admissible symmetric tangent"));
        }
      return out;
    }
  }
}
#endif
