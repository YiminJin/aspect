// Qualified experimental alternative to the retained monotone predictor.
// Cache fixed geometry only; every solve anchors the current accepted state.
mutable CohesiveEvolution::Operators signed_op;
mutable std::vector<unsigned> signed_ids;
mutable unsigned signed_generation = std::numeric_limits<unsigned>::max();
Trial signed_front_trial(const std::vector<PointState> &old, double dt, double du)const
{
  if (signed_generation != geometry_publications)
    {
      signed_ids.clear();
      for (unsigned i = 0; i < ns; ++i)if (movable_node(i))signed_ids.push_back(i);
      const unsigned n = signed_ids.size();
      signed_op.K.reinit(n + 1, n + 1);
      signed_op.Q.reinit(3 * ns, n);
      signed_op.Nm.reinit(3 * ns, n + 1);
      signed_op.Np.reinit(3 * ns, n + 1);
      signed_op.weights = qweights;
      signed_op.mass.resize(n);
      signed_op.K(0, 0) = law.G * L / H;
      for (unsigned i = 0; i < n; ++i)
        {
          const unsigned a = signed_ids[i];
          signed_op.mass[i] = weights[a];
          signed_op.K(0, i + 1) = signed_op.K(i + 1, 0) = -weights[a] * law.G / H;
          for (unsigned j = 0; j < n; ++j)signed_op.K(i + 1, j + 1) = weights[a] * K(a, signed_ids[j]);
          for (unsigned q = 0; q < 3 * ns; ++q)
            {
              signed_op.Q(q, i) = Q(q, a);
              for (unsigned k = 0; k < ns; ++k)
                {
                  signed_op.Nm(q, i + 1) += QP(q, k) * normal_lower(k, a);
                  signed_op.Np(q, i + 1) += QP(q, k) * normal_upper(k, a);
                }
            }
        }
      signed_generation = geometry_publications;
    }
  CohesiveEvolution::Incoming incoming;
  incoming.affine = wall;
  incoming.history = integration_points();
  for (const auto i : signed_ids)incoming.memory.push_back(old[i].relaxed_jump);
  // Reuse the established cheap front predictor. Its positivity restriction
  // is only a search strategy: the complete signed Gauss certificate and
  // unconstrained nullspace Newton solve remain the acceptance equations.
  Trial predictor;
  bool predicted = false;
  std::vector<bool> hint_ties(3 * ns);
  dealii::Vector<double> hint(signed_ids.size());
  dealii::deal_II_exceptions::disable_abort_on_exception();
  try
    {
      predictor = front_trial(old, dt, du);
      predicted = true;
    }
  catch (const std::exception &e)
    {
      const std::string error = e.what();
      if (error.find("line search") == std::string::npos && error.find("lost closed contact") == std::string::npos)
        {
          dealii::deal_II_exceptions::enable_abort_on_exception();
          throw;
        }
    }
  dealii::deal_II_exceptions::enable_abort_on_exception();
  for (unsigned j = 0; j < signed_ids.size(); ++j)hint[j] = predicted ? predictor.increment[signed_ids[j]] : properties(signed_ids[j])[rate_property] * dt;
  for (unsigned q = 0; q < 3 * ns; ++q)hint_ties[q] = predicted ? predictor.tied[q] : incoming.history[q].damage == 0.;
  if (!predicted)
    {
      auto f = output("predictor-fallback.csv", true);
      if (f.tellp() == 0)f << "target_s,dt_s\n";
      f << accepted_time + dt << ',' << dt << '\n';
    }
  const auto result = CohesiveEvolution::solve(signed_op, incoming, dt, du, &hint, &hint_ties);
  Trial out;
  out.base_publications = publications;
  out.base_geometry_publications = geometry_publications;
  out.wall = result.coefficients[0];
  out.nodal = old;
  out.point = result.point;
  out.tied = result.tied;
  out.increment.assign(ns, 0.);
  out.active.assign(ns, false);
  out.jacobian.reinit(ns, ns);
  out.stress.assign(ns, law.G / H * out.wall);
  out.pressure.assign(ns, law.p0);
  out.qstress.assign(3 * ns, 0.);
  out.qpressure.assign(3 * ns, 0.);
  const double beta = 1. / (1. + law.G / law.eta * dt);
  for (unsigned j = 0; j < signed_ids.size(); ++j)
    {
      const unsigned i = signed_ids[j];
      out.increment[i] = result.increment[j];
      out.nodal[i].jump += result.increment[j];
      out.nodal[i].relaxed_jump = result.coefficients[j + 1];
      out.nodal[i].absolute_slip += std::abs(result.increment[j]);
      double norm = 0.;
      for (unsigned k = 0; k < result.free; ++k)norm += std::abs(result.basis(j, k));
      out.active[i] = norm > 1e-10;
      for (unsigned k = 0; k < signed_ids.size(); ++k)out.jacobian(i, signed_ids[k]) = -result.tangent(j, k) / weights[i];
    }
  // Preserve derivatives of fixed-row reactions for the existing full
  // diagnostic FD check, although those rows are not Newton test functions.
  for (unsigned i = 0; i < ns; ++i)if (!movable_node(i))for (unsigned k = 0; k < signed_ids.size(); ++k)
        {
          const unsigned j = signed_ids[k];
          out.jacobian(i, j) = -beta * K(i, j);
          for (unsigned q = 0; q < 3 * ns; ++q)out.jacobian(i, j) -= qweights[q] * Q(q, i) / weights[i] * (result.point[q].derivative * Q(q, j) + result.point[q].pressure_derivative * beta * .5 * (signed_op.Nm(q, k + 1) + signed_op.Np(q, k + 1)));
        }
  for (unsigned i = 0; i < ns; ++i)for (unsigned j = 0; j < ns; ++j)
      {
        out.stress[i] -= K(i, j) * beta * (old[j].relaxed_jump + out.increment[j]);
        out.pressure[i] += N(i, j) * beta * (old[j].relaxed_jump + out.increment[j]);
      }
  for (unsigned q = 0; q < 3 * ns; ++q)
    for (unsigned i = 0; i < ns; ++i)
      {
        out.qstress[q] += QP(q, i) * out.stress[i];
        out.qpressure[q] += QP(q, i) * out.pressure[i];
        if (Q(q, i) != 0 && out.point[q].state.damage > 0)out.nodal[i].damage = 1.;
      }
  out.error = out.weak_error = result.weak;
  out.curvature = result.curvature;
  out.minimum_trace_pressure = result.contact;
  out.iterations = result.iterations + (predicted ? predictor.iterations : 0);
  out.certificate_iterations = result.certificate_iterations;
  return out;
}
