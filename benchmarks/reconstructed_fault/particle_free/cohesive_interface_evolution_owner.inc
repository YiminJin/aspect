// Default-off native time solve. Inputs contain operators and one initial
// physical snapshot, never accepted target trajectories.
bool evolution_ready = false;
double evolution_clock = .2, evolution_gamma = .0015;
unsigned evolution_publications = 0;
std::string evolution_configuration()const
{
  std::ostringstream s;
  for (const char *k :
  {"CI_NATIVE_EVOLUTION", "CI_EVOLUTION_HOLD", "CI_EVOLUTION_RATE"
  })
  {
    const char *v = std::getenv(k);
    s << k << '=' << (v ? v : "") << ';';
  }
  return s.str();
}
CohesiveEvolution::Operators evolution_operators(bool extended)const
{
  const unsigned n = extended ? nj : nj / 2;
  CohesiveEvolution::Operators op;
  op.K.reinit(n + 1, n + 1);
  op.Q.reinit(nq, n);
  op.Nm.reinit(nq, n + 1);
  op.Np.reinit(nq, n + 1);
  op.mass.assign(n, 0.);
  op.weights = w;
  for (unsigned i = 0; i <= n; ++i)for (unsigned j = 0; j <= n; ++j)op.K(i, j) = K[i * (nj + 1) + j];
  for (unsigned q = 0; q < nq; ++q)
    {
      for (unsigned j = 0; j <= n; ++j)
        {
          op.Nm(q, j) = Nm[q * (nj + 1) + j];
          op.Np(q, j) = Np[q * (nj + 1) + j];
        }
      for (unsigned j = 0; j < n; ++j)
        {
          op.Q(q, j) = Q[q * nj + j];
          op.mass[j] += w[q] * op.Q(q, j);
        }
    }
  return op;
}
AdmissionRecord solve_evolution(double target, CohesiveEvolution::Result &result)const
{
  const auto op = evolution_operators(generation == 1);
  const unsigned n = op.Q.n();
  CohesiveEvolution::Incoming old;
  old.affine = coefficients[0];
  old.memory.assign(memory.begin(), memory.begin() + n);
  old.history.resize(nq);
  for (unsigned q = 0; q < nq; ++q)old.history[q] = {hist[6 * q], hist[6 * q + 1], hist[6 * q + 2], hist[6 * q + 3], hist[6 * q + 4], hist[6 * q + 5]};
  const double hold = std::getenv("CI_EVOLUTION_HOLD") ? std::stod(std::getenv("CI_EVOLUTION_HOLD")) : .65;
  const double rate = std::getenv("CI_EVOLUTION_RATE") ? std::stod(std::getenv("CI_EVOLUTION_RATE")) : .001;
  const double du = rate * (std::min(target, hold) - std::min(evolution_clock, hold));
  result = CohesiveEvolution::solve(op, old, target - evolution_clock, du);
  AdmissionRecord r;
  r.time = target;
  r.gamma = evolution_gamma + du;
  r.c.assign(nj + 1, 0.);
  r.memory.assign(nj, 0.);
  r.displacement = delta;
  r.c[0] = result.coefficients[0];
  for (unsigned j = 0; j < n; ++j)
    {
      r.c[j + 1] = r.memory[j] = result.coefficients[j + 1];
      r.displacement[j] += result.increment[j];
    }
  r.history.resize(6 * nq);
  r.tied.resize(nq);
  r.lambda.resize(nq);
  for (unsigned q = 0; q < nq; ++q)
    {
      const auto &p = result.point[q].state;
      double v[] = {p.jump, p.damage, p.theta, p.reference, p.relaxed_jump, p.absolute_slip};
      for (unsigned j = 0; j < 6; ++j)r.history[6 * q + j] = v[j];
      r.tied[q] = result.tied[q];
      r.lambda[q] = result.lambda[q];
    }
  r.bulk.assign(nm, 0.); // Native response is used for independent field reconstruction.
  for (unsigned i = 0; i < nm; ++i)for (unsigned j = 0; j <= nj; ++j)r.bulk[i] += native_response[i * (nj + 1) + j] * r.c[j];
  return r;
}
void publish_evolution(const AdmissionRecord &r, unsigned base)
{
  AssertThrow(base == evolution_publications && r.time > evolution_clock, ExcMessage("Stale native evolution publication"));
  for (unsigned q = 0; q < nq; ++q)AssertThrow(r.history[6 * q + 1] >= hist[6 * q + 1] && r.history[6 * q + 1] <= 1 && r.history[6 * q + 2] > 0, ExcMessage("Invalid native evolved history"));
  auto staged = owner.clone_prescribed_geometry();
  for (unsigned i = 0; i < (generation == 1 ? nx / 2 : nx / 4); ++i)
    {
      auto p = staged->get_fault(0).get_properties(i);
      for (unsigned j = 0; j < 18; ++j)p[j] = r.history[18 * i + j];
    }
  owner.swap_prescribed_geometry(*staged);
  coefficients = r.c;
  bulk = r.bulk;
  delta = r.displacement;
  memory = r.memory;
  hist = r.history;
  evolution_clock = r.time;
  evolution_gamma = r.gamma;
  ++evolution_publications;
}
void evolve(double target)
{
  if (target <= evolution_clock)return;
  if (generation == 0 && evolution_clock >= .25 - 1e-12)
    {
      auto op = evolution_operators(true);
      dealii::Vector<double> c(nj + 1), f(nj + 1), force(nj), mult;
      for (unsigned j = 0; j <= nj; ++j)c[j] = coefficients[j];
      op.K.vmult(f, c);
      for (unsigned j = 0; j < nj; ++j)force[j] = f[j + 1];
      unsigned it = 0;
      const double error = CohesiveEvolution::certificate(op, force, std::vector<bool>(nq, true), mult, it);
      AssertThrow(error <= 1e-8, ExcMessage("Native evolution tied insertion infeasible"));
      lambda.resize(nq / 2);
      for (unsigned q = nq / 2; q < nq; ++q)lambda[q - nq / 2] = mult[q];
      insertion();
    }
  const auto before = bytes() + evolution_bytes();
  CohesiveEvolution::Result solved;
  AdmissionRecord trial;
  try
    {
      trial = solve_evolution(target, solved);
    }
  catch (...)
    {
      std::ofstream f(this->get_output_directory() + "evolution-last-accepted.archive");
      f << before;
      throw;
    }
  // The existing independent publication checker applies only to the full
  // installed trace space. Before insertion certify the actual old subspace.
  if (generation == 1)
    {
      physical_clock = evolution_clock;
      physical_gamma = evolution_gamma;
      check_admission(trial, admission_index);
    }
  if (evolution_publications == 0 || (generation == 1 && evolution_clock < .582 && target >= .582))
    {
      CohesiveEvolution::Result retry;
      auto again = solve_evolution(target, retry);
      AssertThrow(again.history == trial.history && again.memory == trial.memory && before == bytes() + evolution_bytes(), ExcMessage("Native evolution retry changed accepted state"));
      auto bad = trial;
      bad.history[1] = -1.;
      bool rejected = false;
      dealii::deal_II_exceptions::disable_abort_on_exception();
      try
        {
          publish_evolution(bad, evolution_publications);
        }
      catch (const std::exception&)
        {
          rejected = true;
        }
      dealii::deal_II_exceptions::enable_abort_on_exception();
      AssertThrow(rejected && before == bytes() + evolution_bytes(), ExcMessage("Native evolution rejection changed accepted state"));
      auto lifecycle = std::ofstream(this->get_output_directory() + "evolution-lifecycle.csv", std::ios::app);
      lifecycle << std::setprecision(17) << target << ",1,1\n";
    }

  if (solved.free && (evolution_publications % 20 == 0 || target >= .65))
    {
      const auto op = evolution_operators(generation == 1);
      const unsigned n = op.Q.n();
      dealii::Vector<double> direction(n);
      for (unsigned j = 0; j < n; ++j)for (unsigned k = 0; k < solved.free; ++k)direction[j] += solved.basis(j, k) * std::cos(double(k + 1));
      direction /= direction.l2_norm();
      auto force = [&](const dealii::Vector<double> &increment)
      {
        dealii::Vector<double> c(n + 1), bulk(n + 1), f(n);
        const double beta = 1. / (1. + .01 * (target - evolution_clock));
        c[0] = trial.c[0];
        for (unsigned j = 0; j < n; ++j)c[j + 1] = beta * (memory[j] + increment[j]);
        op.K.vmult(bulk, c);
        for (unsigned j = 0; j < n; ++j)f[j] = bulk[j + 1];
        for (unsigned q = 0; q < nq; ++q)if (!solved.tied[q])
            {
              double dq = 0., pn = 10000.;
              for (unsigned j = 0; j < n; ++j)dq += op.Q(q, j) * increment[j];
              for (unsigned j = 0; j <= n; ++j)pn += .5 * (op.Nm(q, j) + op.Np(q, j)) * c[j];
              CohesiveInterface::PointState old {hist[6 * q], hist[6 * q + 1], hist[6 * q + 2], hist[6 * q + 3], hist[6 * q + 4], hist[6 * q + 5]};
              const auto value = CohesiveInterface::evaluate(CohesiveInterface::Parameters(), old, dq, pn, target - evolution_clock);
              for (unsigned j = 0; j < n; ++j)f[j] += op.weights[q] * op.Q(q, j) * value.resistance;
            }
        return f;
      };
      const double eps = std::max(solved.increment.l2_norm(), 1e-20) * 1e-5;
      auto plus = solved.increment, minus = plus;
      plus.add(eps, direction);
      minus.add(-eps, direction);
      auto fd = force(plus);
      fd -= force(minus);
      fd /= 2 * eps;
      dealii::Vector<double> exact(n);
      solved.tangent.vmult(exact, direction);
      double err = 0., norm = 0.;
      for (unsigned k = 0; k < solved.free; ++k)
        {
          double d = 0., v = 0.;
          for (unsigned j = 0; j < n; ++j)
            {
              d += solved.basis(j, k) * (fd[j] - exact[j]);
              v += solved.basis(j, k) * exact[j];
            }
          err += d * d;
          norm += v * v;
        }
      const double relative = std::sqrt(err / std::max(norm, 1.));
      AssertThrow(relative < 1e-6, ExcMessage("Native evolution coupled derivative failed"));
      std::ofstream f(this->get_output_directory() + "evolution-derivatives.csv", std::ios::app);
      f << std::setprecision(17) << target << ',' << relative << '\n';
    }
  const double dt = target - evolution_clock, du = trial.gamma - evolution_gamma;
  auto energy = [&](const std::vector<double>&c)
  {
    double v = 0.;
    for (unsigned i = 0; i <= nj; ++i)for (unsigned j = 0; j <= nj; ++j)v += .5 * c[i] * K[i * (nj + 1) + j] * c[j];
    return v;
  };
  const double oldE = energy(coefficients), E = energy(trial.c);
  double bond = 0., oldbond = 0., fracture = 0., sliding = 0., surface = 0., work = 0., trap = 0., be = 0.;
  std::vector<double>dc(nj + 1);
  for (unsigned i = 0; i <= nj; ++i)dc[i] = trial.c[i] - coefficients[i];
  be = energy(dc);
  for (unsigned j = 0; j <= nj; ++j)
    {
      work += K[j] * trial.c[j] * du;
      trap += .5 * K[j] * (trial.c[j] + coefficients[j]) * du;
    }
  for (unsigned q = 0; q < nq; ++q)
    {
      const auto &p = solved.point[q];
      const double a = hist[6 * q + 1], e = hist[6 * q] - hist[6 * q + 3];
      const double oldstorage = a == 0 || a == 1 ? 0. : .5 * (1 - a) * e * e / ((2. / 9e6) * a);
      double dq = 0.;
      for (unsigned j = 0; j < nj; ++j)dq += Q[q * nj + j] * (trial.displacement[j] - delta[j]);
      bond += w[q] * p.storage;
      oldbond += w[q] * oldstorage;
      fracture += w[q] * (p.state.damage - a);
      sliding += w[q] * (p.friction + p.radiation);
      surface += w[q] * (p.q * dq - p.storage + oldstorage - (p.state.damage - a));
    }
  const double maxwell = .02 * dt * E, ledger = work - (E - oldE + bond - oldbond + fracture + sliding + maxwell + be + surface), remainder = trap - (E - oldE + bond - oldbond + fracture + sliding + maxwell);
  AssertThrow(std::abs(ledger) <= 1e-10 * std::max(1., std::abs(work)), ExcMessage("Native evolution ledger failed"));
  const unsigned base = evolution_publications;
  publish_evolution(trial, base);
  const auto committed = bytes() + evolution_bytes();
  bool duplicate = false;
  dealii::deal_II_exceptions::disable_abort_on_exception();
  try
    {
      publish_evolution(trial, base);
    }
  catch (const std::exception&)
    {
      duplicate = true;
    }
  dealii::deal_II_exceptions::enable_abort_on_exception();
  AssertThrow(duplicate && committed == bytes() + evolution_bytes(), ExcMessage("Duplicate native evolution publication"));
  auto f = std::ofstream(this->get_output_directory() + "evolution-history.csv", std::ios::app);
  f << std::setprecision(17);
  if (f.tellp() == 0)f << "time_s,dt_s,publications,generation,gamma,bulk_J_m,bond_J_m,fracture_increment_J_m,sliding_loss_J_m,maxwell_loss_J_m,endpoint_work_J_m,trapezoid_work_J_m,physical_remainder_J_m,ledger_error_J_m,weak_error_Pa,curvature_Pa_m,min_normal_Pa,iterations\n";
  f << target << ',' << dt << ',' << evolution_publications << ',' << generation << ',' << evolution_gamma << ',' << E << ',' << bond << ',' << fracture << ',' << sliding << ',' << maxwell << ',' << work << ',' << trap << ',' << remainder << ',' << ledger << ',' << solved.weak << ',' << solved.curvature << ',' << solved.contact << ',' << solved.iterations << '\n';
  auto h = std::ofstream(this->get_output_directory() + "evolution-states.csv", std::ios::app);
  h << std::setprecision(17);
  if (h.tellp() == 0)h << "time_s,q,jump_m,damage,theta,reference_m,memory_m,absolute_slip_m\n";
  for (unsigned q = 0; q < nq; ++q)
    {
      h << target << ',' << q;
      for (unsigned j = 0; j < 6; ++j)h << ',' << hist[6 * q + j];
      h << '\n';
    }
}
std::string evolution_bytes()const
{
  std::ostringstream s;
  {
    aspect::oarchive a(s);
    a << evolution_clock << evolution_gamma << evolution_publications << native_response << evolution_configuration();
  }
  return s.str();
}
void restore_evolution(const std::string &s)
{
  std::string config;
  std::istringstream f(s);
  aspect::iarchive a(f);
  a >> evolution_clock >> evolution_gamma >> evolution_publications >> native_response >> config;
  AssertThrow(config == evolution_configuration(), ExcMessage("Native evolution restart configuration changed"));
  evolution_ready = true;
  // ASPECT checkpoints precede the postprocessor at a checkpointed host step.
  // Preserve, then remove diagnostic rows ahead of the restored physical state.
  for (const char *name :
  {"evolution-history.csv", "evolution-states.csv", "evolution-lifecycle.csv"
  })
  {
    const std::string path = this->get_output_directory() + name;
    std::ifstream in(path);
    if (!in)continue;
    std::ostringstream keep, ahead;
    std::string line;
    while (std::getline(in, line))
      {
        std::istringstream row(line);
        double t;
        if (!(row >> t) || t <= evolution_clock)keep << line << '\n';
        else ahead << line << '\n';
      }
    in.close();
    if (!ahead.str().empty())
      {
        std::ofstream old(path + ".ahead-of-checkpoint", std::ios::app);
        old << ahead.str();
      }
    std::ofstream out(path);
    out << keep.str();
  }
}
void initialize_evolution()
{
  const char *path = std::getenv("CI_EVOLUTION_RESPONSE");
  AssertThrow(path, ExcMessage("Missing native response"));
  std::ifstream f(path);
  native_response.resize(nm * (nj + 1));
  for (auto &v : native_response)f >> v;
  AssertThrow(f, ExcMessage("Bad native response"));
  evolution_ready = true;
}
