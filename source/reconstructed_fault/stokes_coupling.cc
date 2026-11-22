/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/boundary_velocity/interface.h>
#include <aspect/simulator/assemblers/fault_stress.h>
#include <aspect/simulator/assemblers/stokes.h>
#include <aspect/simulator.h>
#include <aspect/prescribed_dilation/interface.h>
#include <deal.II/fe/fe_tools.h>

namespace aspect
{
  namespace
  {
    // A constant pressure changes the weak momentum balance if an exterior
    // face admits normal test velocity. A prescribed tangential component
    // still allows this flux. A periodic pair cancels it.
    template <int dim>
    bool has_normal_traction_boundary(const SimulatorAccess<dim> &sim)
    {
      const auto &geometry = sim.get_geometry_model();
      const auto &velocity = sim.get_boundary_velocity_manager();
      auto faces = geometry.get_used_boundary_indicators();
      for (const auto &pair : geometry.get_periodic_boundary_pairs())
        {
          faces.erase(pair.first.first);
          faces.erase(pair.first.second);
        }
      for (const auto face : faces)
        {
          // Opposite faces of the supported Cartesian box are 2*d and 2*d+1.
          const auto normal_component = sim.introspection().component_indices.velocities[face/2];
          if (!velocity.get_zero_boundary_velocity_indicators().count(face) &&
              !velocity.get_tangential_boundary_velocity_indicators().count(face) &&
              (!velocity.get_prescribed_boundary_velocity_indicators().count(face) ||
               !velocity.get_component_mask(face)[normal_component]))
            return true;
        }
      return false;
    }
  }

  template <int dim>
  FaultStokesCoupling<dim>::FaultStokesCoupling(MaterialModel::PhaseFieldFault<dim> &material)
    : material(material)
  {}

  template <int dim>
  void FaultStokesCoupling<dim>::initialize()
  {
    auto &signals = this->get_signals();
    signals.set_assemblers.connect([](const SimulatorAccess<dim> &sim, Assemblers::Manager<dim> &assemblers)
    {
      AssertThrow(has_normal_traction_boundary(sim),
                  ExcMessage("Stationary fault friction requires a pressure datum from a nonperiodic box face "
                             "with unconstrained normal velocity (prescribed or natural normal traction)."));
      // For the 3-D deviator in plane strain the velocity matrix needs
      // -2*kappa/3 div(u) div(v), even with incompressible continuity.
      AssertThrow(sim.get_prescribed_dilation_manager().get_active_plugin_names().empty(),
                  ExcMessage("Phase field fault requires incompressible continuity without prescribed dilation."));
      if (!sim.get_parameters().use_full_A_block_preconditioner)
        assemblers.stokes_preconditioner.push_back(std::make_unique<Assemblers::StokesCompressiblePreconditioner<dim>>());
      assemblers.stokes_system.push_back(std::make_unique<Assemblers::StokesCompressibleStrainRateViscosityTerm<dim>>());
      assemblers.stokes_system.push_back(std::make_unique<Assemblers::FaultStressLoad<dim>>());
    });
    using namespace std::placeholders;
    signals.prepare_stokes_picard_iteration.connect(std::bind(&FaultStokesCoupling::prepare_iteration,this,_1,_2));
    signals.check_stokes_picard_convergence.connect(std::bind(&FaultStokesCoupling::check_convergence,this,_1,_2));
    signals.post_accept_timestep.connect(std::bind(&FaultStokesCoupling::accept_step,this,_1));
  }

  template <int dim>
  void FaultStokesCoupling<dim>::prepare_iteration(const LinearAlgebra::BlockVector &iterate, bool &rebuild)
  {
    if (this->get_nonlinear_iteration() == 0)
      rebuild = true; // The Maxwell coefficient depends on this attempt's duration.
    trial_rates = solve_rates(iterate);
  }

  template <int dim>
  void FaultStokesCoupling<dim>::check_convergence(const LinearAlgebra::BlockVector &solution, bool &converged)
  {
    auto rates = solve_rates(solution);
    for (unsigned int j=0; j<rates.size(); ++j)
      if (std::abs(rates[j]-trial_rates[j]) > 1.e-8*std::max(rates[j],trial_rates[j]))
        {
          converged = false;
          return;
        }
    trial_rates.swap(rates);
  }

  template <int dim>
  const FiniteElement<dim> &FaultStokesCoupling<dim>::stress_element() const
  {
    return this->get_fe().base_element(this->introspection().base_elements.compositional_fields[material.stress_fields[0]]);
  }

  template <int dim>
  void FaultStokesCoupling<dim>::prepare_parent(const double beta)
  {
    const double physical_dt = this->get_timestep();
    trial_rates.clear();
    for (const auto &label : material.fault.labels())
      trial_rates.push_back(label.rate);

    const QGauss<dim> rule(8);
    const auto &dg = stress_element();
    FullMatrix<double> projector(dg.n_dofs_per_cell(),rule.size());
    FETools::compute_projection_from_quadrature_points_matrix(dg,rule,rule,projector);
    FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_values | update_gradients | update_quadrature_points);
    const auto &fields = this->get_old_solution();
    const auto &extract = this->introspection().extractors;
    std::vector<Tensor<2,dim>> gradients(rule.size());
    std::array<std::vector<double>,3> components;
    for (auto &v : components)
      v.resize(rule.size());
    std::vector<Stress> rotated(rule.size());
    std::vector<double> phase(rule.size());
    double min_phase = 1., max_phase = 0.;
    bool aligned = true;
    relaxed_parent.clear();
    for (const auto &cell : this->get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          // With fixed constant phase and resolved Q1 knots the localized
          // slip stress is already in DGQ2. This makes the physical stress
          // representation exact also on the stationary donor's split path.
          aligned &= material.fault.resolves_segments(cell->vertex(0),cell->vertex(GeometryInfo<dim>::vertices_per_cell-1));
          fe.reinit(cell);
          fe[extract.velocities].get_function_gradients(fields,gradients);
          fe[extract.compositional_fields[material.phase_field]].get_function_values(fields,phase);
          for (unsigned int c=0; c<3; ++c)
            fe[extract.compositional_fields[material.stress_fields[c]]].get_function_values(fields,components[c]);
          for (unsigned int q=0; q<rule.size(); ++q)
            {
              min_phase = std::min(min_phase,phase[q]);
              max_phase = std::max(max_phase,phase[q]);
              Stress parent;
              for (unsigned int c=0; c<3; ++c)
                parent.access_raw_entry(c) = components[c][q];
              const auto foot = material.fault.project(fe.quadrature_point(q));
              const double rate = foot.interpolate(trial_rates[foot.left],trial_rates[foot.left+1]);
              rotated[q] = material.relaxed_parent(parent,gradients[q],rate,physical_dt,beta);
            }
          auto &coefficients = relaxed_parent[cell->id()];
          coefficients.resize(dg.n_dofs_per_cell());
          FETools::compute_projection_from_quadrature_points(projector,rotated,coefficients);
        }
    const auto comm = this->get_mpi_communicator();
    min_phase = Utilities::MPI::min(min_phase,comm);
    max_phase = Utilities::MPI::max(max_phase,comm);
    AssertThrow(min_phase > 0. && max_phase < 1. && max_phase-min_phase < 1.e-12,
                ExcMessage("The stationary section normalization requires a constant phase between zero and one."));
    AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(aligned),comm),
                ExcMessage("Interior fault labels must lie on transverse bulk cell faces."));
  }

  template <int dim>
  std::vector<typename FaultStokesCoupling<dim>::BulkPoint>
  FaultStokesCoupling<dim>::sample_bulk(const LinearAlgebra::BlockVector &iterate) const
  {
    const QGauss<dim> rule(8);
    FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FullMatrix<double> basis(rule.size(),stress_element().n_dofs_per_cell());
    FETools::compute_interpolation_to_quadrature_points_matrix(stress_element(),rule,basis);
    std::vector<Stress> strain(rule.size());
    std::vector<double> pressure(rule.size());
    std::vector<BulkPoint> samples;
    const auto S = symmetrize(material.fault.slip_dyad());
    bool locally_closed = true;
    for (const auto &cell : this->get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe.reinit(cell);
          fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(iterate,strain);
          fe[this->introspection().extractors.pressure].get_function_values(iterate,pressure);
          const auto &parent = relaxed_parent.at(cell->id());
          for (unsigned int q=0; q<rule.size(); ++q)
            {
              Stress stress = material.viscous_stress(strain[q]);
              for (unsigned int i=0; i<parent.size(); ++i)
                stress += basis(q,i)*parent[i];
              const double compression = material.background_compression + pressure[q] - material.fault.normal_vector()*(stress*material.fault.normal_vector());
              locally_closed &= std::isfinite(compression) && compression > 0.;
              samples.push_back({material.fault.project(fe.quadrature_point(q)),fe.JxW(q)*material.fault.localization(),
                                 material.background_shear + stress*S,compression
                                });
            }
        }
    // The extrapolated Picard predictor can open an otherwise closed fault.
    // All ranks reject this trial before evaluating friction or staging rates.
    if (!Utilities::MPI::min(static_cast<unsigned int>(locally_closed),this->get_mpi_communicator()))
      {
        this->get_pcout() << "Stationary fault trial requires finite, positive normal compression." << std::endl;
        throw ExcNonlinearSolverNoConvergence();
      }
    return samples;
  }

  template <int dim>
  typename FaultStokesCoupling<dim>::Stress
  FaultStokesCoupling<dim>::stress_offset(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                          const Point<dim> &position) const
  {
    const auto foot = material.fault.project(position);
    const double rate = foot.interpolate(trial_rates[foot.left],trial_rates[foot.left+1]);
    Stress result = material.slip_stress(rate);
    const auto unit = this->get_mapping().transform_real_to_unit_cell(cell,position);
    const auto &parent = relaxed_parent.at(cell->id());
    for (unsigned int i=0; i<parent.size(); ++i)
      result += stress_element().shape_value(i,unit)*parent[i];
    return result;
  }

  template <int dim>
  void FaultStokesCoupling<dim>::accept_step(LinearAlgebra::BlockVector &solution)
  {
    const double dt = this->get_timestep();
    if (dt > 0.)
      {
        // With constant coefficients and cell-resolved Q1 rates the endpoint
        // is DGQ2 on each cell. Evaluation at DG nodes is its exact projection.
        const auto &system = this->get_fe();
        const auto &intro = this->introspection();
        const Quadrature<dim> nodes(stress_element().get_unit_support_points());
        FEValues<dim> fe(this->get_mapping(),system,nodes,update_gradients | update_quadrature_points);
        std::vector<Stress> strain(nodes.size());
        std::vector<types::global_dof_index> indices(system.n_dofs_per_cell());
        std::array<LinearAlgebra::Vector,3> owned;
        for (unsigned int c=0; c<3; ++c)
          owned[c].reinit(solution.block(intro.block_indices.compositional_fields[material.stress_fields[c]]).locally_owned_elements(),this->get_mpi_communicator());
        for (const auto &cell : this->get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              fe.reinit(cell);
              cell->get_dof_indices(indices);
              fe[intro.extractors.velocities].get_function_symmetric_gradients(solution,strain);
              for (unsigned int q=0; q<nodes.size(); ++q)
                {
                  const Stress stress = stress_offset(cell,fe.quadrature_point(q))
                                        + material.viscous_stress(strain[q]);
                  for (unsigned int c=0; c<3; ++c)
                    {
                      const auto local = system.component_to_system_index(intro.component_indices.compositional_fields[material.stress_fields[c]],q);
                      const auto block = intro.block_indices.compositional_fields[material.stress_fields[c]];
                      owned[c][indices[local]-solution.get_block_indices().block_start(block)] = stress.access_raw_entry(c);
                    }
                }
            }
        for (unsigned int c=0; c<3; ++c)
          {
            owned[c].compress(VectorOperation::insert);
            solution.block(intro.block_indices.compositional_fields[material.stress_fields[c]]) = owned[c];
          }
      }
    material.fault.accept(trial_rates,dt,material.friction);
  }

  template <int dim>
  std::vector<double> FaultStokesCoupling<dim>::solve_rates(const LinearAlgebra::BlockVector &iterate) const
  {
    const auto points = sample_bulk(iterate);
    const auto &accepted = material.fault.labels();
    const unsigned int n = accepted.size();
    std::vector<double> rates(n);
    for (unsigned int j=0; j<n; ++j)
      rates[j] = accepted[j].rate;

    // The derivative of the localized shear stress with respect to V is
    // -2 kappa (h/I) S:S = -kappa h/I for S=sym(t x n).
    const double stiffness = material.kappa*material.fault.localization();
    for (unsigned int iteration=0; iteration<100; ++iteration)
      {
        std::vector<double> rhs(n), diagonal(n), upper(n-1), scale(n);
        for (const auto &point : points)
          {
            const unsigned int j = point.foot.left;
            const double N[2] = {1.-point.foot.xi,point.foot.xi};
            const double speed = point.foot.interpolate(rates[j],rates[j+1]);
            const double age = point.foot.interpolate(accepted[j].age,accepted[j+1].age);
            const auto resistance = material.friction.resistance(speed,age,point.compression);
            const double residual = point.driving_shear - stiffness*speed - resistance.traction;
            const double derivative = stiffness + resistance.speed_derivative;
            for (unsigned int i=0; i<2; ++i)
              {
                const double test = point.weight*N[i];
                rhs[j+i] += test*residual;
                diagonal[j+i] += test*N[i]*derivative;
                scale[j+i] += test*std::max(std::abs(point.driving_shear),std::abs(resistance.traction));
              }
            upper[j] += point.weight*N[0]*N[1]*derivative;
          }

        for (auto *row :
             {
               &rhs,&diagonal,&upper,&scale
             })
          Utilities::MPI::sum(*row,this->get_mpi_communicator(),*row);
        double residual = 0.;
        for (unsigned int j=0; j<n; ++j)
          {
            AssertThrow(scale[j] > 0.,ExcMessage("A stationary fault label has no resolved bulk support."));
            residual = std::max(residual,std::abs(rhs[j])/scale[j]);
          }
        if (residual <= 64.*std::numeric_limits<double>::epsilon())
          return rates;

        // Eliminate the symmetric positive tridiagonal Q1 tangent in place.
        for (unsigned int j=1; j<n; ++j)
          {
            AssertThrow(diagonal[j-1] > 0.,ExcMessage("Singular stationary fault tangent."));
            const double multiplier = upper[j-1]/diagonal[j-1];
            diagonal[j] -= multiplier*upper[j-1];
            rhs[j] -= multiplier*rhs[j-1];
          }
        AssertThrow(diagonal.back() > 0.,ExcMessage("Singular stationary fault tangent."));
        rhs.back() /= diagonal.back();
        for (unsigned int j=n-1; j>0; --j)
          rhs[j-1] = (rhs[j-1]-upper[j-1]*rhs[j])/diagonal[j-1];

        double fraction = 1.;
        for (unsigned int j=0; j<n; ++j)
          if (rates[j]+rhs[j] < material.minimum_speed)
            fraction = std::min(fraction,.9*(rates[j]-material.minimum_speed)/(-rhs[j]));
        if (fraction <= 0.)
          {
            this->get_pcout() << "Stationary fault Newton correction reached the minimum admissible speed." << std::endl;
            throw ExcNonlinearSolverNoConvergence();
          }
        for (unsigned int j=0; j<n; ++j)
          rates[j] += fraction*rhs[j];
      }
    this->get_pcout() << "Stationary fault Q1 balance did not converge in 100 iterations." << std::endl;
    throw ExcNonlinearSolverNoConvergence();
  }


#define INSTANTIATE(dim) template class FaultStokesCoupling<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
