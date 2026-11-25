/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/boundary_velocity/interface.h>
#include <aspect/simulator/solver/dg_stress_transport.h>
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
    signals.set_assemblers.connect([this](const SimulatorAccess<dim> &sim, Assemblers::Manager<dim> &assemblers)
    {
      const bool anchored = has_normal_traction_boundary(sim);
      AssertThrow(material.normal_stress.uses_adiabat() || anchored,
                  ExcMessage("Fault friction requires a pressure datum from a nonperiodic box face "
                             "with unconstrained normal velocity (prescribed or natural normal traction)."));
      if (material.normal_stress.uses_adiabat())
        AssertThrow(anchored == (sim.get_parameters().pressure_normalization=="no"),
                    ExcMessage("Use a normal-traction boundary with no pressure normalization, or normalize "
                               "the bulk pressure in a closed velocity box with adiabatic fault friction."));
      // For the 3-D deviator in plane strain the velocity matrix needs
      // -2*kappa/3 div(u) div(v), even with incompressible continuity.
      AssertThrow(sim.get_prescribed_dilation_manager().get_active_plugin_names().empty(),
                  ExcMessage("Phase field fault requires incompressible continuity without prescribed dilation."));
      if (!sim.get_parameters().use_full_A_block_preconditioner)
        assemblers.stokes_preconditioner.push_back(std::make_unique<Assemblers::StokesCompressiblePreconditioner<dim>>());
      assemblers.stokes_system.push_back(std::make_unique<Assemblers::StokesCompressibleStrainRateViscosityTerm<dim>>());
      assemblers.stokes_system.push_back(std::make_unique<Assemblers::FaultStressLoad<dim>>());
    });
    signals.prepare_stokes_picard_iteration.connect([this](const LinearAlgebra::BlockVector &,bool &rebuild)
    {
      if (this->get_nonlinear_iteration()==0) rebuild=true; // kappa depends on the attempt duration.
    });
    signals.post_accept_timestep.connect(std::bind(&FaultStokesCoupling::accept_step,this,std::placeholders::_1));
  }

  template <int dim>
  void FaultStokesCoupling<dim>::connect_section_balance()
  {
    auto &signals = this->get_signals();
    using namespace std::placeholders;
    signals.prepare_stokes_picard_iteration.connect(std::bind(&FaultStokesCoupling::prepare_iteration,this,_1,_2));
    signals.check_stokes_picard_convergence.connect(std::bind(&FaultStokesCoupling::check_convergence,this,_1,_2));
  }

  template <int dim>
  void FaultStokesCoupling<dim>::prepare_iteration(const LinearAlgebra::BlockVector &iterate, bool &)
  {
    material.bind_candidate(iterate);
    trial_rates = solve_rates(iterate);
    if (material.interior_seed) project_reaction(iterate,false);
  }

  template <int dim>
  void FaultStokesCoupling<dim>::check_convergence(const LinearAlgebra::BlockVector &solution, bool &converged)
  {
    const auto rates_agree = [](const std::vector<double> &a, const std::vector<double> &b)
    {
      for (unsigned int j=0; j<a.size(); ++j)
        if (std::abs(a[j]-b[j]) > 1.e-8*std::max(a[j],b[j]))
          return false;
      return true;
    };
    auto returned_rates = solve_rates(solution);
    converged &= rates_agree(returned_rates,trial_rates);
    trial_rates.swap(returned_rates);
    if (material.interior_seed)
      {
        // Compare the returned Q8 moments to those used in the last assembly.
        // Keep this stress/rate pair for publication if all vetoes pass.
        const double change = project_reaction(solution,true);
        if (change > this->get_parameters().nonlinear_tolerance)
          {
            this->get_pcout() << "      Relative change in returned fault stress projection: " << change << std::endl;
            converged = false;
          }
        if (converged && this->get_timestep()>0.)
          converged &= rates_agree(solve_rates(solution),trial_rates);
      }
  }

  template <int dim>
  const FiniteElement<dim> &FaultStokesCoupling<dim>::stress_element() const
  {
    return this->get_fe().base_element(this->introspection().base_elements.compositional_fields[material.stress_fields[0]]);
  }

  template <int dim>
  void FaultStokesCoupling<dim>::prepare_parent()
  {
    const double physical_dt = this->get_timestep();
    trial_rates.clear();
    for (const auto &label : material.section().trace.labels())
      trial_rates.push_back(label.rate);

    const QGauss<dim> rule(8);
    const auto &dg = stress_element();
    FullMatrix<double> projector(dg.n_dofs_per_cell(),rule.size());
    FETools::compute_projection_from_quadrature_points_matrix(dg,rule,rule,projector);
    FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_values | update_gradients | update_quadrature_points);
    const auto &fields = this->get_old_solution();
    Transport::CellStress transported;
    FullMatrix<double> basis;
    if constexpr (dim == 2)
      if (physical_dt > 0. && material.inflow_stress)
        {
          TimerOutput::Scope timer(this->get_computing_timer(),"Transport DG stress");
          transported = Transport::advect_stress(*this,material.stress_fields,fields,material.exterior_stress(),
                                                 this->get_time()-physical_dt,physical_dt);
          basis.reinit(rule.size(),dg.n_dofs_per_cell());
          FETools::compute_interpolation_to_quadrature_points_matrix(dg,rule,basis);
        }
    const auto &extract = this->introspection().extractors;
    std::vector<Tensor<2,dim>> gradients(rule.size());
    std::array<std::vector<double>,3> components;
    for (auto &v : components)
      v.resize(rule.size());
    std::vector<Stress> rotated(rule.size());
    std::vector<double> phase(rule.size());

    objective_parent.clear();
    reaction.clear();
    for (const auto &cell : this->get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe.reinit(cell);
          fe[extract.velocities].get_function_gradients(fields,gradients);
          fe[extract.compositional_fields[material.phase_field]].get_function_values(fields,phase);
          if (transported.empty())
            for (unsigned int c=0; c<3; ++c)
              fe[extract.compositional_fields[material.stress_fields[c]]].get_function_values(fields,components[c]);
          else
            {
              const auto &coefficients = transported.at(cell->id());
              for (unsigned int c=0; c<3; ++c)
                for (unsigned int q=0; q<rule.size(); ++q)
                  {
                    components[c][q] = 0.;
                    for (unsigned int i=0; i<coefficients.size(); ++i)
                      components[c][q] += basis(q,i)*coefficients[i].access_raw_entry(c);
                  }
            }
          for (unsigned int q=0; q<rule.size(); ++q)
            {
              Stress parent;
              for (unsigned int c=0; c<3; ++c)
                parent.access_raw_entry(c) = components[c][q];
              Tensor<2,dim> slip;
              if (physical_dt>0.)
                {
                  const auto transfer = material.accepted->profile.sample(fe.quadrature_point(q),phase[q]);
                  slip = material.accepted->profile.slip_gradient(transfer, transfer.rate(trial_rates));
                }
              rotated[q] = material.rotate_parent(parent,gradients[q],slip,physical_dt);
            }
          auto &coefficients = objective_parent[cell->id()];
          coefficients.resize(dg.n_dofs_per_cell());
          FETools::compute_projection_from_quadrature_points(projector,rotated,coefficients);
        }

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
    std::vector<double> pressure(rule.size()), phase(rule.size());
    std::vector<BulkPoint> samples;
    bool locally_closed = true;
    Point<dim> invalid_position;
    double invalid_pressure = 0.;
    for (const auto &cell : this->get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe.reinit(cell);
          fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(iterate,strain);
          fe[this->introspection().extractors.pressure].get_function_values(iterate,pressure);
          fe[this->introspection().extractors.compositional_fields[material.phase_field]].get_function_values(iterate,phase);
          const auto &parent = objective_parent.at(cell->id());
          for (unsigned int q=0; q<rule.size(); ++q)
            {
              const auto &x=fe.quadrature_point(q);
              const auto transfer = material.section().profile.sample(x,phase[q]);
              if (transfer.chi==0.) continue;
              const auto &frame = material.section().trace.segment(transfer.foot.first_vertex);
              Stress stress = material.viscous_stress(strain[q],x);
              if (material.interior_seed && this->get_timestep()>0. && !reaction.empty())
                {
                  const auto &frozen=reaction.at(cell->id());
                  const double k=material.stiffness(x);
                  for (unsigned int i=0; i<parent.size(); ++i)
                    stress+=basis(q,i)*(frozen.stress[i]-material.viscous_stress(frozen.strain[i],x));
                  stress+=2.*k*symmetrize(material.section().profile.slip_gradient(transfer,transfer.rate(reaction_rates)));
                }
              else
                for (unsigned int i=0; i<parent.size(); ++i)
                  stress+=basis(q,i)*parent[i]*material.relaxation(x);
              const auto &foot=transfer.foot;
              const Point<dim> position = material.section().trace.labels()[foot.first_vertex].position
                                          + (foot.xi*frame.length)*frame.tangent;
              const double compression = material.normal_compression(position,pressure[q],frame.normal*(stress*frame.normal));
              if (!(std::isfinite(compression) && compression>0.))
                {
                  locally_closed = false;
                  invalid_position = position;
                  invalid_pressure = compression;
                }
              const double shear=foot.interpolate(material.initial_shear[foot.first_vertex],material.initial_shear[foot.first_vertex+1]);
              samples.push_back({transfer,fe.JxW(q),shear + stress*symmetrize(frame.dyad),compression,material.stiffness(fe.quadrature_point(q))});
            }
        }
    // The extrapolated Picard predictor can open an otherwise closed fault.
    // All ranks reject this trial before evaluating friction or staging rates.
    if (!Utilities::MPI::min(static_cast<unsigned int>(locally_closed),this->get_mpi_communicator()))
      {
        if (material.normal_stress.uses_adiabat())
          {
            const auto comm = this->get_mpi_communicator();
            const auto owner = Utilities::MPI::min(locally_closed ? numbers::invalid_unsigned_int :
                                                   Utilities::MPI::this_mpi_process(comm),comm);
            std::ostringstream message;
            message << "Invalid adiabatic fault compression " << invalid_pressure << " Pa at trace position "
                    << invalid_position << " m. Friction requires finite positive compression.";
            const auto diagnostic = Utilities::MPI::broadcast(comm,message.str(),owner);
            AssertThrow(false,ExcMessage(diagnostic));
          }
        this->get_pcout() << "Fault trial requires finite, positive normal compression." << std::endl;
        throw ExcNonlinearSolverNoConvergence();
      }
    return samples;
  }

  template <int dim>
  typename FaultStokesCoupling<dim>::Stress
  FaultStokesCoupling<dim>::stress_offset(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                          const Point<dim> &position, const double phase) const
  {
    const auto unit = this->get_mapping().transform_real_to_unit_cell(cell,position);
    if (!reaction.empty())
      {
        // For Cartesian Q2 velocity the test gradient lies in DGQ2. Integrating
        // its product with the Q8 mass projection at native Q3 gives exactly
        // the Q8 stress action. Keep the Picard viscous block implicit.
        Stress offset;
        const auto &frozen=reaction.at(cell->id());
        for (unsigned int i=0; i<frozen.stress.size(); ++i)
          offset+=stress_element().shape_value(i,unit)*(frozen.stress[i]-material.viscous_stress(frozen.strain[i],position));
        return offset;
      }
    const auto transfer = material.section().profile.sample(position,phase);
    Stress result = -2.*material.stiffness(position)*symmetrize(material.section().profile.slip_gradient(transfer,transfer.rate(trial_rates)));
    const auto &parent = objective_parent.at(cell->id());
    for (unsigned int i=0; i<parent.size(); ++i)
      result += material.relaxation(position)*stress_element().shape_value(i,unit)*parent[i];
    return result;
  }

  template <int dim>
  double FaultStokesCoupling<dim>::project_reaction(const LinearAlgebra::BlockVector &iterate,
                                                    const bool compare_with_assembly)
  {
    const QGauss<dim> rule(8);
    const auto &dg=stress_element();
    FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_values|update_gradients|update_quadrature_points);
    FullMatrix<double> projection(dg.n_dofs_per_cell(),rule.size()),basis(rule.size(),dg.n_dofs_per_cell());
    FETools::compute_projection_from_quadrature_points_matrix(dg,rule,rule,projection);
    FETools::compute_interpolation_to_quadrature_points_matrix(dg,rule,basis);
    const auto &ex=this->introspection().extractors;
    std::vector<double> phase(rule.size());
    std::vector<Stress> strain(rule.size()), physical(rule.size());
    double difference=0., scale=1.;
    for (const auto &cell:this->get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe.reinit(cell);
          fe[ex.velocities].get_function_symmetric_gradients(iterate,strain);
          fe[ex.compositional_fields[material.phase_field]].get_function_values(iterate,phase);
          const auto &parent=objective_parent.at(cell->id());
          for (unsigned int q=0; q<rule.size(); ++q)
            {
              const auto &x=fe.quadrature_point(q);
              const auto transfer=material.section().profile.sample(x,phase[q]);
              const auto slip=material.section().profile.slip_gradient(transfer,transfer.rate(trial_rates));
              Stress anchor;
              for (unsigned int i=0; i<parent.size(); ++i) anchor+=basis(q,i)*parent[i];
              physical[q]=material.relaxation(x)*anchor+material.viscous_stress(strain[q],x)-2.*material.stiffness(x)*symmetrize(slip);
            }
          auto &frozen=reaction[cell->id()];
          std::vector<Stress> assembled;
          if (compare_with_assembly)
            assembled.swap(frozen.stress);
          frozen.strain.resize(parent.size());
          frozen.stress.resize(parent.size());
          FETools::compute_projection_from_quadrature_points(projection,strain,frozen.strain);
          FETools::compute_projection_from_quadrature_points(projection,physical,frozen.stress);
          if (compare_with_assembly)
            for (unsigned int i=0; i<parent.size(); ++i)
              {
                scale=std::max(scale,frozen.stress[i].norm());
                difference=std::max(difference,(assembled[i]-frozen.stress[i]).norm());
              }
        }
    const auto comm=this->get_mpi_communicator();
    reaction_rates=trial_rates;
    return compare_with_assembly ? Utilities::MPI::max(difference,comm)/Utilities::MPI::max(scale,comm) : 0.;
  }

  template <int dim>
  void FaultStokesCoupling<dim>::accept_step(LinearAlgebra::BlockVector &solution)
  {
    const double dt = this->get_timestep();
    if (dt > 0.)
      {
        // Project the complete physical response: localized slip and cut Q1
        // knots generally lie outside DGQ2. The Cartesian cell Jacobian is constant.
        const auto &system = this->get_fe();
        const auto &intro = this->introspection();
        const QGauss<dim> rule(8);
        FullMatrix<double> projector(stress_element().n_dofs_per_cell(),rule.size());
        FETools::compute_projection_from_quadrature_points_matrix(stress_element(),rule,rule,projector);
        FEValues<dim> fe(this->get_mapping(),system,rule,update_values | update_gradients | update_quadrature_points);
        std::vector<Stress> strain(rule.size()), physical(rule.size()), coefficients(stress_element().n_dofs_per_cell());
        std::vector<double> phase(rule.size());
        std::vector<types::global_dof_index> indices(system.n_dofs_per_cell());
        std::array<LinearAlgebra::Vector,3> owned;
        for (unsigned int c=0; c<3; ++c)
          owned[c].reinit(solution.block(intro.block_indices.compositional_fields[material.stress_fields[c]]).locally_owned_elements(),this->get_mpi_communicator());
        for (const auto &cell : this->get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const auto cut=material.interior_seed ? Quadrature<dim>() : material.section().trace.cut_quadrature(cell->vertex(0),cell->vertex(GeometryInfo<dim>::vertices_per_cell-1));
              const auto project=[&](FEValues<dim> &values, const FullMatrix<double> &projection)
              {
                values.reinit(cell);
                strain.resize(values.n_quadrature_points);
                phase.resize(values.n_quadrature_points);
                physical.resize(values.n_quadrature_points);
                values[intro.extractors.velocities].get_function_symmetric_gradients(solution,strain);
                values[intro.extractors.compositional_fields[material.phase_field]].get_function_values(solution,phase);
                for (unsigned int q=0; q<values.n_quadrature_points; ++q)
                  physical[q]=material.endpoint_stress(cell,values.quadrature_point(q),phase[q],strain[q]);
                FETools::compute_projection_from_quadrature_points(projection,physical,coefficients);
              };
              if (material.interior_seed)
                coefficients=reaction.at(cell->id()).stress;
              else if (cut.size())
                {
                  FEValues<dim> cut_fe(this->get_mapping(),system,cut,fe.get_update_flags());
                  FullMatrix<double> cut_projector(coefficients.size(),cut.size());
                  FETools::compute_projection_from_quadrature_points_matrix(stress_element(),rule,cut,cut_projector);
                  project(cut_fe,cut_projector);
                }
              else
                project(fe,projector);
              cell->get_dof_indices(indices);
              for (unsigned int q=0; q<coefficients.size(); ++q)
                {
                  const auto &stress=coefficients[q];
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
    material.accept_section(trial_rates,dt);
  }

  template <int dim>
  void FaultStokesCoupling<dim>::BulkPoint::add_balance(const double speed, const MaterialModel::Rheology::FaultTraction &resistance, std::vector<double> &rhs,
                                                        std::vector<double> &diagonal, std::vector<double> &upper, std::vector<double> &scale) const
  {
    const auto j=transfer.foot.first_vertex;
    const double N[2]= {1.-transfer.foot.xi,transfer.foot.xi};
    // One chi enters local slip stress, another enters the virtual-work measure.
    const double slip_stiffness=stiffness*transfer.chi;
    transfer.add_load(volume*(driving_shear-slip_stiffness*speed-resistance.traction),rhs);
    transfer.add_load(volume*std::max(std::abs(driving_shear),std::abs(resistance.traction)),scale);
    const double tangent=volume*transfer.chi*(slip_stiffness+resistance.speed_derivative);
    for (unsigned int i=0; i<2; ++i)
      diagonal[j+i]+=tangent*N[i]*N[i];
    upper[j]+=tangent*N[0]*N[1];
  }

  template <int dim>
  std::vector<double> FaultStokesCoupling<dim>::solve_rates(const LinearAlgebra::BlockVector &iterate) const
  {
    const auto points = sample_bulk(iterate);
    const auto &accepted = material.section().trace.labels();
    const unsigned int n = accepted.size();
    std::vector<double> rates(n);
    for (unsigned int j=0; j<n; ++j)
      rates[j] = accepted[j].rate;

    for (unsigned int iteration=0; iteration<100; ++iteration)
      {
        std::vector<double> rhs(n), diagonal(n), upper(n-1), scale(n);
        for (const auto &point : points)
          {
            const auto &foot=point.transfer.foot;
            const double speed=point.transfer.rate(rates);
            const auto resistance=material.friction->at_speed(foot.first_vertex,foot.xi,speed,point.compression);
            point.add_balance(speed,resistance,rhs,diagonal,upper,scale);
          }

        for (auto *row :
             {
               &rhs,&diagonal,&upper,&scale
             })
          Utilities::MPI::sum(*row,this->get_mpi_communicator(),*row);
        double residual = 0.;
        for (unsigned int j=0; j<n; ++j)
          {
            AssertThrow(scale[j] > 0.,ExcMessage("A material fault label has no resolved bulk support."));
            residual = std::max(residual,std::abs(rhs[j])/scale[j]);
          }
        if (residual <= 64.*std::numeric_limits<double>::epsilon())
          return rates;

        // Eliminate the symmetric positive tridiagonal Q1 tangent in place.
        for (unsigned int j=1; j<n; ++j)
          {
            AssertThrow(diagonal[j-1] > 0.,ExcMessage("Singular fault section tangent."));
            const double multiplier = upper[j-1]/diagonal[j-1];
            diagonal[j] -= multiplier*upper[j-1];
            rhs[j] -= multiplier*rhs[j-1];
          }
        AssertThrow(diagonal.back() > 0.,ExcMessage("Singular fault section tangent."));
        rhs.back() /= diagonal.back();
        for (unsigned int j=n-1; j>0; --j)
          rhs[j-1] = (rhs[j-1]-upper[j-1]*rhs[j])/diagonal[j-1];

        double fraction = 1.;
        for (unsigned int j=0; j<n; ++j)
          if (rates[j]+rhs[j] < material.minimum_speed)
            fraction = std::min(fraction,.9*(rates[j]-material.minimum_speed)/(-rhs[j]));
        if (fraction <= 0.)
          {
            this->get_pcout() << "Fault section Newton correction reached the minimum admissible speed." << std::endl;
            throw ExcNonlinearSolverNoConvergence();
          }
        for (unsigned int j=0; j<n; ++j)
          rates[j] += fraction*rhs[j];
      }
    this->get_pcout() << "Fault Q1 section balance did not converge in 100 iterations." << std::endl;
    throw ExcNonlinearSolverNoConvergence();
  }


#define INSTANTIATE(dim) template class FaultStokesCoupling<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
