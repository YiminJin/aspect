/*
  Copyright (C) 2025 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
 */

#include <aspect/phase_field.h>
#include <aspect/utilities.h>
#include <aspect/particle/particle_domain.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    PhaseFieldInputs<dim>::PhaseFieldInputs(const unsigned int n_points)
      : phase_field_values(n_points, numbers::signaling_nan<double>())
      , phase_field_gradients(n_points, numbers::signaling_nan<Tensor<1, dim>>())
    {}



    template <int dim>
    void
    PhaseFieldInputs<dim>::fill(const LinearAlgebra::BlockVector &solution,
                                const FEValuesBase<dim>          &fe_values,
                                const Introspection<dim>         &introspection)
    {
      AssertDimension(phase_field_values.size(), fe_values.n_quadrature_points);

      const FEValuesExtractors::Scalar &extractor = introspection.variable("phase_field").extractor_scalar();
      fe_values[extractor].get_function_values(solution, phase_field_values);
      fe_values[extractor].get_function_gradients(solution, phase_field_gradients);
    }
  }



  namespace PhaseField
  {
    /*------------------------ GeometricFunction ---------------------------*/

    GeometricFunction::GeometricFunction(const double xi_)
      : xi(xi_)
    {
      AssertThrow(xi >= 0.0 && xi <= 2.0,
                  ExcMessage("Parameter $\\xi$ in the geometric function must be "
                             "in the range of [0,2]."));
    }


    
    double
    GeometricFunction::value(const double d) const
    {
      const double dc = std::min(1.0, std::max(0.0, d));
      return dc * (xi + (1.0 - xi) * dc);
    }



    double
    GeometricFunction::first_derivative(const double d) const
    {
      const double dc = std::min(1.0, std::max(0.0, d));
      return xi + 2.0 * (1.0 - xi) * dc;
    }



    double
    GeometricFunction::second_derivative(const double /*d*/) const
    {
      return 2.0 * (1.0 - xi);
    }



    /*------------------------ DegradationFunction ---------------------------*/

    DegradationFunction::
    DegradationFunction(const double p_,
                        const double m_)
      : p(p_)
      , m(m_)
    {
      AssertThrow(m >= 0.0,
                  ExcMessage("Parameter $m$ in the degradation function must be non-negative."));
    }



    double
    DegradationFunction::value(const double d) const
    {
      const double dc = std::min(1.0, std::max(0.0, d));
      const double one_minus_dc_squared = (1.0 - dc) * (1.0 - dc);
      return std::min(1.0, std::max(0.0, one_minus_dc_squared / (one_minus_dc_squared + m * dc * (1.0 + p * dc))));
    }



    double
    DegradationFunction::first_derivative(const double d) const
    {
      // We approimate the first derivative of the degradation function
      // with a finite difference scheme
      const double dc  = std::min(0.9999999, std::max(0.0000001, d));
      const double ddc = dc * 1.e-7;
      return (value(dc + ddc) - value(dc)) / ddc;
    }



    double
    DegradationFunction::second_derivative(const double d) const
    {
      // We approimate the second derivative of the degradation function
      // with a finite difference scheme
      const double dc  = std::min(0.9999999, std::max(0.0000001, d));
      const double ddc = dc * 1.e-7;
      return (value(dc + ddc) + value(dc - ddc) - 2.0 * value(dc)) / (ddc * ddc);
    }
  }


  /*--------------------------- PhaseFieldHandler ---------------------------*/

  template <int dim>
  void
  PhaseFieldHandler<dim>::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Phase field model");
    {
      prm.declare_entry("Length scale", "1000",
                        Patterns::Double(0.),
                        "The length scale that characterizes the width of the fracture zone. "
                        "Units: \\si{meter}.");

      prm.declare_entry("Geometric function type", "AT1",
                        Patterns::Selection("AT1|AT2|CZM"),
                        "Type of the geometric function $\\alpha(d)$. The options are:\n"
                        "AT1: $\\alpha(d) = d$;\n"
                        "AT2: $\\alpha(d) = d^2;\n$"
                        "CZM: $\\alpha(d) = 2d - d^2$.");

      prm.declare_entry("Degradation curvature parameter", "1",
                        Patterns::Double(0),
                        "The curvature parameter $p$ for the Lorentz-type degradation function.\n"
                        "Units: none.");

      prm.declare_entry("Threshold driving forces", "",
                        Patterns::List(Patterns::Double(0.)),
                        "List of the threshold driving forces, $H_t$, for background "
                        "material and compositional fields, for a total of N+1 values, "
                        "where N is the number of all compositional fields or only those "
                        "corresponding to chemical compositions. If this parameter is left "
                        "vacant, then ASPECT will compute the the threshold driving forces "
                        "automatically, which makes the degradation functions reduce to "
                        "the quadratic form $g(d) = (1 - d)^2$.\n"
                        "Units: \\si{\\pascal}.");

      prm.declare_entry("Critical energies", "1.e5",
                        Patterns::Anything(),
                        "List of the critical energies, $G_c$, for background material "
                        "and compositional fields, for a total of N+1 values, where N "
                        "is the number of all compositional fields or only those "
                        "corresponding to chemical compositions. "
                        "Units: \\si{\\joule\\per\\square\\meter}.");
    }
    prm.leave_subsection();
  }



  template <int dim>
  void
  PhaseFieldHandler<dim>::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Phase field model");
    {
      parameters.length_scale                    = prm.get_double("Length scale");
      parameters.degradation_curvature_parameter = prm.get_double("Degradation curvature parameter");
      
      // Initialize the geometric function
      const std::string type = prm.get("Geometric function type");
      if (type == "AT2")
        {
          geometric_function = std::make_unique<PhaseField::GeometricFunction>(0);
          parameters.geometric_normalization_parameter = 2.0;
        }
      else if (type == "AT1")
        {
          geometric_function = std::make_unique<PhaseField::GeometricFunction>(1);
          parameters.geometric_normalization_parameter = 2.666666666666666667;
        }
      else if (type == "CZM")
        {
          geometric_function = std::make_unique<PhaseField::GeometricFunction>(2);
          parameters.geometric_normalization_parameter = numbers::PI;
        }
      else
        AssertThrow(false, ExcNotImplemented());
    }
    prm.leave_subsection();
  }



  template <int dim>
  void PhaseFieldHandler<dim>::initialize()
  {
    AssertThrow(this->introspection().compositional_name_exists("phase_field"),
                ExcMessage("The phase field method requires the compositional fields to include "
                           "a field named `phase_field'"));
    const unsigned int phase_field_index = this->introspection().compositional_index_for_name("phase_field");
    AssertThrow(this->get_parameters().compositional_field_methods[phase_field_index]
                == Parameters<dim>::AdvectionFieldMethod::phase_field,
                ExcMessage("The advection method for the phase field must be set to `phase_field'."));
    AssertThrow(this->get_parameters().composition_degrees[phase_field_index] == 1 &&
                this->get_parameters().use_discontinuous_composition_discretization[phase_field_index] == false,
                ExcMessage("The phase field must be descritized with standard Q1 element."));

    // Get the critical energy release rate and the threshold crack driving force
    // from the material model
    const MaterialModel::PhaseFieldModel<dim> *phase_field_model 
      = dynamic_cast<const MaterialModel::PhaseFieldModel<dim>*>(&this->get_material_model());
    AssertThrow(phase_field_model != nullptr,
                ExcMessage("The phase field method requires the material model to be derived from "
                           "MaterialModel::PhaseFieldModel."));
    const std::vector<double> &Gc = phase_field_model->get_critical_energy_release_rates();
    const std::vector<double> &Ht = phase_field_model->get_threshold_crack_driving_forces();

    // Compute the critical energy densities
    const unsigned int n_comp = Gc.size();
    parameters.critical_energy_densities.resize(n_comp);
    for (unsigned int j = 0; j < n_comp; ++j)
      parameters.critical_energy_densities[j] = Gc[j] / (parameters.geometric_normalization_parameter * parameters.length_scale);

    // Initialize the degradation functions
    degradation_functions.clear();
    for (unsigned int j = 0; j < n_comp; ++j)
      {
        const double m = Gc[j] / (parameters.geometric_normalization_parameter * parameters.length_scale * Ht[j]);
        degradation_functions.push_back(std::make_unique<PhaseField::DegradationFunction>(parameters.degradation_curvature_parameter, m));
      }

    // Find the particle manager handling the crack driving force
    AssertThrow(this->n_particle_managers() > 0,
                ExcMessage("The phase field method requires particles to be included in the model. "
                           "Please add 'particles' to the list of postprocessors."));

    AssertThrow(this->get_parameters().mapped_particle_properties.size() > 0,
                ExcMessage("The phase field method requires a map between compositional fields and particle properties"));

    for (unsigned int i = 0; i < this->n_particle_managers(); ++i)
      {
        const Particle::Manager<dim> &manager = this->get_particle_manager(i);
        if (manager.get_property_manager().get_data_info().fieldname_exists("crack_driving_force"))
          {
            particle_manager = &manager;
            break;
          }
      }
    AssertThrow(particle_manager != nullptr, 
                ExcMessage("The phase field method requires one of the particle sets to include a "
                           "particle property named 'crack_driving_force'."));

    AssertThrow(particle_manager->particle_domains_requested(),
                ExcMessage("The phase field method requires the particle manager to generate "
                           "particle domains."));

    AssertThrow(particle_manager->get_particle_domain_handler().cpdi_data_requested(),
                ExcMessage("The phase field method requires the particle domain handler to "
                           "generate CPDI data."));

    // Initialize the system information
    system_info.initialize(this->introspection(), this->get_parameters(), *particle_manager);
  }



  template <int dim>
  void
  PhaseFieldHandler<dim>::SystemInformation::
  initialize(const Introspection<dim>     &introspection,
             const Parameters<dim>        &parameters,
             const Particle::Manager<dim> &particle_manager)
  {
    const FEVariable<dim> &phase_field_variable = introspection.variable("phase field");
    phase_field_component_index = phase_field_variable.first_component_index;

    // Get the block indices of the coupled system
    block_indices.velocities  = introspection.block_indices.velocities;
    block_indices.pressure    = introspection.block_indices.pressure;
    block_indices.phase_field = phase_field_variable.block_index;

    // Get the index sets of the coupled system
    index_sets.coupled_system_partitioning.resize(introspection.n_blocks);
    index_sets.coupled_system_partitioning[block_indices.velocities]  = introspection.index_sets.system_partitioning[block_indices.velocities];
    index_sets.coupled_system_partitioning[block_indices.pressure]    = introspection.index_sets.system_partitioning[block_indices.pressure];
    index_sets.coupled_system_partitioning[block_indices.phase_field] = introspection.index_sets.system_partitioning[block_indices.phase_field];

    index_sets.coupled_system_relevant_partitioning.resize(introspection.n_blocks);
    index_sets.coupled_system_relevant_partitioning[block_indices.velocities]  = introspection.index_sets.system_relevant_partitioning[block_indices.velocities];
    index_sets.coupled_system_relevant_partitioning[block_indices.pressure]    = introspection.index_sets.system_relevant_partitioning[block_indices.pressure];
    index_sets.coupled_system_relevant_partitioning[block_indices.phase_field] = introspection.index_sets.system_relevant_partitioning[block_indices.phase_field];

    // Get the positions of the particle properties required by the coupled system
    const auto &particle_data_info = particle_manager.get_property_manager().get_data_info();
    particle_data_positions.crack_driving_force = particle_data_info.get_position_by_field_name("crack_driving_force");

    particle_data_positions.chemical_fields.clear();
    for (const unsigned int index : introspection.chemical_composition_field_indices())
      {
        AssertThrow(parameters.compositional_field_methods[index] == Parameters<dim>::AdvectionFieldMethod::particles,
                    ExcMessage("The phase field method requires all the chemical composition fields to be advected "
                               "by particles."));
        const std::string &property_name = parameters.mapped_particle_properties.find(index)->second.first;
        AssertThrow(particle_data_info.fieldname_exists(property_name),
                    ExcMessage("The phase field method requires all the chemical composition fields to be in the "
                               "same particle set as the crack driving force."));
        particle_data_positions.chemical_fields.push_back(particle_data_info.get_position_by_field_name(property_name));
      }
  }



  template <int dim>
  double
  PhaseFieldHandler<dim>::
  crack_surface_density(const double          d,
                        const Tensor<1, dim> &grad_d) const
  {
    return ( geometric_function->value(d) / parameters.length_scale
             + (grad_d * grad_d) * parameters.length_scale
           ) / parameters.geometric_normalization_parameter;
  }



  template <int dim>
  double
  PhaseFieldHandler<dim>::
  energetic_degradation(const double               d,
                        const std::vector<double> &volume_fractions) const
  {
    AssertDimension(volume_fractions.size(), degradation_functions.size());

    double g = 0.0;
    for (unsigned int j = 0; j < volume_fractions.size(); ++j)
      if (volume_fractions[j] > 0.0)
        g += degradation_functions[j]->value(d) * volume_fractions[j];

    return std::min(1.0, std::max(0.0, g));
  }



  template <int dim>
  double
  PhaseFieldHandler<dim>::
  compute_microforce(const double               d,
                     const double               H,
                     const std::vector<double> &volume_fractions) const
  {
    double K = 0.0;
    for (unsigned int j = 0; j < volume_fractions.size(); ++j)
      if (volume_fractions[j] > 0.0)
        {
          const double Gc_over_c0l = parameters.critical_energy_densities[j];
          const double da_dd = geometric_function->first_derivative(d);
          const double dg_dd = degradation_functions[j]->first_derivative(d);

          K += (H * dg_dd + Gc_over_c0l * da_dd) * volume_fractions[j];
        }

    return K;
  }



  template <int dim>
  double
  PhaseFieldHandler<dim>::
  compute_microforce_derivative(const double               d,
                                const double               H,
                                const std::vector<double> &volume_fractions) const
  {
    double dK_dd = 0.0;
    for (unsigned int j = 0; j < volume_fractions.size(); ++j)
      if (volume_fractions[j] > 0.0)
        {
          const double Gc_over_c0l = parameters.critical_energy_densities[j];
          const double d2a_dd2 = geometric_function->second_derivative(d);
          const double d2g_dd2 = degradation_functions[j]->second_derivative(d);

          dK_dd += (H * d2g_dd2 + Gc_over_c0l * d2a_dd2) * volume_fractions[j];
        }

    return dK_dd;
  }



  template <int dim>
  double
  PhaseFieldHandler<dim>::
  compute_microstress_prefactor(const std::vector<double> &volume_fractions) const
  {
    const double l2 = parameters.length_scale * parameters.length_scale;

    double F = 0.0;
    for (unsigned int j = 0; j < volume_fractions.size(); ++j)
      if (volume_fractions[j] > 0.0)
        {
          const double Gc_over_c0l = parameters.critical_energy_densities[j];
          F += (2.0 * Gc_over_c0l * l2) * volume_fractions[j];
        }

    return F;
  }



  template <int dim>
  void
  PhaseFieldHandler<dim>::
  assemble_and_solve(LinearAlgebra::BlockSparseMatrix &system_matrix,
                     LinearAlgebra::BlockVector       &system_rhs,
                     LinearAlgebra::BlockVector       &solution_vector) const
  {
    // Map vertex indices to phase-field DoF indices
    std::vector<types::global_dof_index> vertex_to_dof_indices(this->get_triangulation().n_vertices(),
                                                               numbers::invalid_dof_index);

    for (const auto &cell : this->get_dof_handler().active_cell_iterators())
      if (!cell->is_artificial())
        for (const unsigned int v : cell->vertex_indices())
          {
            const unsigned int vertex_index = cell->vertex_index(v);
            if (vertex_to_dof_indices[vertex_index] == numbers::invalid_dof_index)
              vertex_to_dof_indices[vertex_index] = cell->vertex_dof_index(v, system_info.phase_field_component_index);
          }

    const Particles::ParticleHandler<dim> &particle_handler = particle_manager->get_particle_handler();
    const Particle::ParticleDomainHandler<dim> &particle_domain_handler = particle_manager->get_particle_domain_handler();

    // Initialize the corresponding block of the system matrix and the system rhs
    const unsigned int block_index = system_info.block_indices.phase_field;
    system_matrix.block(block_index, block_index) = 0;
    system_rhs.block(block_index) = 0;

    const LinearAlgebra::BlockVector current_solution = this->get_solution();

    for (const auto &cell : this->get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          for (const auto &particle : particle_handler.particles_in_cell(cell))
            {
              // Get access to the CPDI data
              const auto &particle_domain = particle_domain_handler.get_particle_domain(particle.get_local_index());

              // Stuff for local assembly
              const unsigned int n_dofs = particle_domain.n_relevant_vertices();
              FullMatrix<double> particle_matrix(n_dofs, n_dofs);
              Vector<double> particle_rhs(n_dofs);
              std::vector<types::global_dof_index> particle_dof_indices(n_dofs);

              // Get the weighting functions and the DoF values
              std::vector<double>         phase_field_dof_values(n_dofs);
              std::vector<double>         weighting_function_values(n_dofs);
              std::vector<Tensor<1, dim>> weighting_function_gradients(n_dofs);
              for (unsigned int i = 0; i < n_dofs; ++i)
                {
                  const unsigned int vertex_index = particle_domain.relevant_vertex_index(i);
                  const types::global_dof_index dof_index = vertex_to_dof_indices[vertex_index];
                  particle_dof_indices[i] = dof_index;
                  phase_field_dof_values[i] = current_solution[dof_index];

                  weighting_function_values[i]    = particle_domain.weighting_function_value(i);
                  weighting_function_gradients[i] = particle_domain.weighting_function_gradient(i);
                }

              // Compute the phase field value and gradient, which are assumed to be
              // uniform in this particle domain
              double phase_field_value = 0.0;
              Tensor<1, dim> phase_field_gradient;
              for (unsigned int i = 0; i < n_dofs; ++i)
                {
                  phase_field_value    += phase_field_dof_values[i] * weighting_function_values[i];
                  phase_field_gradient += phase_field_dof_values[i] * weighting_function_gradients[i];
                }

              // Get the driving force and the chemical fields
              const ArrayView<const double> particle_properties = particle.get_properties();
              const double crack_driving_force = particle_properties[system_info.particle_data_positions.crack_driving_force];
              std::vector<double> chemical_field_values(system_info.particle_data_positions.chemical_fields.size());
              for(unsigned int c = 0; c < chemical_field_values.size(); ++c)
                chemical_field_values[c] = particle_properties[system_info.particle_data_positions.chemical_fields[c]];

              // Compute the volume fractions
              const std::vector<double> volume_fractions
                = MaterialModel::MaterialUtilities::compute_composition_fractions(chemical_field_values);

              const double microforce            = compute_microforce(phase_field_value, crack_driving_force, volume_fractions);
              const double microforce_derivative = compute_microforce_derivative(phase_field_value, crack_driving_force, volume_fractions);
              const double microstress_prefactor = compute_microstress_prefactor(volume_fractions);
              const Tensor<1, dim> microstress   = microstress_prefactor * phase_field_gradient;

              const double V_p = particle_domain.volume();
              for (unsigned int i = 0; i < n_dofs; ++i)
                {
                  const double phi_ip               = weighting_function_values[i];
                  const Tensor<1, dim> &grad_phi_ip = weighting_function_gradients[i];

                  particle_rhs(i) += (microforce * phi_ip + microstress * grad_phi_ip) * V_p;

                  for (unsigned int j = 0; j < n_dofs; ++j)
                    {
                      const double phi_jp               = weighting_function_values[j];
                      const Tensor<1, dim> &grad_phi_jp = weighting_function_gradients[j];

                      particle_matrix(i, j) += ( microforce_derivative * (phi_ip * phi_jp)
                                                 + microstress_prefactor * (grad_phi_ip * grad_phi_jp)
                                               ) * V_p;
                    }
                }

              this->get_current_constraints().distribute_local_to_global(particle_matrix,
                                                                         particle_rhs,
                                                                         particle_dof_indices,
                                                                         system_matrix,
                                                                         system_rhs);
            }
        }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    // Set the preconditioner
    LinearAlgebra::PreconditionAMG preconditioner;
    LinearAlgebra::PreconditionAMG::AdditionalData amg_data;

    std::vector<bool> phase_field_component_mask(this->introspection().n_components, false);
    phase_field_component_mask[system_info.phase_field_component_index] = true;
#if DEAL_II_VERSION_GTE(9,7,0)
    amg_data.constant_modes = DoFTools::extract_constant_modes(
                                this->get_dof_handler(),
                                ComponentMask(phase_field_component_masks));
#else
    std::vector<std::vector<bool>> constant_modes;
    DoFTools::extract_constant_modes(
      this->get_dof_handler(),
      ComponentMask(phase_field_component_mask),
      constant_modes);
    amg_data.constant_modes = constant_modes;
#endif

    amg_data.elliptic = true;
    amg_data.higher_order_elements = false;
    amg_data.smoother_sweeps = 2;
    amg_data.aggregation_threshold = 0.02;

    preconditioner.initialize(system_matrix.block(block_index, block_index), amg_data);

    // Create a distributed vector
    LinearAlgebra::BlockVector distributed_solution(this->introspection().index_sets.system_partitioning,
                                                    this->get_mpi_communicator());

    this->get_pcout() << "   Solving for phase field..." << std::flush;

    // TODO: How to apply the hanging node constraints?
    SolverControl solver_control(1000, 1.e-8 * system_rhs.block(block_index).l2_norm());
    SolverCG<LinearAlgebra::Vector> solver(solver_control);

    try
      {
        solver.solve(system_matrix.block(block_index, block_index),
                     distributed_solution.block(block_index),
                     system_rhs.block(block_index),
                     preconditioner);
      }
    catch (const std::exception &exc)
      {
        // if the solver fails, report the error from processor 0 with some additional
        // information about its location, and throw a quiet exception on all other
        // processors
        Utilities::throw_linear_solver_failure_exception("iterative solver for phase field",
                                                         "perform_convected_particle_domain_interpolation()",
                                                         std::vector<SolverControl> {solver_control},
                                                         exc,
                                                         this->get_mpi_communicator());
      }

    this->get_pcout() << solver_control.last_step() << " iterations." << std::endl;

    this->get_current_constraints().distribute(distributed_solution);
    solution_vector.block(block_index) = distributed_solution.block(block_index);
  }



  template <int dim>
  void
  PhaseFieldHandler<dim>::
  make_sparsity_pattern(LinearAlgebra::BlockDynamicSparsityPattern &sp) const
  {
    // Create the vertex-to-cell map
    GridTools::Cache<dim> grid_cache(this->get_triangulation(), this->get_mapping());
    const auto &vertex_to_cell_map = grid_cache.get_vertex_to_cell_map();

    std::vector<types::global_dof_index> coupled_dofs;

    // Loop over the locally owned cells and add the nonzero entries of CPDI system
    for (const auto &cell : this->get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          // All the phase-field DoFs in the one-layer-patch around a cell are
          // possible to be coupled
          std::set<typename Triangulation<dim>::active_cell_iterator> neighboring_cells;
          for (const unsigned int v : cell->vertex_indices())
            {
              const unsigned int vertex_index = cell->vertex_index(v);
              neighboring_cells.insert(vertex_to_cell_map[vertex_index].begin(),
                                       vertex_to_cell_map[vertex_index].end());
            }

          // Since the CPDI method requires the fields to be discretized by FE_Q(1) element,
          // we only need to loop over the vertices and extract the DoFs corresponding to
          // the first CPDI field
          std::set<types::global_dof_index> coupled_dofs;
          for (const auto &neighbor : neighboring_cells)
            {
              typename DoFHandler<dim>::active_cell_iterator dof_cell(&this->get_triangulation(),
                                                                      neighbor->level(),
                                                                      neighbor->index(),
                                                                      &this->get_dof_handler());

              for (const unsigned int v : dof_cell->vertex_indices())
                coupled_dofs.insert(dof_cell->vertex_dof_index(v, system_info.phase_field_component_index));

              this->get_current_constraints().add_entries_local_to_global(std::vector<types::global_dof_index>(coupled_dofs.begin(),
                                                                                                               coupled_dofs.end()),
                                                                          sp, true);
            }
        }
  }
}

// explicit instantiations
namespace aspect
{
#define INSTANTIATE(dim) \
  namespace MaterialModel \
  { \
    template class PhaseFieldInputs<dim>; \
  } \
  template class PhaseFieldHandler<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
