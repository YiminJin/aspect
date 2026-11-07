/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/phase_field.h>
#include <aspect/phase_field/dg_operator.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools_evaluate.h>
#include <iomanip>

namespace aspect
{
  template <int dim>
  void PhaseFieldHandler<dim>::limit_dg_phase(LinearAlgebra::BlockVector &state) const
  {
    AssertThrow(uses_dg_phase(), ExcMessage("DG phase limiter requires the native DG field."));
    const auto &variable = this->introspection().variable("phase_field");
    const auto &sets = this->introspection().index_sets;
    PhaseField::DGOperator<dim> op(this->get_mapping(), this->get_dof_handler(), variable.first_component_index,
                                   state.get_block_indices().block_start(variable.block_index),
                                   sets.system_partitioning[variable.block_index], sets.system_relevant_partitioning[variable.block_index],
                                   this->get_mpi_communicator());
    const auto &material = dynamic_cast<const MaterialModel::PhaseFieldModel<dim>&>(this->get_material_model());
    const auto range = material.get_phase_field_range();
    typename PhaseField::DGOperator<dim>::TransportStatistics stats;
    op.limit(state.block(variable.block_index), range.first, range.second, stats);
    this->get_pcout() << "   DG phase initialization: limited cells=" << stats.limited_cells
                      << ", minimum alpha=" << stats.minimum_alpha << std::endl;
  }

  template <int dim>
  void PhaseFieldHandler<dim>::transport_dg_phase(const LinearAlgebra::BlockVector &accepted,
                                                  const LinearAlgebra::BlockVector &velocity,
                                                  double time, double dt, LinearAlgebra::BlockVector &candidate) const
  {
    TimerOutput::Scope timer(this->get_computing_timer(), "Phase: DG material transport");
    AssertThrow(uses_dg_phase(), ExcMessage("DG phase transport requires the native DG field."));
    const auto &variable = this->introspection().variable("phase_field");
    const auto &sets = this->introspection().index_sets;
    PhaseField::DGOperator<dim> op(this->get_mapping(), this->get_dof_handler(), variable.first_component_index,
                                   accepted.get_block_indices().block_start(variable.block_index),
                                   sets.system_partitioning[variable.block_index], sets.system_relevant_partitioning[variable.block_index],
                                   this->get_mpi_communicator());
    op.assemble_transport([&](const auto & cell, const auto & points, auto & speed, auto & divergence)
    {
      std::vector<Point<dim>> unit;
      for (const auto &point : points) unit.push_back(this->get_mapping().transform_real_to_unit_cell(cell, point));
      FEValues<dim> fe(this->get_mapping(), this->get_fe(), Quadrature<dim>(unit), update_values | update_gradients);
      fe.reinit(cell);
      fe[this->introspection().extractors.velocities].get_function_values(velocity, speed);
      std::vector<Tensor<2, dim>> gradients(points.size());
      fe[this->introspection().extractors.velocities].get_function_gradients(velocity, gradients);
      for (unsigned int q = 0; q < points.size(); ++q) divergence[q] = trace(gradients[q]);
    });
    const auto &material = dynamic_cast<const MaterialModel::PhaseFieldModel<dim>&>(this->get_material_model());
    const auto range = material.get_phase_field_range();
    // Compact-profile benchmark convention: zero exterior phase on every
    // incoming face. Stress has its separate explicitly supplied exterior law.
    const auto stats = op.transport(accepted.block(variable.block_index), candidate.block(variable.block_index),
                                    [](const Point<dim> &, double)
    {
      return 0.;
    }, time, dt, range.first, range.second,
    this->get_parameters().use_limiter_for_discontinuous_phase_solution);
    this->get_pcout() << std::setprecision(17) << "   DG phase transport: substeps=" << stats.substeps
                      << ", limited cells=" << stats.limited_cells << ", minimum alpha=" << stats.minimum_alpha
                      << ", initial integral=" << stats.initial_integral << ", final integral=" << stats.final_integral
                      << ", source=" << stats.integrated_source << ", boundary flux=" << stats.integrated_boundary_flux
                      << ", balance error=" << stats.balance_error << std::endl;
  }

  template <int dim>
  std::vector<double> PhaseFieldHandler<dim>::phase_point_values(
    const Utilities::MPI::RemotePointEvaluation<dim> &lookup, const LinearAlgebra::BlockVector &state) const
  {
    const auto c = this->introspection().variable("phase_field").first_component_index;
    if (!uses_dg_phase())
      return VectorTools::point_values<1>(lookup, this->get_dof_handler(), state, VectorTools::EvaluationFlags::avg, c);
    using Sample = Tensor < 1, dim + 1 >;
    const auto samples = lookup.template evaluate_and_process<Sample>(
                           [&](const ArrayView<Sample> &values, const auto & data)
    {
      for (const auto cell_index : data.cell_indices())
        {
          const auto cell = data.get_active_cell_iterator(cell_index)->as_dof_handler_iterator(this->get_dof_handler());
          const auto points = data.get_unit_points(cell_index);
          const std::vector<Point<dim>> unit(points.begin(), points.end());
          FEValues<dim> fe(this->get_mapping(), this->get_fe(), Quadrature<dim>(unit), update_values);
          fe.reinit(cell);
          std::vector<double> phase(unit.size());
          fe[FEValuesExtractors::Scalar(c)].get_function_values(state, phase);
          auto output = data.get_data_view(cell_index, values);
          for (unsigned int q = 0; q < phase.size(); ++q)
            {
              output[q][dim] = phase[q];
              for (unsigned int d = 0; d < dim; ++d) output[q][d] = cell->vertex(0)[d];
            }
        }
    });
    const auto &ptr = lookup.get_point_ptrs();
    std::vector<double> result(ptr.size() - 1, 0.);
    for (unsigned int i = 0; i < result.size(); ++i)
      if (ptr[i] < ptr[i + 1])
        {
          unsigned int selected = ptr[i];
          for (unsigned int j = ptr[i] + 1; j < ptr[i + 1]; ++j)
            for (unsigned int d = 0; d < dim; ++d)
              if (samples[j][d] != samples[selected][d])
                {
                  if (samples[j][d] > samples[selected][d]) selected = j;
                  break;
                }
          result[i] = samples[selected][dim];
        }
    return result;
  }

#define INSTANTIATE(dim) \
  template std::vector<double> PhaseFieldHandler<dim>::phase_point_values(const Utilities::MPI::RemotePointEvaluation<dim> &,const LinearAlgebra::BlockVector &) const; \
  template void PhaseFieldHandler<dim>::limit_dg_phase(LinearAlgebra::BlockVector &) const; \
  template void PhaseFieldHandler<dim>::transport_dg_phase(const LinearAlgebra::BlockVector &,const LinearAlgebra::BlockVector &,double,double,LinearAlgebra::BlockVector &) const;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
