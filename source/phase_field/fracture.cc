/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/phase_field.h>
#include <aspect/phase_field/dg_operator.h>
#include <aspect/reconstructed_fault/endpoint_mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  template <int dim>
  double
  PhaseFieldHandler<dim>::assemble_dg_fracture(const LinearAlgebra::BlockVector &state,
                                             LinearAlgebra::Vector &gradient,
                                             LinearAlgebra::SparseMatrix *hessian) const
  {
    for (const auto a : critical_energy_densities)
      AssertThrow(a == critical_energy_densities[0], ExcMessage("Coupled DG phase currently requires homogeneous fracture coefficients."));
    const double A = critical_energy_densities[0], ell = geometric_function->get_length_scale(), k = 2 * A * ell * ell;
    const auto &variable = this->introspection().phase_field();
    const auto b = variable.block_index, c = variable.first_component_index;
    const auto offset = state.get_block_indices().block_start(b);
    const auto &sets = this->introspection().index_sets;
    const auto comm = this->get_mpi_communicator();
    auto &op = native_operator(state);
    LinearAlgebra::Vector rhs(sets.system_partitioning[b], comm), values(rhs), action(rhs);
    const double penalty = this->get_parameters().discontinuous_penalty;
    if (!native_fracture_matrix || native_fracture_coefficient != k || native_fracture_penalty != penalty)
      {
        native_fracture_matrix = std::make_shared<LinearAlgebra::SparseMatrix>();
        op.reinit_matrix(*native_fracture_matrix);
        op.assemble_sipg(*native_fracture_matrix, rhs, [k](const Point<dim> &)
        {
          return k;
        }, penalty);
        native_fracture_coefficient = k;
        native_fracture_penalty = penalty;
      }
    const auto &matrix = *native_fracture_matrix;
    // The supported physical fracture boundary is homogeneous natural Neumann.
    rhs = 0.;
    const double constant = 0.;
    values = state.block(b);
    matrix.vmult(action, values);
    double energy = .5 * (values * action) - values * rhs + constant;
    action -= rhs;
    gradient.reinit(sets.system_partitioning[b], comm);
    gradient = action;
    if (hessian) hessian->copy_from(matrix);
    const QGauss<dim> rule(5);
    FEValues<dim> fe(this->get_mapping(), this->get_fe(), rule, update_values | update_JxW_values);
    const FEValuesExtractors::Scalar phase(c);
    std::vector<double> phi(rule.size());
    std::vector<types::global_dof_index> indices(this->get_fe().n_dofs_per_cell());
    const auto &element = this->get_fe().base_element(this->get_fe().component_to_base_index(c).first);
    double local_energy = 0.;
    for (const auto &cell : this->get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe.reinit(cell);
          cell->get_dof_indices(indices);
          fe[phase].get_function_values(state, phi);
          for (unsigned int q = 0; q < rule.size(); ++q)
            {
              local_energy += fe.JxW(q) * A * geometric_function->value(phi[q]);
              for (unsigned int a = 0; a < element.n_dofs_per_cell(); ++a)
                {
                  const auto i = this->get_fe().component_to_system_index(c, a);
                  const auto row = indices[i] - offset;
                  const double term = fe.JxW(q) * A * geometric_function->first_derivative(phi[q]) * fe[phase].value(i, q);
                  gradient[row] += term;
                  if (hessian)
                    for (unsigned int d = 0; d < element.n_dofs_per_cell(); ++d)
                      {
                        const auto j = this->get_fe().component_to_system_index(c, d);
                        hessian->add(row, indices[j] - offset, fe.JxW(q) * A * geometric_function->second_derivative(phi[q])
                                                                   * fe[phase].value(i, q) * fe[phase].value(j, q));
                      }
                }
            }
        }
    gradient.compress(VectorOperation::add);
    if (hessian) hessian->compress(VectorOperation::add);
    return energy + Utilities::MPI::sum(local_energy, comm);
  }

  template <int dim>
  typename PhaseFieldHandler<dim>::CellFractureEnergy
  PhaseFieldHandler<dim>::evaluate_dg_fracture(const LinearAlgebra::BlockVector &state,
                                             const bool tangent,
                                             std::map<std::pair<unsigned int, unsigned int>, double> *hessian) const
  {
    AssertThrow(!tangent || hessian, ExcMessage("DG fracture requires a sparse Hessian."));
    LinearAlgebra::Vector gradient;
    LinearAlgebra::SparseMatrix matrix;
    CellFractureEnergy result;
    result.energy = assemble_dg_fracture(state, gradient, tangent ? &matrix : nullptr);
    const auto b = this->introspection().phase_field().block_index;
    const auto comm = this->get_mpi_communicator();
    result.gradient.assign(gradient.size(), 0.);
    result.roundoff_scale.assign(gradient.size(), 0.);
    if (hessian) hessian->clear();
    for (const auto i : gradient.locally_owned_elements())
      {
        result.gradient[i] = gradient[i];
        // Preserve the retained observer's arithmetic allowance: absolute
        // SIPG actions plus absolute local potential terms, not |total force|.
        for (auto entry = native_fracture_matrix->begin(i); entry != native_fracture_matrix->end(i); ++entry)
          result.roundoff_scale[i] += std::abs(entry->value() * state.block(b)[entry->column()]);
        if (tangent)
          for (auto entry = matrix.begin(i); entry != matrix.end(i); ++entry)
            (*hessian)[{i, entry->column()}] = entry->value();
      }
    const auto c = this->introspection().phase_field().first_component_index;
    const auto offset = state.get_block_indices().block_start(b);
    const QGauss<dim> rule(5);
    FEValues<dim> values(this->get_mapping(), this->get_fe(), rule, update_values | update_JxW_values);
    std::vector<double> phi(rule.size());
    std::vector<types::global_dof_index> indices(this->get_fe().n_dofs_per_cell());
    const FEValuesExtractors::Scalar phase(c);
    for (const auto &cell : this->get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          values.reinit(cell); cell->get_dof_indices(indices);
          values[phase].get_function_values(state, phi);
          for (unsigned int i=0; i<indices.size(); ++i)
            if (this->get_fe().system_to_component_index(i).first==c)
              for (unsigned int q=0; q<rule.size(); ++q)
                result.roundoff_scale[indices[i]-offset] += std::abs(values.JxW(q)*critical_energy_densities[0]
                  * geometric_function->first_derivative(phi[q])*values[phase].value(i,q));
        }
    // Compatibility boundary for the retained endpoint experiments. The active
    // bulk-tip coupling calls the distributed operation above directly.
    internal::endpoint_sum(result.gradient, comm);
    internal::endpoint_sum(result.roundoff_scale, comm);
    if (tangent)
      {
        const auto local = *hessian;
        hessian->clear();
        for (const auto &part : Utilities::MPI::all_gather(comm, local))
          for (const auto &entry : part) (*hessian)[entry.first] += entry.second;
      }
    return result;
  }
#define INSTANTIATE(dim) \
  template double PhaseFieldHandler<dim>::assemble_dg_fracture(const LinearAlgebra::BlockVector &,LinearAlgebra::Vector &,LinearAlgebra::SparseMatrix *) const; \
  template PhaseFieldHandler<dim>::CellFractureEnergy PhaseFieldHandler<dim>::evaluate_dg_fracture(const LinearAlgebra::BlockVector &,bool,std::map<std::pair<unsigned int,unsigned int>,double> *) const;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
