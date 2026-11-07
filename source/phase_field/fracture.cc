/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/phase_field.h>
#include <aspect/phase_field/dg_operator.h>
#include <aspect/reconstructed_fault/endpoint_mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  template <int dim>
  typename PhaseFieldHandler<dim>::CellFractureEnergy
  PhaseFieldHandler<dim>::evaluate_dg_fracture(const LinearAlgebra::BlockVector &state, bool tangent,
                                               std::map<std::pair<unsigned int, unsigned int>, double> *hessian) const
  {
    AssertThrow(!tangent || hessian,
                ExcMessage("DG phase fracture uses sparse assembly, not a replicated dense Hessian."));
    if (hessian) hessian->clear();
    for (const auto a : critical_energy_densities)
      AssertThrow(a == critical_energy_densities[0], ExcMessage("Coupled DG phase currently requires homogeneous fracture coefficients."));
    const double A = critical_energy_densities[0], ell = geometric_function->get_length_scale(), k = 2 * A * ell * ell;
    const auto &variable = this->introspection().variable("phase_field");
    const auto b = variable.block_index, c = variable.first_component_index;
    const auto offset = state.get_block_indices().block_start(b);
    const auto &sets = this->introspection().index_sets;
    const auto comm = this->get_mpi_communicator();
    PhaseField::DGOperator<dim> op(this->get_mapping(), this->get_dof_handler(), c, offset,
                                   sets.system_partitioning[b], sets.system_relevant_partitioning[b], comm);
    LinearAlgebra::SparseMatrix matrix;
    op.reinit_matrix(matrix);
    LinearAlgebra::Vector rhs(sets.system_partitioning[b], comm), values(rhs), action(rhs);
    const double constant = op.assemble_sipg(matrix, rhs, [&](const Point<dim> &)
    {
      return k;
    },
    this->get_parameters().discontinuous_penalty);
    values = state.block(b);
    matrix.vmult(action, values);
    double energy = .5 * (values * action) - values * rhs + constant;
    action -= rhs;
    CellFractureEnergy result;
    const auto n = state.block(b).size();
    result.gradient.assign(n, 0.);
    result.roundoff_scale.assign(n, 0.);
    for (const auto i : sets.system_partitioning[b])
      {
        result.gradient[i] = action[i];
        for (auto entry = matrix.begin(i); entry != matrix.end(i); ++entry)
          {
            result.roundoff_scale[i] += std::abs(entry->value() * state.block(b)[entry->column()]);
            if (tangent) (*hessian)[ {i, entry->column()}] += entry->value();
          }
      }
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
                  result.gradient[row] += term;
                  result.roundoff_scale[row] += std::abs(term);
                  if (tangent)
                    for (unsigned int d = 0; d < element.n_dofs_per_cell(); ++d)
                      {
                        const auto j = this->get_fe().component_to_system_index(c, d);
                        (*hessian)[ {row, indices[j] - offset}] += fe.JxW(q) * A * geometric_function->second_derivative(phi[q])
                                                                   * fe[phase].value(i, q) * fe[phase].value(j, q);
                      }
                }
            }
        }
    result.energy = energy + Utilities::MPI::sum(local_energy, comm);
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
  template PhaseFieldHandler<dim>::CellFractureEnergy PhaseFieldHandler<dim>::evaluate_dg_fracture(const LinearAlgebra::BlockVector &,bool,std::map<std::pair<unsigned int,unsigned int>,double> *) const;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
