/* Independent scalar-energy quadrature for the mapped SIPG test.
 * This evaluates traces/gradients directly, without the assembled matrix or
 * scalar_symmetric_penalty kernel. Interior faces contribute half on each side. */
#include <aspect/phase_field/mapped_face.h>

    template <int dim>
    class ReferenceFaceScratch
    {
      public:
        ReferenceFaceScratch(const Mapping<dim> &mapping, const FiniteElement<dim> &element,
                          const Quadrature<dim-1> &rule)
          : face(mapping, element, rule, update_values | update_gradients |
                 update_quadrature_points | update_normal_vectors | update_JxW_values),
          subface(mapping, element, rule, update_values | update_gradients |
                  update_quadrature_points | update_normal_vectors | update_JxW_values)
        {}

        template <class Iterator>
        const FEFaceValuesBase<dim> &reinit(const Iterator &cell, unsigned int f,
                                            unsigned int sub = numbers::invalid_unsigned_int)
        {
          const typename Triangulation<dim>::cell_iterator native(cell);
          if (sub == numbers::invalid_unsigned_int)
            {
              face.reinit(native, f);
              return face;
            }
          subface.reinit(native, f, sub);
          return subface;
        }

      private:
        FEFaceValues<dim> face;
        FESubfaceValues<dim> subface;
    };

std::array<double,3> check_sipg_energy(
  const dealii::Mapping<2> &mapping, const dealii::DoFHandler<2> &dofs,
  const aspect::LinearAlgebra::SparseMatrix &matrix,
  const aspect::LinearAlgebra::Vector &boundary_rhs, const double boundary_constant,
  const aspect::PhaseField::DGOperator<2>::Coefficient &coefficient,
  const aspect::PhaseField::DGOperator<2>::Coefficient &dirichlet)
{
  using namespace aspect;
  using VectorType = aspect::LinearAlgebra::Vector;
  const auto comm = MPI_COMM_WORLD;
  const auto owned = dofs.locally_owned_dofs();
  IndexSet relevant;
  DoFTools::extract_locally_relevant_dofs(dofs, relevant);
  const auto &element = dofs.get_fe();
  const QGauss<2> volume_rule(5);
  const QGauss<1> face_rule(5);
  FEValues<2> volume(mapping, element, volume_rule,
                     update_gradients | update_quadrature_points | update_JxW_values);
  ReferenceFaceScratch<2> face_scratch(mapping, element, face_rule);
  const auto energy = [&](const VectorType &state)
  {
    VectorType ghost(owned, relevant, comm);
    ghost = state;
    double local = 0.;
    std::vector<Tensor<1,2>> gradients(volume_rule.size());
    for (const auto &cell : dofs.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          volume.reinit(cell);
          volume.get_function_gradients(ghost, gradients);
          for (unsigned int q = 0; q < volume_rule.size(); ++q)
            local += .5 * volume.JxW(q) * coefficient(volume.quadrature_point(q)) * gradients[q].norm_square();
          const double jacobian = PhaseField::minimum_cell_jacobian(mapping, element, cell);
          for (const auto f : cell->face_indices())
            {
              const bool interior = !cell->at_boundary(f);
              std::vector<DoFHandler<2>::active_cell_iterator> neighbors;
              if (!interior) neighbors.emplace_back();
              else if (cell->neighbor(f)->has_children())
                for (unsigned int subface = 0; subface < 2; ++subface)
                  neighbors.push_back(cell->neighbor_child_on_subface(f, subface));
              else neighbors.push_back(cell->neighbor(f));
              for (unsigned int patch = 0; patch < neighbors.size(); ++patch)
                {
                  const auto &face = face_scratch.reinit(cell, f,
                                                         neighbors.size() > 1 ? patch : numbers::invalid_unsigned_int);
                  std::vector<double> values(face_rule.size()), other_values(face_rule.size());
                  std::vector<Tensor<1,2>> normal_gradients(face_rule.size()), other_gradients(face_rule.size());
                  std::vector<types::global_dof_index> indices(element.n_dofs_per_cell());
                  cell->get_dof_indices(indices);
                  for (unsigned int q = 0; q < face_rule.size(); ++q)
                    for (unsigned int i = 0; i < indices.size(); ++i)
                      {
                        values[q] += ghost[indices[i]] * face.shape_value(i,q);
                        normal_gradients[q] += static_cast<double>(ghost[indices[i]]) * face.shape_grad(i,q);
                      }
                  double other_jacobian = jacobian;
                  if (interior)
                    {
                      const auto &neighbor = neighbors[patch];
                      std::vector<Point<2>> points;
                      for (const auto &point : face.get_quadrature_points())
                        points.push_back(mapping.transform_real_to_unit_cell(neighbor, point));
                      FEValues<2> other(mapping, element, Quadrature<2>(points), update_values | update_gradients);
                      other.reinit(neighbor);
                      other.get_function_values(ghost, other_values);
                      other.get_function_gradients(ghost, other_gradients);
                      other_jacobian = PhaseField::minimum_cell_jacobian(mapping, element, neighbor);
                    }
                  for (unsigned int q = 0; q < face_rule.size(); ++q)
                    {
                      const auto point = face.quadrature_point(q);
                      const double k = coefficient(point);
                      const double surface_jacobian = face.JxW(q) / face_rule.weight(q);
                      const double other_scale = interior && cell->neighbor_is_coarser(f) ? 2. : 1.;
                      const double penalty = 90. * k * std::max(surface_jacobian * neighbors.size() / jacobian,
                                                                surface_jacobian * other_scale / other_jacobian);
                      const double jump = values[q] - (interior ? other_values[q] : dirichlet(point));
                      const auto gradient = interior ? .5 * (normal_gradients[q] + other_gradients[q]) : normal_gradients[q];
                      local += (interior ? .5 : 1.) * face.JxW(q)
                               * (-k * (gradient * face.normal_vector(q)) * jump + .5 * penalty * jump * jump);
                    }
                }
            }
        }
    return dealii::Utilities::MPI::sum(local, comm);
  };
  VectorType x(owned, comm), direction(owned, comm), ax(owned, comm), ad(owned, comm), trial(owned, comm);
  for (const auto i : owned)
    {
      x[i] = .3 + .05 * std::sin(1.3*i);
      direction[i] = std::cos(.7*i);
    }
  x.compress(VectorOperation::insert);
  direction.compress(VectorOperation::insert);
  matrix.vmult(ax, x);
  matrix.vmult(ad, direction);
  const double direct = energy(x);
  const double assembled = .5 * (x*ax) - x*boundary_rhs + boundary_constant;
  const double first = direction*ax - direction*boundary_rhs;
  const double second = direction*ad;
  const double epsilon = 1.e-3;
  trial = x;
  trial.add(epsilon, direction);
  const double plus = energy(trial);
  trial = x;
  trial.add(-epsilon, direction);
  const double minus = energy(trial);
  const std::array<double,3> errors = {{
      std::abs(direct-assembled) / std::max(1.,std::abs(assembled)),
      std::abs((plus-minus)/(2.*epsilon)-first) / std::max(1.,std::abs(first)),
      std::abs((plus-2.*direct+minus)/(epsilon*epsilon)-second) / std::max(1.,std::abs(second))
    }
  };
  AssertThrow(errors[0] < 1.e-12 && errors[1] < 1.e-9 && errors[2] < 1.e-7,
              ExcMessage("Mapped SIPG matrix/residual/tangent disagree with independent energy quadrature."));
  return errors;
}
