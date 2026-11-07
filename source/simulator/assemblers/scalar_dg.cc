/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/simulator/assemblers/advection.h>
#include <array>

namespace aspect
{
  namespace internal
  {
    namespace Assembly
    {
      namespace Scratch
      {
        template <int dim>
        ScalarFace<dim>::ScalarFace(const Mapping<dim> &mapping_, const FiniteElement<dim> &element_,
                                    const Quadrature<dim-1> &rule, unsigned int component_,
                                    const std::vector<unsigned int> &shape_indices_)
          : mapping(mapping_), element(element_), component(component_), shape_indices(shape_indices_),
            owned_face(std::make_unique<FEFaceValues<dim>>(mapping,element,rule,
                                                            update_values | update_gradients | update_quadrature_points | update_normal_vectors | update_JxW_values)),
            owned_neighbor(std::make_unique<FEFaceValues<dim>>(mapping,element,rule,
                                                                update_values | update_gradients | update_quadrature_points | update_normal_vectors | update_JxW_values)),
            owned_subface(std::make_unique<FESubfaceValues<dim>>(mapping,element,rule,
                                                                  update_values | update_gradients | update_quadrature_points | update_normal_vectors | update_JxW_values)),
            face(*owned_face), neighbor_face(*owned_neighbor), subface(*owned_subface)
        {}

        template <int dim>
        ScalarFace<dim>::ScalarFace(const Mapping<dim> &mapping_, const FiniteElement<dim> &element_,
                                    unsigned int component_, const std::vector<unsigned int> &shape_indices_,
                                    FEFaceValues<dim> &face_, FEFaceValues<dim> &neighbor_, FESubfaceValues<dim> &subface_)
          : mapping(mapping_), element(element_), component(component_), shape_indices(shape_indices_),
            face(face_), neighbor_face(neighbor_), subface(subface_), owner_face_is_prepared(true)
        {}
      }
    }
  }

  namespace Assemblers
  {
    template <int dim>
    void AdvectionSystem<dim>::assemble_spatial(internal::Assembly::Scratch::ScalarCell<dim> &scratch,
                                                internal::Assembly::CopyData::AdvectionSystem<dim> &data)
    {
      const auto &fe = scratch.values;
      std::vector<double> values(data.local_matrix.m());
      std::vector<Tensor<1,dim>> gradients(values.size());
      for (unsigned int q = 0; q < fe.n_quadrature_points; ++q)
        {
          for (unsigned int i = 0; i < values.size(); ++i)
            {
              values[i] = fe.shape_value(i,q);
              if (scratch.advection != 0. || scratch.diffusion != 0.) gradients[i] = fe.shape_grad(i,q);
            }
          assemble_scalar_cell_point(values,gradients,
                                     scratch.advection != 0. ? scratch.velocity[q] : Tensor<1,dim>(),
                                     scratch.advection != 0. && scratch.form == ScalarAdvectionForm::conservative_with_divergence ? scratch.divergence[q] : 0.,
                                     fe.JxW(q),scratch.mass,scratch.advection,
                                     scratch.diffusion != 0. ? scratch.diffusion*scratch.diffusivity[q] : 0.,1.,scratch.form,data.local_matrix);
        }
    }

    template <int dim>
    void AdvectionSystemBoundaryFace<dim>::assemble_spatial(internal::Assembly::Scratch::ScalarFace<dim> &scratch,
                                                            internal::Assembly::CopyData::AdvectionSystem<dim> &data)
    {
      const auto cell = scratch.cell;
      const auto f = scratch.face_number;
      if (!scratch.owner_face_is_prepared) scratch.face.reinit(cell, f);
      const auto &face = scratch.face;
      const auto coefficients = scratch.coefficients(face, cell, f, false);
      if (scratch.observe) scratch.observe(cell, f, face, coefficients, 1);
      const auto boundary = cell->face(f)->boundary_id();
      const auto n = scratch.shape_indices.size();
      for (unsigned int q = 0; q < face.n_quadrature_points; ++q)
        {
          const auto &c = coefficients[q];
          const double vn = c.velocity * face.normal_vector(q);
          const double w = scratch.scale * face.JxW(q);
          const double sigma = c.capacity * c.penalty;
          const bool prescribed = static_cast<bool>(scratch.boundary_value);
          const double g = prescribed ? scratch.boundary_value(boundary, face.quadrature_point(q)) : 0.;
          Vector<double> load(n), trace(n), normal_derivative(n);
          for (unsigned int i = 0; i < n; ++i)
            {
              trace[i] = face.shape_value_component(scratch.shape_indices[i],q,scratch.component);
              normal_derivative[i] = face.shape_grad_component(scratch.shape_indices[i],q,scratch.component) * face.normal_vector(q);
            }
          if (prescribed) scratch.boundary_energy += .5 * w * sigma * g * g;
          for (unsigned int i = 0; i < n; ++i)
            {
              const double vi = trace[i], di = normal_derivative[i];
              if (prescribed)
                load[i] = w * (sigma * vi - c.diffusivity * di - c.capacity * std::min(vn,0.) * vi);
              data.local_rhs[i] += load[i] * g;
              for (unsigned int j = 0; j < n; ++j)
                {
                  const double vj = trace[j], dj = normal_derivative[j];
                  double entry = prescribed ? scalar_symmetric_penalty(vi,c.diffusivity*di,vj,c.diffusivity*dj,sigma)
                                 - c.capacity * std::min(vn,0.) * vi * vj : 0.;
                  if (scratch.form == ScalarAdvectionForm::conservative_with_divergence)
                    entry += c.capacity * vn * vi * vj;
                  data.local_matrix(i,j) += w * entry;
                }
            }
          if (scratch.boundary_load) scratch.boundary_load(cell,face,q,load,trace,face.JxW(q)*vn);
        }
    }

    template <int dim>
    void AdvectionSystemInteriorFace<dim>::assemble_spatial(internal::Assembly::Scratch::ScalarFace<dim> &scratch,
                                                            internal::Assembly::CopyData::AdvectionSystem<dim> &data)
    {
      const auto cell = scratch.cell;
      const unsigned int f = scratch.face_number;
      const bool periodic = cell->has_periodic_neighbor(f);
      const auto neighbor = cell->neighbor_or_periodic_neighbor(f);
      const unsigned int patches = neighbor->has_children()
                                   ? neighbor->face(periodic ? cell->periodic_neighbor_face_no(f) : cell->neighbor_face_no(f))->n_children()
                                   : 1;
      // Preserve ASPECT's single-face ownership: coarse cell, or the lower
      // local index/subdomain on an equal-level face. All four blocks follow it.
      const bool assemble = neighbor->has_children() ||
                            (neighbor->level() == cell->level() && neighbor->is_active() &&
                             ((neighbor->is_locally_owned() && cell->index() < neighbor->index()) ||
                              (!neighbor->is_locally_owned() && cell->subdomain_id() < neighbor->subdomain_id())));
      if (!assemble && !scratch.observe) return;
      for (unsigned int patch = 0; patch < patches; ++patch)
        {
          const FEFaceValuesBase<dim> *face;
          if (neighbor->has_children())
            {
              scratch.subface.reinit(cell,f,patch);
              face = &scratch.subface;
            }
          else
            {
              if (!scratch.owner_face_is_prepared) scratch.face.reinit(cell,f);
              face = &scratch.face;
            }
          const auto coefficients = scratch.coefficients(*face,cell,f,patches>1);
          if (scratch.observe) scratch.observe(cell,f,*face,coefficients,patches);
          if (!assemble) continue;
          const typename DoFHandler<dim>::active_cell_iterator other = neighbor->has_children()
                                                                       ? (periodic ? cell->periodic_neighbor_child_on_subface(f,patch) : cell->neighbor_child_on_subface(f,patch))
                                                                       : typename DoFHandler<dim>::active_cell_iterator(neighbor);
          const unsigned int nf = neighbor->has_children()
                                  ? (periodic ? cell->periodic_neighbor_face_no(f) : cell->neighbor_face_no(f))
                                  : (periodic ? cell->periodic_neighbor_of_periodic_neighbor(f) : cell->neighbor_of_neighbor(f));
          scratch.neighbor_face.reinit(other,nf);
          const auto exterior_coefficients = scratch.coefficients(scratch.neighbor_face,other,nf,false);
          const auto slot = nth_interface_matrix(scratch.element.reference_cell(),f,patch);
          data.neighbor_dof_indices[slot] = scratch.indices(other);
          data.assembled_matrices[slot] = true;

          // Match physical points, rather than assuming the same reference-face
          // orientation. Periodic ordinary fields retain their existing pairing.
          std::vector<unsigned int> permutation(face->n_quadrature_points);
          for (unsigned int q = 0; q < face->n_quadrature_points; ++q)
            {
              permutation[q] = q;
              if (!periodic)
                {
                  double distance = std::numeric_limits<double>::max();
                  for (unsigned int r = 0; r < face->n_quadrature_points; ++r)
                    if (face->quadrature_point(q).distance_square(scratch.neighbor_face.quadrature_point(r)) < distance)
                      {
                        distance = face->quadrature_point(q).distance_square(scratch.neighbor_face.quadrature_point(r));
                        permutation[q] = r;
                      }
                  AssertThrow(distance < 1e-20 * std::max(1.,cell->diameter()*cell->diameter()),
                              ExcMessage("Scalar DG face quadratures do not match."));
                }
            }
          for (unsigned int q = 0; q < face->n_quadrature_points; ++q)
            {
              const unsigned int r = permutation[q];
              const auto &a = coefficients[q];
              const auto &b = exterior_coefficients[r];
              const auto normal = face->normal_vector(q);
              const double vn = a.velocity * normal;
              const double sigma = std::max(a.capacity,b.capacity) * std::max(a.penalty,b.penalty);
              const double w = scratch.scale * face->JxW(q);
              const auto n = scratch.shape_indices.size();
              std::array<std::vector<double>,2> values, derivatives;
              for (unsigned int side = 0; side < 2; ++side)
                {
                  values[side].resize(n);
                  derivatives[side].resize(n);
                  const auto &evaluation = side == 0 ? *face : scratch.neighbor_face;
                  const auto point = side == 0 ? q : r;
                  for (unsigned int i = 0; i < n; ++i)
                    {
                      values[side][i] = evaluation.shape_value_component(scratch.shape_indices[i],point,scratch.component);
                      derivatives[side][i] = evaluation.shape_grad_component(scratch.shape_indices[i],point,scratch.component) * normal;
                    }
                }
              // Jumps are (+test,-test); both fluxes use the owner's normal.
              for (unsigned int side_i = 0; side_i < 2; ++side_i)
                for (unsigned int side_j = 0; side_j < 2; ++side_j)
                  {
                    auto &matrix = side_i == 0
                                   ? (side_j == 0 ? data.local_matrix : data.local_matrices_int_ext[slot])
                                   : (side_j == 0 ? data.local_matrices_ext_int[slot] : data.local_matrices_ext_ext[slot]);
                    const double sign_i = side_i == 0 ? 1. : -1., sign_j = side_j == 0 ? 1. : -1.;
                    const double ki = side_i == 0 ? a.diffusivity : b.diffusivity;
                    const double kj = side_j == 0 ? a.diffusivity : b.diffusivity;
                    const double capacity = side_i == 0 ? a.capacity : b.capacity;
                    const double outward = sign_i * vn;
                    for (unsigned int i = 0; i < n; ++i)
                      for (unsigned int j = 0; j < n; ++j)
                        {
                          const double vi = values[side_i][i], vj = values[side_j][j];
                          const double di = derivatives[side_i][i], dj = derivatives[side_j][j];
                          double entry = scalar_symmetric_penalty(sign_i*vi,.5*ki*di,sign_j*vj,.5*kj*dj,sigma);
                          if (outward < 0.)
                            entry += capacity * outward * (side_i == side_j ? -1. : 1.) * vi * vj;
                          if (scratch.form == ScalarAdvectionForm::conservative_with_divergence && side_i == side_j)
                            entry += capacity * outward * vi * vj;
                          matrix(i,j) += w * entry;
                        }
                  }
            }
        }
    }
  }
}

namespace aspect
{
  namespace internal
  {
    namespace Assembly
    {
      namespace CopyData
      {
        template <int dim>
        AdvectionSystem<dim>::
        AdvectionSystem (const FiniteElement<dim> &finite_element,
                         const bool                field_is_discontinuous)
          :
          local_matrix (finite_element.dofs_per_cell,
                        finite_element.dofs_per_cell),
          local_matrices_int_ext ((field_is_discontinuous
                                   ?
                                   Assemblers::n_interface_matrices(finite_element.reference_cell())
                                   :
                                   0),
                                  FullMatrix<double>(finite_element.dofs_per_cell,
                                                     finite_element.dofs_per_cell)),
          local_matrices_ext_int ((field_is_discontinuous
                                   ?
                                   Assemblers::n_interface_matrices(finite_element.reference_cell())
                                   :
                                   0),
                                  FullMatrix<double>(finite_element.dofs_per_cell,
                                                     finite_element.dofs_per_cell)),
          local_matrices_ext_ext ((field_is_discontinuous
                                   ?
                                   Assemblers::n_interface_matrices(finite_element.reference_cell())
                                   :
                                   0),
                                  FullMatrix<double>(finite_element.dofs_per_cell,
                                                     finite_element.dofs_per_cell)),
          local_rhs (finite_element.dofs_per_cell),

          assembled_matrices ((field_is_discontinuous
                               ?
                               Assemblers::n_interface_matrices(finite_element.reference_cell())
                               :
                               0), false),

          local_dof_indices (finite_element.dofs_per_cell),
          neighbor_dof_indices ((field_is_discontinuous
                                 ?
                                 Assemblers::n_interface_matrices(finite_element.reference_cell())
                                 :
                                 0),
                                std::vector<types::global_dof_index>(finite_element.dofs_per_cell))
        {}
      }
    }
  }
}
namespace aspect
{
  namespace Assemblers
  {
    unsigned int
    n_interface_matrices (const ReferenceCell &reference_cell)
    {
      // The current implementation assumes that all faces are
      // the same; so no wedges or pyramids please.
      Assert ((reference_cell == ReferenceCells::Triangle)
              ||
              (reference_cell == ReferenceCells::Quadrilateral)
              ||
              (reference_cell == ReferenceCells::Tetrahedron)
              ||
              (reference_cell == ReferenceCells::Hexahedron),
              ExcNotImplemented());
      return (reference_cell.n_faces() *
              reference_cell.face_reference_cell(0).n_isotropic_children());
    }



    unsigned int
    nth_interface_matrix (const ReferenceCell &reference_cell,
                          const unsigned int face)
    {
      AssertIndexRange (face, reference_cell.n_faces());
      return (face *
              reference_cell.face_reference_cell(0).n_isotropic_children());
    }



    unsigned int
    nth_interface_matrix (const ReferenceCell &reference_cell,
                          const unsigned int face,
                          const unsigned int sub_face)
    {
      AssertIndexRange (face, reference_cell.n_faces());
      AssertIndexRange (sub_face,
                        reference_cell.face_reference_cell(0).n_isotropic_children());
      return (face *
              reference_cell.face_reference_cell(0).n_isotropic_children()
              + sub_face);
    }




  }
}

namespace aspect
{
#define INSTANTIATE(dim) \
  template struct internal::Assembly::Scratch::ScalarFace<dim>; \
  template struct internal::Assembly::CopyData::AdvectionSystem<dim>; \
  template void Assemblers::AdvectionSystem<dim>::assemble_spatial(internal::Assembly::Scratch::ScalarCell<dim> &,internal::Assembly::CopyData::AdvectionSystem<dim> &); \
  template void Assemblers::AdvectionSystemBoundaryFace<dim>::assemble_spatial(internal::Assembly::Scratch::ScalarFace<dim> &,internal::Assembly::CopyData::AdvectionSystem<dim> &); \
  template void Assemblers::AdvectionSystemInteriorFace<dim>::assemble_spatial(internal::Assembly::Scratch::ScalarFace<dim> &,internal::Assembly::CopyData::AdvectionSystem<dim> &);
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
