/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_ASSEMBLERS_SCALAR_DG_H
#define ASPECT_ASSEMBLERS_SCALAR_DG_H

#include <aspect/simulator/assemblers/interface.h>
#include <functional>

namespace aspect
{
  namespace Assemblers
  {
    enum class ScalarAdvectionForm { advective, conservative_with_divergence };

    /** Spatial coefficients, with time integration deliberately external. A
     * thermal caller supplies capacity rho*c_p+L and its historical trace
     * penalty; fracture supplies capacity=1 and its gradient coefficient. */
    template <int dim>
    struct ScalarFaceCoefficients
    {
      Tensor<1,dim> velocity;
      double capacity = 1., diffusivity = 0., penalty = 0.;
    };

    inline double scalar_symmetric_penalty(const double test_jump,
                                           const double test_normal_flux,
                                           const double trial_jump,
                                           const double trial_normal_flux,
                                           const double penalty)
    {
      return -test_normal_flux * trial_jump - trial_normal_flux * test_jump
             + penalty * test_jump * trial_jump;
    }

    /** Add one quadrature point's mass, material advection and diffusion.
     * Ordinary fields provide their BE/BDF2 scales and capacity; spatial callers
     * select one term with unit scale. SUPG and forcing stay with their owners. */
    template <int dim>
    void assemble_scalar_cell_point(const std::vector<double> &values,
                                    const std::vector<Tensor<1,dim>> &gradients,
                                    const Tensor<1,dim> &velocity, const double divergence,
                                    const double weight, const double mass,
                                    const double advection, const double diffusion,
                                    const double capacity, const ScalarAdvectionForm form,
                                    FullMatrix<double> &matrix)
    {
      for (unsigned int i = 0; i < values.size(); ++i)
        for (unsigned int j = 0; j < values.size(); ++j)
          {
            const double advective = form == ScalarAdvectionForm::advective
              ? values[i] * (velocity * gradients[j])
              : -values[j] * (velocity * gradients[i] + divergence * values[i]);
            matrix(i,j) += (diffusion * (gradients[i] * gradients[j])
                            + (advection * advective + mass * values[i] * values[j]) * capacity) * weight;
          }
    }

    /** Insert the three neighbor blocks produced by the ordinary DG face
     * assemblers. Both implicit block systems and spatial scalar matrices use
     * this path, including off-process rows and subsequent compress(add). */
    template <int dim, class Matrix>
    void distribute_scalar_dg_faces(const internal::Assembly::CopyData::AdvectionSystem<dim> &data,
                                    Matrix &matrix)
    {
      for (unsigned int f = 0; f < data.assembled_matrices.size(); ++f)
        if (data.assembled_matrices[f])
          {
            for (unsigned int i = 0; i < data.local_dof_indices.size(); ++i)
              for (unsigned int j = 0; j < data.neighbor_dof_indices[f].size(); ++j)
                {
                  matrix.add(data.local_dof_indices[i], data.neighbor_dof_indices[f][j], data.local_matrices_int_ext[f](i,j));
                  matrix.add(data.neighbor_dof_indices[f][j], data.local_dof_indices[i], data.local_matrices_ext_int[f](j,i));
                }
            for (unsigned int i = 0; i < data.neighbor_dof_indices[f].size(); ++i)
              for (unsigned int j = 0; j < data.neighbor_dof_indices[f].size(); ++j)
                matrix.add(data.neighbor_dof_indices[f][i], data.neighbor_dof_indices[f][j], data.local_matrices_ext_ext[f](i,j));
          }
    }
  }

  namespace internal
  {
    namespace Assembly
    {
      namespace Scratch
      {
        /** Borrowed volume data for M, A or K, before any time discretization.
         * A represents +v.grad(phi); the conservative form includes -phi div v.
         * Scales permit mass-only, advection-only and diffusion-only assembly. */
        template <int dim>
        struct ScalarCell : ScratchBase<dim>
        {
          explicit ScalarCell(const FEValues<dim> &values) : values(values) {}
          const FEValues<dim> &values;
          std::vector<Tensor<1,dim>> velocity;
          std::vector<double> divergence, diffusivity;
          double mass = 0., advection = 0., diffusion = 0.;
          Assemblers::ScalarAdvectionForm form = Assemblers::ScalarAdvectionForm::advective;
        };

        /** Spatial face request consumed by AdvectionSystem{Interior,Boundary}Face.
         * The same traversal, quadrature loop and four-block copy data serve
         * ordinary implicit fields, explicit transport and fracture diffusion.
         * Coefficient and boundary policies carry no solver or history ownership. */
        template <int dim>
        struct ScalarFace : ScratchBase<dim>
        {
          using Cell = typename DoFHandler<dim>::active_cell_iterator;
          using Coefficients = std::vector<Assemblers::ScalarFaceCoefficients<dim>>;
          ScalarFace(const Mapping<dim> &mapping, const FiniteElement<dim> &element,
                     const Quadrature<dim-1> &rule, unsigned int component,
                     const std::vector<unsigned int> &shape_indices);
          /** Borrow the ordinary WorkStream scratch, avoiding FEValues allocation
           * and duplicate owner-face evaluation for every visited face. */
          ScalarFace(const Mapping<dim> &mapping, const FiniteElement<dim> &element,
                     unsigned int component, const std::vector<unsigned int> &shape_indices,
                     FEFaceValues<dim> &face, FEFaceValues<dim> &neighbor,
                     FESubfaceValues<dim> &subface);
          const Mapping<dim> &mapping;
          const FiniteElement<dim> &element;
          const unsigned int component;
          const std::vector<unsigned int> shape_indices;
          std::unique_ptr<FEFaceValues<dim>> owned_face, owned_neighbor;
          std::unique_ptr<FESubfaceValues<dim>> owned_subface;
          FEFaceValues<dim> &face, &neighbor_face;
          FESubfaceValues<dim> &subface;
          bool owner_face_is_prepared = false;
          double scale = 1.;
          Assemblers::ScalarAdvectionForm form = Assemblers::ScalarAdvectionForm::advective;
          std::function<std::vector<types::global_dof_index>(const Cell &)> indices;
          std::function<Coefficients(const FEFaceValuesBase<dim> &, const Cell &, unsigned int, bool)> coefficients;
          // No value means natural diffusion and no prescribed transport inflow.
          std::function<double(types::boundary_id, const Point<dim> &)> boundary_value;
          // Optional observer for explicit stage-time loads and physical fluxes.
          std::function<void(const Cell &, const FEFaceValuesBase<dim> &, unsigned int,
                             const Vector<double> &, const Vector<double> &, double)> boundary_load;
          // Local-row diagnostic, also visited when the neighbor assembles the face.
          std::function<void(const Cell &, unsigned int, const FEFaceValuesBase<dim> &,
                             const Coefficients &, unsigned int)> observe;
          double boundary_energy = 0.;
        };
      }
    }
  }
}
#endif
