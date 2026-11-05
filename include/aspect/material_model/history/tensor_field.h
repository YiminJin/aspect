/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_MATERIAL_MODEL_TENSOR_FIELD_H
#define ASPECT_MATERIAL_MODEL_TENSOR_FIELD_H

#include <aspect/simulator_access.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

#include <array>
#include <functional>
#include <map>

namespace aspect
{
  namespace MaterialModel
  {
    namespace HistoryDiscretization
    {
      /**
       * Numerical symmetric-tensor field on a shared discontinuous Q2 space.
       *
       * This component assigns no constitutive meaning, units, objective frame,
       * or acceptance status to its coefficients. The material owner supplies
       * those meanings. In particular, a field does not own a residual history
       * or fault/profile observations. Copies share the space and copy vectors;
       * numerical operations return new fields without changing their inputs.
       * The currently enabled transport/projection algorithms require 2-D cells.
       */
      template <int dim>
      struct TensorField
      {
        /** Native or DG cell iterators indexed by persistent CellId. */
        using Cells = std::map<CellId, typename DoFHandler<dim>::active_cell_iterator>;

        /**
         * Shared FE infrastructure. Rebuild DoF indices and clear numerical
         * caches after mesh transfer. It contains no phase values, fault
         * associations or accepted stress samples.
         */
        struct Space
        {
          /** Distribute Q2 DoFs collectively on the supplied triangulation. */
          explicit Space(const Triangulation<dim> &triangulation)
            : element(2), dofs(triangulation)
          {
            dofs.distribute_dofs(element);
            relevant = DoFTools::extract_locally_relevant_dofs(dofs);
            for (const auto &cell : dofs.active_cell_iterators())
              if (!cell->is_artificial())
                cells.emplace(cell->id(), cell);
          }

          /** Inverse Gauss3 mass and factors used to bound projection roundoff. */
          struct SourceGeometry
          {
            /** Local inverse mass, in the element's DoF order. */
            FullMatrix<double> inverse;
            /** Extended-precision source-projection factors in that order. */
            std::vector<long double> projection_factor;
          };

          /**
           * Numerical geometry cache keyed by nine Gauss3 JxW values and cell
           * widths. Nine is the local Q2 size in the supported 2-D algorithm.
           * Prepared outside parallel mechanical assembly; not serialized.
           */
          mutable std::map<std::array<long double, 9+dim>, SourceGeometry> source_geometry;
          /** Discontinuous Q2 tensor-component element. */
          FE_DGQ<dim> element;
          /** Tensor DoFs, distinct from the simulator's mixed FE DoFs. */
          DoFHandler<dim> dofs;
          /** Owned DoFs plus readable neighbor ghosts. */
          IndexSet relevant;
          /** All non-artificial tensor cells, including readable ghosts. */
          Cells cells;
        };

        /** Shared space; must outlive coefficient evaluation and transfer. */
        std::shared_ptr<Space> space;
        /**
         * One vector per independent in-plane component, in deal.II ordering.
         * Published/returned vectors have locally owned values and readable
         * ghosts. Assembly first writes separate owned vectors and compresses
         * them collectively before filling these vectors.
         */
        std::array<LinearAlgebra::Vector,
            SymmetricTensor<2, dim>::n_independent_components> coefficients;

        /**
         * Evaluate raw stored components at a reference-cell point. @p indices
         * are that cell's nine Q2 DoFs, readable in the coefficient vectors.
         * No physical reconstruction, plane-strain completion or mutation occurs.
         */
        SymmetricTensor<2, dim>
        tensor_at(const std::vector<types::global_dof_index> &indices,
                  const Point<dim> &unit_point) const
        {
          SymmetricTensor<2, dim> value;
          for (unsigned int component = 0; component < coefficients.size(); ++component)
            for (unsigned int i = 0; i < 9; ++i)
              value[SymmetricTensor<2, dim>::unrolled_to_component_indices(component)]
              += coefficients[component][indices[i]] * space->element.shape_value(i, unit_point);
          return value;
        }

        /**
         * Evaluator for whole-field mass projection. Arguments are the owned
         * cell identifier, its tensor DoF indices, initialized target FEValues,
         * and a pre-sized output array (one tensor per quadrature point).
         * The callback fills all entries; it may sample its own immutable FE
         * inputs but must not publish state or invoke MPI collectives. It is
         * called synchronously, once per cell, before that cell's mass assembly.
         */
        using CellEvaluator = std::function<void(
                                const CellId &cell_id,
                                const std::vector<types::global_dof_index> &indices,
                                const FEValues<dim> &target_values,
                                std::vector<SymmetricTensor<2, dim>> &values)>;

        /** Multiple whole fields, indexed [field][quadrature point]. The same
         * callback/lifetime/collective rules as CellEvaluator apply. */
        using MultiCellEvaluator = std::function<void(
                                     const CellId &cell_id,
                                     const std::vector<types::global_dof_index> &indices,
                                     const FEValues<dim> &target_values,
                                     std::vector<std::vector<SymmetricTensor<2, dim>>> &values)>;

        /** Project n_fields together, assembling/inverting one cell mass matrix.
         * Every field uses prototype's space and the same supplied quadrature.
         * Output order is evaluator field order; all ranks pass the same count. */
        static std::vector<std::shared_ptr<TensorField>>
        project(const SimulatorAccess<dim> &context,
                const TensorField &prototype,
                const Cells &owned_cells,
                const Quadrature<dim> &quadrature,
                unsigned int n_fields,
                const MultiCellEvaluator &evaluate_cell);

        /**
         * L2-project an evaluated tensor field using the supplied @p quadrature.
         * Only the keys of @p owned_cells select cells; @p prototype supplies
         * the tensor space. All ranks participate, including empty ranks.
         * Return owned/ghost coefficients in that same space. Inputs and their
         * lifetimes remain caller-owned; no callback or FEValues is retained.
         */
        static std::shared_ptr<TensorField>
        project(const SimulatorAccess<dim> &context,
                const TensorField &prototype,
                const Cells &owned_cells,
                const Quadrature<dim> &quadrature,
                const CellEvaluator &evaluate_cell);
      };
    }
  }
}
#endif
