/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#pragma once
#include <aspect/material_model/phase_field_fault/bulk_cohesive_coulomb.h>
#include <deal.II/base/point.h>
#include <deal.II/grid/cell_id.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <map>

namespace aspect
{
  namespace MaterialModel
  {
    namespace BulkCohesive
    {
      namespace SlipGradient
      {
        using dealii::CellId;
        using dealii::Point;
        using dealii::Vector;
        using dealii::SparseMatrix;
        using dealii::SparsityPattern;
        /** Explicit SI length for the bounded default-off benchmark route. Captured
         * by the accepted owner and checkpoint material tuple. Default is zero. */
        double experimental_length();
        /** Local projection onto the closed-contact admissible force set (Pa). */
        struct Projection
        {
          Tensor force, input, stress;
          double pressure = 0., mu = 0., cohesion = 0.;
          unsigned int active = 0;
          int corner_sign = 1;
        };
        /** Evaluate the local force projection without changing input/stress. */
        Projection project(const Tensor &input, const Tensor &stress, double pressure, double mu, double cohesion);
        /** Differentiate the selected projection branch. */
        Tensor projection_action(const Projection &response, const Tensor &input_variation,
                                 const Tensor &stress_variation, double pressure_variation);

        /** Coupled local return and slip-gradient residual, unpublished. */
        struct JointResponse
        {
          CoulombReturn::Response local;
          Projection projection;
          Tensor residual;
          double force_scale = 1., b = 0.;
        };
        /** Fixed incoming history; increment/K/X define the current joint iterate. */
        JointResponse joint_point(const CoulombReturn::Parameters &parameters, const TensorReturn::State &incoming,
                                  const Tensor &increment, const Tensor &K, const Tensor &X,
                                  double dt, double pressure);
        /** Directional physical stress/residual action of the joint return. */
        struct JointAction
        {
          Tensor stress, residual;
        };
        /** Differentiate that iterate with dt and incoming history fixed. */
        JointAction joint_action(const CoulombReturn::Parameters &parameters, const JointResponse &response,
                                 const Tensor &increment_variation, const Tensor &K_variation,
                                 const Tensor &X_variation, double dt, double pressure_variation);

        /** Fixed Cartesian auxiliary H1 tensor space: its nodal values are the
         * actual native history-point K records. Linear extrapolation to the bulk
         * boundary preserves affine fields; zero microtraction is NATURAL, not a
         * K=0 skin. The constitutive mass is native JxW, while gradient stiffness
         * integrates this continuous bilinear interpolant exactly. */
        class Space
        {
          public:
            /** Native history quadrature point; coordinates m, weight m^2. */
            struct Sample
            {
              CellId cell;
              unsigned int q;
              Point<2> point;
              double weight;
            };
            Space(const std::vector<Sample> &samples, const Point<2> &minimum, const Point<2> &maximum);
            std::map<CellId, std::vector<unsigned int>> ids; ///< Native cell/QP to nodal index.
            std::vector<Point<2>> points;
            Vector<double> mass; ///< Native integration weights.
            SparseMatrix<double> stiffness; ///< Bilinear-gradient operator.
            double coordinate_error = 0.;
            std::array<unsigned int, 2> axis_sizes {{0, 0}};
            /** Return slip-gradient force for K; G in Pa, length in m. */
            std::vector<Tensor> apply(const std::vector<Tensor> &K, double G, double length) const;
            /** Integrated gradient energy (J/m in 2-D), with the same stiffness. */
            double energy(const std::vector<Tensor> &K, double G, double length) const;
          private:
            SparsityPattern sparsity;
        };
      }
    }
  }
}
