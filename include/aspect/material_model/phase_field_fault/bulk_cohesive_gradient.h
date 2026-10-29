/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#pragma once
#include <aspect/material_model/phase_field_fault/bulk_cohesive_coulomb.h>
#include <deal.II/base/point.h>
#include <deal.II/grid/cell_id.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <map>

namespace aspect { namespace MaterialModel { namespace BulkCohesive { namespace SlipGradient
{
  using dealii::CellId;
  using dealii::Point;
  using dealii::Vector;
  using dealii::SparseMatrix;
  using dealii::SparsityPattern;
  // Explicit SI length for the bounded default-off benchmark route. It is
  // captured in the accepted owner and included in its checkpoint material
  // tuple. No production/default path supplies a nonzero value.
  double experimental_length();
  struct Projection
  {
    Tensor force,input,stress;
    double pressure=0.,mu=0.,cohesion=0.;
    unsigned int active=0;
    int corner_sign=1;
  };
  Projection project(const Tensor &,const Tensor &,double pressure,double mu,double cohesion);
  Tensor projection_action(const Projection &,const Tensor &,const Tensor &,double);

  struct JointResponse
  {
    CoulombReturn::Response local;
    Projection projection;
    Tensor residual;
    double force_scale=1.,b=0.;
  };
  JointResponse joint_point(const CoulombReturn::Parameters &,const TensorReturn::State &,
                            const Tensor &increment,const Tensor &K,const Tensor &X,
                            double dt,double pressure);
  struct JointAction {Tensor stress,residual;};
  JointAction joint_action(const CoulombReturn::Parameters &,const JointResponse &,
                           const Tensor &increment_variation,const Tensor &K_variation,
                           const Tensor &X_variation,double dt,double pressure_variation);

  /** Fixed Cartesian auxiliary H1 tensor space: its nodal values are the
   * actual native history-point K records. Linear extrapolation to the bulk
   * boundary preserves affine fields; zero microtraction is NATURAL, not a
   * K=0 skin. The constitutive mass is native JxW, while gradient stiffness
   * integrates this continuous bilinear interpolant exactly. */
  class Space
  {
    public:
      struct Sample {CellId cell;unsigned int q;Point<2> point;double weight;};
      Space(const std::vector<Sample> &,const Point<2> &minimum,const Point<2> &maximum);
      std::map<CellId,std::vector<unsigned int>> ids;
      std::vector<Point<2>> points;
      Vector<double> mass;
      SparseMatrix<double> stiffness;
      double coordinate_error=0.;
      std::array<unsigned int,2> axis_sizes{{0,0}};
      std::vector<Tensor> apply(const std::vector<Tensor> &,double G,double length) const;
      double energy(const std::vector<Tensor> &,double G,double length) const;
    private:
      SparsityPattern sparsity;
  };
}}}}
