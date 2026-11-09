/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_FE_DGQ_BERNSTEIN_H
#define ASPECT_FE_DGQ_BERNSTEIN_H

#include <deal.II/base/polynomials_bernstein.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_tools.templates.h>

namespace aspect
{
  /** Cell-local tensor Bernstein basis. The polynomial space and DG topology
   * are FE_DGQ's; coefficients are not values at interpolation points.
   * Generalized interpolation uses the same Lobatto interpolant as FE_DGQ.
   * Embedding and additive L2 restriction reuse FE_DGQ's polynomial machinery.
   * Currently supports full-dimensional cells (the inherited codimension
   * transfer implementation substitutes a nodal element).
   */
  template <int dim, int spacedim = dim>
  class FE_DGQBernstein : public dealii::FE_DGQ<dim, spacedim>
  {
    public:
      explicit FE_DGQBernstein(const unsigned int degree)
        : dealii::FE_DGQ<dim, spacedim>(stable_basis(degree))
      {
        AssertThrow(dim == spacedim, dealii::ExcMessage("DG Bernstein transfer requires dim == spacedim."));
        const dealii::FE_DGQ<dim, spacedim> nodal(degree);
        this->generalized_support_points = nodal.get_unit_support_points();
        interpolation_inverse.reinit(this->n_dofs_per_cell(), this->n_dofs_per_cell());
        for (unsigned int i = 0; i < this->n_dofs_per_cell(); ++i)
          for (unsigned int j = 0; j < this->n_dofs_per_cell(); ++j)
            interpolation_inverse(i,j) = this->shape_value(j, this->generalized_support_points[i]);
        interpolation_inverse.gauss_jordan();
        // A factory owns no element and does not recurse into this constructor.
        static const bool registered = []()
        {
          dealii::FETools::add_fe_name<dim, spacedim>("FE_DGQBernstein",
            new dealii::FETools::FEFactory<FE_DGQBernstein<dim, spacedim>>());
          return true;
        }();
        (void)registered;
      }

      std::string get_name() const override
      {
        return "FE_DGQBernstein<" + dealii::Utilities::dim_string(dim, spacedim)
               + ">(" + std::to_string(this->degree) + ")";
      }

      std::unique_ptr<dealii::FiniteElement<dim, spacedim>> clone() const override
      {
        return std::make_unique<FE_DGQBernstein<dim, spacedim>>(*this);
      }

      void convert_generalized_support_point_values_to_dof_values(
        const std::vector<dealii::Vector<double>> &values,
        std::vector<double> &coefficients) const override
      {
        AssertDimension(values.size(), this->n_dofs_per_cell());
        AssertDimension(coefficients.size(), this->n_dofs_per_cell());
        for (unsigned int i = 0; i < coefficients.size(); ++i)
          {
            coefficients[i] = 0.;
            for (unsigned int j = 0; j < values.size(); ++j)
              {
                AssertDimension(values[j].size(), 1);
                coefficients[i] += interpolation_inverse(i,j) * values[j][0];
              }
          }
      }

      void get_interpolation_matrix(const dealii::FiniteElement<dim, spacedim> &source,
                                    dealii::FullMatrix<double> &matrix) const override
      {
        // FE_DGQ's implementation indexes ordinary unit_support_points. Use
        // generalized samples here instead; never pretend these DoFs are nodal.
        AssertDimension(matrix.m(), this->n_dofs_per_cell());
        AssertDimension(matrix.n(), source.n_dofs_per_cell());
        dealii::FullMatrix<double> samples(this->n_dofs_per_cell(), source.n_dofs_per_cell());
        for (unsigned int i = 0; i < samples.m(); ++i)
          for (unsigned int j = 0; j < samples.n(); ++j)
            samples(i,j) = source.shape_value(j, this->generalized_support_points[i]);
        interpolation_inverse.mmult(matrix, samples);
      }

      bool has_support_on_face(const unsigned int index, const unsigned int face) const override
      {
        AssertIndexRange(index, this->n_dofs_per_cell());
        AssertIndexRange(face, dealii::GeometryInfo<dim>::faces_per_cell);
        unsigned int coordinate = index;
        for (unsigned int d = 0; d < face/2; ++d) coordinate /= this->degree+1;
        return coordinate % (this->degree+1) == (face%2 ? this->degree : 0);
      }

    private:
      /** The upstream generator expands powers, losing positivity near x=1.
       * Use its normalization with Polynomial's existing root-product evaluator.
       * Repeated roots at 0/1 are valid: the distinguished normalization point
       * is 1/2, different from every root. Derivatives use the same evaluator.
       */
      static std::vector<dealii::Polynomials::Polynomial<double>> stable_basis(const unsigned int degree)
      {
        auto polynomials = dealii::generate_complete_bernstein_basis<double>(degree);
        for (unsigned int i = 0; i <= degree; ++i)
          {
            std::vector<dealii::Point<1>> roots(degree+1, dealii::Point<1>(1.));
            roots[0][0] = .5;
            for (unsigned int j = 1; j <= i; ++j) roots[j][0] = 0.;
            dealii::Polynomials::Polynomial<double> product(roots, 0);
            product *= polynomials[i].value(.5);
            polynomials[i] = product;
          }
        return polynomials;
      }

      dealii::FullMatrix<double> interpolation_inverse;
  };
}
#endif
